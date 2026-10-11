# V5 Validation Report

Generated during V5 model evaluation. Evaluation only — no training was performed
and no training data, production configuration, or protected model artifact was
modified.

- Script: `scripts/eval_v5_validation.py` (new, additive-only)
- Validation split: `data/v5/vector_val.jsonl` (100 positive / 16 negative)
- Held-out test split `data/v5/vector_test.jsonl`: **not run** (left untouched)
- Path evaluated: Tier 0 matcher → rules-only candidate router → model → candidate admission
- Lifecycle: production `NeedleEnginePool` reset (reset + complete atomic), reused, not reimplemented

## Artifact hashes — before and after (unchanged)

| Artifact | MD5 | Status |
|---|---|---|
| `checkpoints/needle3.safetensors` | `37b800e5dbbcb24e9e8c232137c2033c` | identical |
| `checkpoints/vector_gpu_v4.safetensors` | `75e88841b7f2427bba1f7fc4262c34a9` | identical |
| `models/needle3_vector_v4.cact` | `70fa284b368ba18f8670f4eb26536b7f` | identical |
| `checkpoints/vector_gpu_v5.safetensors` | `93344fe3fcae4e3b5901bdf34de400ce` | identical |
| `models/needle3_vector_v5.cact` | `5e22d8351e00c9b7798dfad538d84e12` | identical |

The eval re-hashes each `.cact` around its own run and reports `md5 unchanged = True`.
Test split mtime `2026-10-11 08:57` — never opened.

## Validation results — `data/v5/vector_val.jsonl` (100 pos / 16 neg)

Order-independent: reversing model order reproduced identical numbers.

| Metric | V4 | V5 |
|---|---:|---:|
| Tool-name accuracy | 87/100 = **87.0%** | 85/100 = 85.0% |
| Name + arguments accuracy | 80/100 = **80.0%** | 78/100 = 78.0% |
| Abstention accuracy (negatives) | 15/16 = 93.8% | 15/16 = 93.8% |
| Fully correct (pos+neg) | 95/116 = **81.9%** | 93/116 = 80.2% |
| Admission violations | 0 | 0 |
| Untrained tool reached model | 0 | 0 |
| High-friction routed | 0 | 0 |
| Negatives that escaped | 1 | 1 |

### Per-authored-family (positives)

| Family | V4 | V5 |
|---|---:|---:|
| app_management | 25/27 (92.6%) | 26/27 (96.3%) |
| media_control | 19/24 (79.2%) | 20/24 (83.3%) |
| system_monitoring | 21/24 (87.5%) | 21/24 (87.5%) |
| volume_control | 22/25 (**88.0%**) | 18/25 (**72.0%**) |

V5 gains +1 in app and media, ties system, loses 4 in volume → net −2.

V5 volume regressions:

- `set_volume → get_volume` ×2 (`volume at 100 percent`, `bring the volume to 50`)
- `get_volume → unmute` ×2 (`what is / what's the audio level`)
- `unmute → mute` (`sound on`)
- `mute → get_volume` (`turn off the volume`)

### Shared failure modes (not fixed by V5)

- `get_battery_status → get_disk_usage`
- `media_previous → media_next`
- Abstentions (Tier 2) on `keep the album rolling`, `freeze the track`, `don't pause the album`
- Single negative escape: `'help me write a resume' → media_play`, caused by the `resume`
  file/media disambiguation rule (pre-existing; both models)

Safety architecture held for both: 0 candidate-set rejections, 0 untrained tools
offered, 0 fabrication of a hit when the engine returned nothing.

## Volume-regression diagnosis

### Comparison bias — the raw gap is mostly a measurement artifact

The V5 validation split is **not clean with respect to V4**: 50 of 116 rows are
exact `data/vector_train.jsonl` queries. `scripts/prepare_v5_dataset.py:235`
builds `excluded = burned | old_val | old_test` — V4 **val/test** are excluded
but V4 **train** is not — and V4-train rows are then carried into the V5 splits
(`prov="v4"`, lines 264-277). V4 memorized those 50 rows.

Splitting the same evaluation along that boundary (both models through the
shipped path):

| Subset | V4 fully correct | V5 fully correct |
|---|---:|---:|
| In V4 train (50 rows: 43 pos / 7 neg) | 43/50 = **86.0%** | 41/50 = 82.0% |
| Not in V4 train, "clean" (66 rows: 57 pos / 9 neg) | 52/66 = **78.8%** | 52/66 = **78.8%** |

On genuinely unseen phrasings the two models are **identical** (46/57 tool name,
43/57 name+args, 9/9 abstain). The apparent V5 deficit is confined to rows V4
memorized.

### Per-regression classification (volume family)

Routing and argument extraction are **not** the cause. Every volume row selected
the `volume_control` family (0 routing errors); negatives escalated correctly
(4/4); and every emitted `set_volume` carried the correct level (V4 8/8,
V5 6/6). All misses are wrong-member selection inside the correct family.

| Query | Expected | V4 | V5 | Classification |
|---|---|---|---|---|
| `volume at 100 percent` | set_volume{100} | set_volume{100} | get_volume | model error (V4 memorized) + matcher-coverage gap |
| `bring the volume to 50` | set_volume{50} | set_volume{50} | get_volume | model error (clean; V5 fails on an unseen phrase) + matcher-coverage gap |
| `what is the audio level` | get_volume | get_volume | unmute | model error (V4 memorized) |
| `what's the audio level` | get_volume | get_volume | unmute | model error (variant of a V4-train phrase) |
| `sound on` | unmute | no-call (abstain) | mute | model error (V5 emits a wrong action; V4 safely abstained) |
| `turn off the volume` | mute | no-call (abstain) | get_volume | both miss; V5 emits a wrong action |
| `restore the volume` | unmute | get_volume | get_volume | both miss; not a regression |

Matcher-coverage gaps in `app/core/matcher.py` send two set-volume phrasings to
the model because Tier 0 does not cover them:

- `SET_VOLUME_REGEXES` (line 137) accepts `%` but not the word **"percent"**.
- `SET_VOLUME_VERB_REGEXES` (line 144) lists `turn|raise|lower|put|bump|crank`
  but not **"bring"**.

### Dataset-design cause for the set_volume weakness

The builder caps V4-carried `set_volume` digit templates
(`scripts/prepare_v5_dataset.py:303-313`, `MAX_PER_TPL = 2`). Volume training
counts, V4 → V5:

| tool | V4 | V5 |
|---|---:|---:|
| set_volume | 123 | 39 |
| get_volume | 23 | 32 |
| mute | 15 | 24 |
| unmute | 13 | 22 |

The set_volume:get_volume ratio moved from ~5.3:1 to ~1.2:1, which directly
predicts the `set_volume → get_volume` confusion. This is evidence from counts,
independent of any loss value.

### Recommended smallest evidence-backed change

1. **Evaluation (no artifact change):** score both models on the 66 clean rows
   and report the 50 V4-carry rows separately. On the clean set V4 and V5 are
   identical; the raw 87% vs 85% should not be read as a V5 deficit.
2. **Pipeline fix (model-independent, surgical):** extend the deterministic
   Tier-0 volume rules — accept `percent|per cent` as a unit in
   `SET_VOLUME_REGEXES`, and add `bring` to the `SET_VOLUME_VERB_REGEXES` verb
   list. Verified in-memory to match exactly the two target phrases
   (`volume at 100 percent`, `bring the volume to 50`, plus `volume at 65
   percent`) and none of the read/mute/unmute phrasings. This removes 2 of the 4
   net regressions at the safest layer for both models, and is a general
   consistency fill rather than curve-fitting. Validate against the untouched
   test split before adopting.
3. **Do not promote V5** on current evidence: the clean-set comparison is a tie.
   If a future V5b is trained, rebalance the volume family (the template cap
   over-corrected); that is a training change, out of scope here.

### Evaluating a future iteration without reusing V5 val

- The V5 val split is now **used** (inspected, decisions made) and is burned for
  model selection.
- Keep `data/v5/vector_test.jsonl` untouched as the single one-time final
  measurement; do not peek.
- For iteration/selection, either: (a) add **V4 train** to the builder's
  exclusion set (fix the gap at line 235) and build a fresh validation slice
  excluded against the union {V4 train/val/test, V5 train/val/test, burned 98,
  V5 val}; or (b) cross-validate inside the training pool while excluding
  V4-carry rows from the eval folds, so any V4↔V5 comparison is always on unseen
  phrasings.
- General rule: exclude every query any candidate model trained on, or V4 will
  always win by memorization.

## Safety / regression tests

- `170 passed`, `1 failed`, `4 errors`.
- The 1 failure (`tests/test_router.py::test_intent_router_tier0_hit`) is **Linux-only env**:
  `No module named 'comtypes'`. Routing itself is correct (Tier 0 matched `set_volume`);
  the tool executor cannot run on Linux. Same class as the documented pre-existing failure.
- The 4 collection errors are **Linux-only missing desktop deps**:
  `PySide6` (`test_ui`), `pyautogui` (`test_system_tools`), `pyttsx3` (`test_voice`),
  `comtypes`/win32 (`test_app_window_tools`).
- No actual code/logic failures.

## Bottom line

Corrected by the diagnosis above: the raw split shows V4 ahead (87.0% → 85.0%
name accuracy; 81.9% → 80.2% fully correct), but that lead comes almost entirely
from 50 validation rows that are V4 **training** examples. On the 66 clean rows
V4 and V5 are **identical** (both 78.8% fully correct). The honest conclusion is
a **tie**, not a V5 deficit; the `0.2552` training loss establishes nothing about
generalization.

Two volume regressions are general Tier-0 coverage gaps ("percent", "bring") that
can be fixed deterministically for both models; the rest are within-family model
errors plus the set_volume class-balance shift caused by the template cap.

V4 remains the production default. V5 is available alongside it as a non-default
artifact. The held-out `data/v5/vector_test.jsonl` has not been run and production
model selection has not been changed.
