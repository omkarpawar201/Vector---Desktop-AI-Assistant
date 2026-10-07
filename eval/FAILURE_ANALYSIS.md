# Fresh Evaluation — Failure Analysis

Source: `eval/fresh_eval_2026_10.json` (98 queries, 10 categories)
Companion to: `eval/FRESH_EVAL_REPORT.md`
No production code, matcher rule, model or dataset was modified in producing this.

---

## 0. BLOCKER FOUND FIRST: the engine pool corrupts model output

Before any accuracy work, a correctness defect was found that changes how every
number below must be read.

`NeedleEnginePool._acquire()` (`app/needle/client.py`) caches one `Needle`
subprocess per candidate set and reuses it across `complete()` calls. The docstring
presents this as a pure performance win ("the tool prefix stays resident instead of
being rebuilt"). In fact the reused `Needle` **retains state between calls**, so
later queries are answered in the context of earlier ones.

**Evidence**

| Test | Result |
|---|---|
| `open it` alone, pristine client | `launch_app {"name": "it"}` |
| `open it` after `i need to find the quarterly report` | `launch_app {"name": "quarterly report"}` |
| `volume percentage right now` ×4 on one cached engine | `get_volume` → `set_volume{level:50}` → `get_volume` → `set_volume{level:100}` |
| same 5 queries ×3, pool purged before each call | **identical every time** |
| same 98 queries, **reversed order** | 9 verdicts flip; score 55.1% → 58.2% |

A pristine engine is fully deterministic. Engine reuse is the sole source of
instability. Arguments from an unrelated earlier command can be emitted verbatim
into a later command's call — a real correctness and safety problem, not just noise.

**Consequences for this evaluation**

| Configuration | Score | Meaning |
|---|---|---|
| As shipped, dataset order | **55.1%** (54/98) | what a user actually gets |
| As shipped, reversed order | **58.2%** (57/98) | same code, same queries |
| Pristine engine per query | **61.2%** (60/98) | V4's true deterministic capability |

14 of 98 verdicts (14.3%) depend on query history. **The as-shipped number is not
reproducible and must not be used as a training or regression baseline.**

The existing 104-row gate *is* reproducible (3 consecutive runs at 104/104) because
86/104 are Tier 0 and the 18 V4 rows use training-distribution phrasings. It simply
never exercises this defect.

**Sequencing recommendation: fix the pool reuse before retraining.** Any V5
evaluation loop built on the current client will be unmeasurable, because you cannot
tell a capability change from state drift.

All analysis below uses the **pristine-engine (deterministic) run**, since that is
the only configuration in which a verdict means one thing. Order-dependent results
are flagged separately.

---

## 1. Wrong-tool failures (14)

Decoded: 11 from V4, 2 from Tier 0, 3 of the 11 are vague queries that should have
escalated.

| # | Category | Family | Expected | Actual | Confusion pair | Wording pattern |
|---|---|---|---|---|---|---|
| 1 | ambiguous | app_management | *Tier 2* | `launch_app` | vague→action | bare pronoun, no antecedent: `open it` |
| 2 | ambiguous | media_control | *Tier 2* | `media_pause` | vague→action | complaint + open-ended remedy: `the music is too loud can you fix it` |
| 3 | ambiguous | media_control | *Tier 2* | `media_previous` | vague→action | narrative reference, no verb: `carry on from where we left the music` |
| 4 | battery | app_management | `get_battery_status` | `get_running_apps` | battery→unrelated family | pronoun "I" + `power`; no battery noun → routed to wrong *family* |
| 5 | battery | system_monitoring | `get_battery_status` | `get_disk_usage` | battery↔disk | indirect noun: `how much juice is left` |
| 6 | battery | system_monitoring | `get_battery_status` | `get_system_stats` | battery↔stats | indirect noun: `is the charger connected right now` |
| 7 | media | volume_control | `media_pause` | `unmute` | media↔volume, wrong polarity | `freeze the audio` — routed to volume family by `audio` |
| 8 | media | media_control | `media_play` | `media_pause` | **play↔pause** | resume register: `carry on with the track` |
| 9 | media | media_control | `media_play` | `media_pause` | **play↔pause** | "again" + start verb: `fire up the music again` |
| 10 | media | media_control | `media_play` | `media_pause` | **play↔pause** | negated pause: `unpause the track` |
| 11 | media | media_control | `media_previous` | `media_next` | previous↔next | direction word `reverse` + `earlier` — both point backwards, read as forwards |
| 12 | volume | volume_control | `mute` | `unmute` | **mute↔unmute (polarity)** | `shush the speakers` — no polarity-bearing word at all |
| 13 | volume | Tier 0 | `mute` | `close_app{"sound"}` | **audio noun→app name** | `kill the sound` — "sound" captured as an application |
| 14 | file_search | Tier 0 | `search_files` | `get_disk_usage` | **file-verb→system-resource** | `search the disk for budget` — "disk" fires system regex before file regex |

### Observations

- **Polarity is broken in both directions.** `mute`↔`unmute` fails twice (#7, #12),
  once in each direction, once with no polarity word present at all (#12). The model
  has not learned a polarity axis; it has learned surface tokens.
- **`play` is the worst single class**: every media_play failure lands on
  `media_pause` (3/3). The model treats any resume-flavoured verb as "stop".
  `unpause` is the clearest case — it contains the substring `pause`, and the model
  appears to be matching subwords rather than meaning.
- **Battery fails 3/4 into three different wrong tools**, i.e. not one consistent
  confusion but diffuse. `system_monitoring` holds 5 tools and the model cannot
  discriminate within it.
- Two of the four non-media classes fail **at family selection, not classification**
  (#4, #7): the rules-only router put the query in the wrong family, then V4 was
  forced to pick from an unrelated candidate set and did so confidently.

---

## 2. Correct intent, unnecessary escalation (20)

**19 of 20 never reached V4 at all** — `select_candidates()` returned `None`, so
this is a *candidate-router* failure, not a model failure. Family scores were
measured directly against `MIN_FAMILY_SCORE = 2`.

| Category | n | Family scores | Why no family was selected | Verdict |
|---|---|---|---|---|
| disk_storage | 4 | `{}`, `{system:1}` ×1 | Missing vocabulary: `drive` (×3), `filesystem`/`capacity` | **router limitation** — plain synonyms of `disk`/`storage` |
| battery | 1 | `{}` | Missing `power` (only `battery`/`charg` are keywords) | **router limitation** — `\bpower\b` is not a system_monitoring keyword anywhere |
| applications | 4 | `{}` ×4 | Missing launch verbs: `bring up`, `get rid of`, `end`, `going` | **router limitation** — all four are ordinary phrasings of launch/close |
| file_search | 3 | `{}` ×3 | Missing search verbs: `where did i put`, `hunt down`, `dig up` | **router limitation** |
| untrained_tools | 8 | `{}` / blocked | Tools have **no training coverage**; `_UNCOVERED_INTENT` and the family table exclude them by design | **by design** — see A/B/C below |
| media | 0 | — | (`back up one track` no longer escalates on a pristine engine) | — |

**The bottleneck is the router's keyword table, not V4.** Twelve queries describe an
intent that lives squarely inside an existing family, and the router never offered
that family. Because family selection is scored on single keywords, any query whose
only marker is a synonym the table happens to lack (`drive`, `power`, `bring up`,
`dig up`) is silently escalated even though V4 would very likely have answered it.

This also explains the escalation *rate* asymmetry: 0 escalations among trained
families once the router fires, versus 100% whenever it does not.

Note that 8 of the 9 non-untrained escalations are recoverable **without touching
V4 at all** — the information needed was already in the query.

---

## 3. Correct tool, wrong arguments (4)

All four are Tier 0 argument-extraction defects, all the same class: **discourse
scaffolding leaks into the captured value.**

| Query | Expected | Actual | Defect |
|---|---|---|---|
| `fire up notepad for me` | `{"name": "notepad"}` | `{"name": "notepad for me"}` | trailing politeness suffix not stripped |
| `close firefox for me` | `{"name": "firefox"}` | `{"name": "firefox for me"}` | same |
| `open up the file explorer` | `{"name": "explorer"}` | `{"name": "up the file explorer"}` | intervening verb particle `up` not skipped |
| `open the folder called Documents` | `{"path": "documents"}` | `{"path": "called documents"}` | introducer `called` not stripped |

**This is deterministic normalization, not a model problem.** Every one is produced
by a regex capture in the matcher, before V4 is consulted. `launch_app{"notepad for
me"}` will fail at the OS level for *any* politeness phrasing, not just this one.

It is also the cheapest defect class in the report: 4/4 failures, one shared root
cause (the app/path capture lacks a trailing-noise filter), no model involvement.

---

## 4. Counts

### Failures by expected tool

| Tool | Failures | Dominant cause |
|---|---|---|
| `get_disk_usage` | 4 | router keyword gap (all escalations) |
| `get_battery_status` | 4 | 3 wrong-tool (intra-family confusion), 1 router gap |
| `launch_app` | 4 | all router keyword gap (escalations) |
| `search_files` | 4 | 2 router gap, 1 wrong-tool, 1 Tier 0 precedence |
| `media_play` | 3 | **all 3 → `media_pause`** |
| `close_app` | 3 | all router keyword gap |
| *Tier 2 expected* | 3 | model failed to abstain |
| `mute` | 2 | polarity (→`unmute`, →`close_app`) |
| `media_previous` | 1 | →`media_next` |
| `media_pause` | 1 | →`unmute` |
| `sleep_pc`, `restart_pc`, `shutdown_pc`, `lock_pc`, `get_active_window`, `minimize_window`, `open_folder`, `get_file_info`, `execute_terminal_command` | 1 each | untrained; no family exists |

**Top three concentrations:** the 12 router escalations, the media resume/play
cluster, and the 9 untrained-tool escalations together account for 24 of 38 failures.

### Confusion pairs

| n | Expected → Actual |
|---|---|
| 3 | `media_play` → `media_pause` |
| 1 | `get_battery_status` → `get_running_apps` |
| 1 | `get_battery_status` → `get_disk_usage` |
| 1 | `get_battery_status` → `get_system_stats` |
| 1 | `mute` → `unmute` |
| 1 | `mute` → `close_app` |
| 1 | `media_pause` → `unmute` |
| 1 | `media_previous` → `media_next` |
| 1 | `search_files` → `get_disk_usage` |
| 1 | *Tier 2* → `media_pause` |
| 1 | *Tier 2* → `media_previous` |
| 1 | *Tier 2* → `launch_app` |

Twelve distinct pairs, eleven of them singletons. **Only one pair repeats**:
`media_play`→`media_pause`. There is no broad systematic confusion matrix — the
model fails in scattered, mostly one-off ways. That is the signature of lexical
memorisation rather than a learned decision boundary: each training phrase is
remembered, and anything unremembered lands on the nearest strong lexical
attractor.

---

## 5. Media contrasts (deep dive)

The requested contrasts, against the deterministic run. All 14 media cases:

| Query | Expected | Actual | Contrast tested |
|---|---|---|---|
| `on to the following track` | `media_next` | `media_next` ✅ | following → next |
| `advance the playlist` | `media_next` | `media_next` ✅ (Tier 0) | advance → next |
| `jump ahead a track` | `media_next` | `media_next` ✅ (Tier 0) | ahead → next |
| `hit next` | `media_next` | `media_next` ✅ | terse, no media noun |
| `gimme the next one` | `media_next` | `media_next` ✅ | terse, slang |
| `back up one track` | `media_previous` | `media_previous` ✅ | ambiguous verb, resolved |
| `step back one track` | `media_previous` | `media_previous` ✅ | step back → previous |
| `reverse to the earlier track` | `media_previous` | **`media_next`** ❌ | previous ↔ next |
| `suspend the song` | `media_pause` | `media_pause` ✅ | suspend → pause |
| `kill the music` | `media_pause` | `media_pause` ✅ | kill → pause |
| `carry on with the track` | `media_play` | **`media_pause`** ❌ | **play ↔ pause** |
| `fire up the music again` | `media_play` | **`media_pause`** ❌ | **play ↔ pause** |
| `unpause the track` | `media_play` | **`media_pause`** ❌ | **play ↔ pause** |
| `freeze the audio` | `media_pause` | **`unmute`** ❌ | media ↔ volume, wrong polarity |

**Revised picture — media is better than the 55% run suggested, but fragile.**

- **Direction (next/previous) is genuinely strong on a clean engine: 7/8.** The
  as-shipped run showed `media_next`→`media_play` twice, `media_previous`→`media_pause`,
  and `media_previous`→`media_play` — those were **state-leak artifacts, not capability
  gaps**. Only one true direction failure remains (`reverse…earlier` → next).
- **Play vs pause is the real and reproducible defect: 0/3.** `carry on`,
  `fire up … again`, and `unpause` all collapse to `media_pause` on a pristine engine.
  Note that `carry on with the track` was *correct* under leakage and wrong without
  it — leakage was accidentally helping here.
- **Terminology gaps that produced apparent media failures are not model failures.**
  The `resume`, `continue`, `unpause`, `keep playing`, `skip`, `forward`, `go back`
  families were never tested on a clean engine except here, and `carry on` /
  `unpause` both fail. `next`/`skip`/`forward` and `previous`/`go back`/`take me back`
  are represented well enough to pass, but they were only sampled twice each, so
  treat 7/8 as provisional rather than established.
- **One cross-family leak:** `freeze the audio` was routed to `volume_control`
  because the router weights `audio` toward volume (3) and the query has no media
  noun. V4 then had no media tool available and chose `unmute`. The router caused
  this, not the model.

---

## 6. Failures outside media

Media accounts for only 5 of 38 deterministic failures. The larger non-media blocks:

- **Candidate-router keyword coverage — 12 failures, the single biggest bucket.**
  Concentrated in disk/storage synonyms (`drive`, `filesystem`, `capacity`, `full`,
  `gigs`) and app/file verb gaps (`bring up`, `get rid of`, `end`, `going`, `where
  did i put`, `hunt down`, `dig up`). 8 of these are on trained families where V4 was
  never consulted.
- **Untrained tools — 8 failures.** `sleep_pc`, `restart_pc`, `shutdown_pc`,
  `lock_pc`, `get_active_window`, `minimize_window`, `get_file_info`,
  `execute_terminal_command`. These have no training data and no family, so V4 can
  never answer them by design. They are not accuracy failures; they are coverage
  decisions.
- **Battery — 4 failures**, 3 of them diffuse wrong-tool inside `system_monitoring`.
  `system_monitoring` is a 5-tool family with no intra-family contrast training.
- **Volume polarity — 2 failures**, plus the cross-family `freeze the audio`.
- **Tier 0 argument capture — 4 failures**, one shared root cause.
- **Abstention — 3 failures**, all vague queries that executed a tool.

---

## 7. Classification: what belongs where

### A. Fix with training data (model capability)

1. **Play vs pause resume cluster** — the only repeated confusion pair. Needs
   minimal-pair contrast training where the *only* difference is the verb.
2. **Polarity (mute/unmute, play/pause)** as an explicit axis. Currently not modelled
   at all; errors occur in both directions and with no polarity word present.
3. **Intra-`system_monitoring` discrimination** — battery vs disk vs stats vs cpu.
   5 tools, zero contrast training between them.
4. **`system_monitoring` battery vocabulary** — `juice`, `charger`, `plugged`,
   `low on power`, `power situation`, `running out`.
5. **Volume read vs write** — `get_volume` when no number is present vs `set_volume`
   when one is. The model invented `{level: 50}` and later `{level: 100}` for a pure
   read query.
6. **`search_files` vs `launch_app`** — find/search/locate + noun phrase vs
   open/launch + app name.
7. **Abstention / negative examples** — the model currently emits a tool for every
   query, including bare pronouns and complaints. It needs training examples that
   correctly produce **no tool call**.
8. **Direction robustness for `previous`** — one confirmed failure, but 7/8 rests on
   two samples per direction word; needs more coverage before it is trusted.

### B. Fix with deterministic rules (no model involvement)

1. **Candidate-router keyword gaps** — the 12 escalations. A synonym set for
   `drive`/`filesystem`/`capacity`/`full`, `power`, and the missing app/file verbs.
   This is the highest-value non-model change: 12 of 38 failures, zero risk to the
   model.
2. **App/path argument normalization** — strip trailing `for me`/`please`, leading
   particles (`up`), and introducers (`called`). One root cause, four failures.
3. **`kill the sound` → `close_app{"sound"}`** — the app-close rule must not treat
   `the sound`/`the music` as an application name.
4. **Tier 0 precedence: `search the disk for budget` → `get_disk_usage`** — the
   system-resource rule preempts the file-search rule on the token `disk`. A
   file-verb + object-noun query should win.
5. **Router cross-family weight: `freeze the audio`** — `audio` alone should not
   pull a media query into `volume_control` when no volume intent is present.

### C. Correctly left as Tier 2

1. `make it quieter` — no target, no level.
2. `the music is too loud can you fix it` — lower volume vs stop playback.
3. `resume` (bare) — file vs media; the router's deterministic disambiguation already
   handles this correctly.
4. `open it` — dangling pronoun, no antecedent in single-turn context.
5. `yeah go ahead`, `do the thing`, `can you handle that`, `same as before`,
   `fix it`, `my computer feels slow today` — genuinely underspecified.

These are working as designed. The defect is that **3 of them did not stay in Tier 2**
(the abstention gap in A7) — not that they were escalated.

For the 8 untrained tools, split by risk rather than lumping them:
- `sleep_pc`, `restart_pc`, `shutdown_pc`, `lock_pc`, `get_active_window`,
  `minimize_window`, `get_file_info` → **B**, deterministic rules. Fixed surface
  forms, no ambiguity, and the first three are friction/confirmation-gated anyway.
- `execute_terminal_command` → **keep out of V4 permanently.** It is already
  whitelist-protected; the eval case (`git push origin main`) is exactly the shape
  that should be an explicit confirm, not a model guess.

---

## 8. What the next training dataset must contain

**Contamination warning.** This analysis has now read all 98 queries and their
outcomes. **The set is burned as a measurement instrument.** Copying these queries —
or minimal paraphrases of them — into training would make the next evaluation
meaningless, and would repeat exactly the mistake that produced the fake 104/104.

Methodology to avoid that:

1. **Author the training data from the contrast matrix below, not from the eval
   queries.** Specify capability contrasts (verb pairs, polarity axes, read/write
   pairs, intra-family confusions, abstention class). Then generate surface forms
   from an independently written paraphrase inventory. If an eval query appears in
   the training set, that is a bug — check for it mechanically.
2. **Treat the 61.2% pristine-engine number as the new reference**, not 55.1%.
3. **After V5, this set is spent. Commission a third fresh set** for the honest
   post-training number. A set is good for exactly one honest measurement.

### Required content

| Block | Purpose | Minimum shape |
|---|---|---|
| 1. Play/pause minimal pairs | the only repeated confusion | Verb pairs differing in one token, both orders: resume-family → `media_play`, hold-family → `media_pause`. Must include negated forms (`un-` prefixed) and forms containing the opposite class's substring. |
| 2. Polarity axis | mute/unmute, play/pause | Explicit on/off pairs across several carriers (verb, adjective, particle), including polite and colloquial carriers. Include the *absence* of a polarity word as its own decision. |
| 3. Direction axis | next/previous | Forward family vs backward family, several carriers each, plus one verb (`back up`) carrying two senses so the model learns the disambiguation. |
| 4. Intra-family system contrast | battery/disk/stats/cpu | Balanced 4-way contrast inside `system_monitoring` — the family is 5 tools with no contrast data. |
| 5. Battery vocabulary | indirect battery language | `juice`, `charger`, `plugged`, `power`, `low on`, `running out`, `charge left`. |
| 6. Volume read vs write | `get_volume` vs `set_volume` | Pairs identical except for the presence of a number. Must include explicit no-argument cases so the model learns to emit *nothing* for a read. |
| 7. Search vs launch | `search_files` vs `launch_app` | Find-class verb + noun phrase vs open-class verb + app name, minimal pairs. |
| 8. Abstention class | the 3 current abstention failures | Examples with a pronoun or complaint and **no resolvable target**, labelled as *no tool call*. This class does not exist in the data today. |
| 9. Cross-family hard negatives | `freeze the audio`, `kill the sound` | Audio-word queries that are media, and kill-class verbs whose object is not an app. |
| 10. Volume/media boundary | router-vs-model division | Only after B5 is fixed — otherwise these queries never reach the model and the data is wasted. |

### Sequencing

```
1. Fix engine pool reuse           (else no evaluation is trustworthy)
2. Category B deterministic fixes  (12 + 4 + 2 failures, no model risk)
3. Re-measure on a NEW fresh set   (this one is spent)
4. Author training data from the matrix, never from the eval queries
5. Train V5
6. Evaluate on a THIRD fresh set
```

Steps 1 and 2 are worth doing regardless of whether retraining ever happens, and
between them they address 16 of 38 deterministic failures without touching V4.

---

## 9. Method notes

- Determinism verified: 5 queries × 3 repeats with the pool purged returned
  identical results.
- Regression gate re-verified: `scripts/eval_pipeline.py` 3 consecutive runs at
  104/104, 0 admission violations — reproducible, and unaffected by the pool defect
  because 86/104 are Tier 0.
- Model and checkpoint hashes unchanged; no production file was modified.
- Analysis scripts were run from `/tmp`; the repository gained only this report.