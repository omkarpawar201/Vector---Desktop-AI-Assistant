<div align="center">

# ⚡ Vector

### Ultra-fast, local-first desktop AI assistant for Windows

*Talk to your PC. It listens, understands, and acts, safely.*

<br>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/Windows-10%20%7C%2011-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![UI](https://img.shields.io/badge/UI-PySide6-41CD52?style=for-the-badge&logo=qt&logoColor=white)](https://doc.qt.io/qtforpython/)
[![License](https://img.shields.io/badge/License-MIT-F7DF1E?style=for-the-badge)](LICENSE)

<br>

[Features](#-features) •
[Architecture](#-architecture) •
[Quick Start](#-quick-start) •
[Usage](#-usage) •
[Security](#-security--privacy) •
[Project Structure](#-project-structure)

</div>

---

## ✨ Overview

**Vector** gives you natural-language **voice and text control** over your computer.
It prioritizes **speed** (local execution first), **safety** (every action passes a permission engine), and **privacy** (works fully offline), with a cloud fallback only for complex, multi-step reasoning.

> 💡 Neither the local model nor Gemini can run OS commands directly. Every action goes through a validated **Tool Registry** and **Permission Engine**.

---

## 🚀 Features

| | Feature | Description |
|---|---|---|
| ⚡ | **Tier 0 Reflex Matcher** | Instant matching (< 5–10 ms) for deterministic commands: volume, media, system stats, power. No LLM needed. |
| 🧠 | **Needle 3 Local Engine** | Sub-second local intent recognition for private desktop control. |
| ☁️ | **Gemini Fallback** | Cloud reasoning for complex tasks, with autonomous multi-tool loops (capped at `MAX_TOOL_CALLS = 10`). |
| 🔒 | **Strict Security Boundary** | Central Tool Registry + Permission Engine (`SAFE`, `CONFIRM`, `DANGEROUS`, `BLOCKED`) + execution state machine. |
| 🛡️ | **Privacy Filter** | Redacts env vars, passwords, API keys, and SSH credentials before anything reaches a cloud model. |
| 🎙️ | **Opt-in Voice Pipeline** | Offline STT via `sherpa-onnx` (opt-in), plus native Windows TTS (SAPI5) with zero extra RAM. |
| 🖥️ | **Async PySide6 UI** | Dark-mode interface on Qt worker thread pools for a smooth 60 FPS experience. |

---

## 📐 Architecture

```mermaid
flowchart TD
    A([🎤 Text / Voice Input]) --> B[Input Normalizer]
    B --> C{Tier 0 Matcher<br/>Rules / Aliases}
    C -- hit --> G
    C -- miss --> D[Needle 3<br/>Local Intent AI]
    D -- high confidence --> G
    D -- low confidence / complex --> E[☁️ Gemini<br/>Multi-tool loop]
    E --> G[Tool Registry]
    G --> H[Validation +<br/>Permission Check]
    H --> I[Tool Executor]
    I --> J[OS]
    I --> K[Apps]
    I --> L[Files]
    J & K & L --> M[ToolResult]
    M --> N([💬 Response / TTS])

    style A fill:#6366f1,color:#fff,stroke:none
    style N fill:#6366f1,color:#fff,stroke:none
    style C fill:#10b981,color:#fff,stroke:none
    style D fill:#f59e0b,color:#fff,stroke:none
    style E fill:#3b82f6,color:#fff,stroke:none
    style H fill:#ef4444,color:#fff,stroke:none
```

### 🏎️ Routing Tiers

| Tier | Engine | Latency | Used for |
|:---:|---|:---:|---|
| **0** | Reflex Matcher | ~5–10 ms | Volume, media, stats, power |
| **1** | Needle 3 (local) | sub-second | Supported tool intents |
| **2** | Gemini (cloud) | varies | Complex, multi-step reasoning |

---

## 🧰 Tech Stack

| Layer | Technologies |
|---|---|
| **Core** | Python 3.11+, PySide6, SQLite (`vector.db`, WAL mode) |
| **Intent Engines** | Needle 3 (local), Google Gemini (`google-genai`), optional Ollama |
| **Voice** | `sherpa-onnx` (offline STT, opt-in), `pyttsx3` / SAPI5 (TTS), `silero-vad` (VAD) |
| **Automation** | `psutil`, `pywin32`, `pycaw`, `PyAutoGUI` |

---

## ⚙️ Quick Start

### Prerequisites

- 🐍 **Python 3.11+** on Windows
- 🔧 **Git**

### Installation

```powershell
# 1. Clone
git clone https://github.com/omkarpawar201/Vector---Desktop-AI-Assistant.git
cd Vector---Desktop-AI-Assistant

# 2. Create & activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your config
copy .env.example .env
```

**Model assets are not included in Git.** Needle V4 (`models/needle3_vector_v4.cact`) is required for Tier 1 local tool calling and is the **production default**; V5 is **experimental**. If the tuned V4 weights are missing, Tier 1 will be unavailable and requests will escalate or fall back as logged—Tier 0 (reflexes) and cloud fallback (if enabled) can still work for supported cases.  
Model acquisition instructions are **pending** (no verified download/license path is documented in this repository). Place any obtained weights at the path set by `NEEDLE_TUNED_WEIGHTS_PATH` (default `models/needle3_vector_v4.cact`).

### Configuration

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
NEEDLE_CONFIDENCE_THRESHOLD=0.85
ENABLE_GEMINI=true
ENABLE_LOCAL_LLM=false
ENABLE_VOICE=false
REQUIRE_CONFIRMATION_FOR_DANGEROUS=true
```

| Variable | Default | Description |
|---|:---:|---|
| `GEMINI_API_KEY` | – | Your Google Gemini API key (never committed) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model identifier |
| `NEEDLE_MODEL_PATH` | `models/needle_2_custom.onnx` | Tier 0.5 ONNX model path |
| `NEEDLE_OFFICIAL_WEIGHTS_PATH` | `models/needle3.cact` | Official Cactus base weights (reference) |
| `NEEDLE_TUNED_WEIGHTS_PATH` | `models/needle3_vector_v4.cact` | Tuned V4 weights for Tier 1 (production default; V5 experimental) |
| `NEEDLE_GENERATION` | `3` | Needle engine generation |
| `NEEDLE_MAX_DECODE_TOKENS` | `64` | Max decode tokens for Needle tool calls |
| `NEEDLE_CANDIDATE_USE_EMBEDDING` | `false` | Enable embedding-based candidate fallback (off by default) |
| `NEEDLE_CANDIDATE_SIMILARITY` | `0.55` | Cosine floor for embedding fallback |
| `NEEDLE_CONFIDENCE_THRESHOLD` | `0.85` | Min confidence before falling back to Gemini |
| `ENABLE_GEMINI` | `true` | Enable cloud Gemini fallback |
| `ENABLE_LOCAL_LLM` | `false` | Enable optional Ollama backend |
| `ENABLE_VOICE` | `false` | Enable STT/TTS voice pipeline |
| `REQUIRE_CONFIRMATION_FOR_DANGEROUS` | `true` | Ask before dangerous actions |
| `VOICE_STT_BACKEND` | `online` | STT backend: `online`, `offline`, or `auto` (`offline` uses local `sherpa-onnx`) |
| `VOICE_STT_MODEL_DIR` | `models/stt` | Directory for local `sherpa-onnx` offline STT model |
| `DATABASE_PATH` | `data/vector.db` | SQLite database path |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

**Offline behavior:** With `ENABLE_GEMINI=false` and no API key, the app runs fully offline where possible. Tier 0 reflexes always work locally. Tier 1 requires the tuned V4 model; without it Tier 1 is unavailable. Cloud features are skipped and the app falls back gracefully with clear messages when Gemini is disabled or unavailable. 

**Voice deps:** TTS uses `pyttsx3` (Windows SAPI5). Offline STT uses `sherpa-onnx` (opt-in; add it if `VOICE_STT_BACKEND=offline` or `auto` requires it). Microphone input is separate from TTS availability—missing TTS does not prevent text chat or STT setup.

### Run

```powershell
python -m app.main
```

---

## 🚦 Usage

| Category | Try saying |
|---|---|
| 📊 **System & Stats** | *"How much RAM am I using?"* · *"What's my CPU usage?"* · *"How much disk space is left?"* |
| 🔊 **Volume** | *"Set volume to 40"* · *"Mute audio"* · *"Unmute"* |
| 🎵 **Media** | *"Pause music"* · *"Next track"* · *"Play"* |
| 🚀 **Apps & Files** | *"Open Chrome"* · *"Close Spotify"* · *"Find resume.pdf"* |
| 🪟 **Windows** | *"Maximize window"* · *"Minimize active window"* |
| ⏻ **Power** | *"Lock PC"* · *"Sleep PC"* (requires confirmation) |
| 💻 **Terminal** | *"git status"* · *"python --version"* (whitelisted commands only) |

---

## 🛡️ Security & Privacy

### Permission Levels

| Level | Behavior | Example |
|:---:|---|---|
| 🟢 `SAFE` | Runs immediately | Volume, system stats |
| 🟡 `CONFIRM` | Requires user approval | Sleep, lock |
| 🔴 `DANGEROUS` | Explicit confirmation dialog | Shutdown, restart, delete |
| ⛔ `BLOCKED` | Never executed | Unvalidated shell commands |

### Guarantees

1. **No arbitrary code execution:** LLMs cannot run unvalidated shell commands.
2. **Explicit confirmations:** destructive or power actions always trigger a UI dialog.
3. **Data redaction:** sensitive files (`.env`, SSH keys) are masked before any cloud call.
4. **Local-first:** set `ENABLE_GEMINI=false` and Vector runs entirely offline where possible (Tier 0 local; Tier 1 needs tuned V4 weights). Cloud fallback only when enabled and available.

---

## 📂 Project Structure

<details>
<summary><b>Click to expand</b></summary>

```text
vector/
├── app/
│   ├── main.py                 # PySide6 app startup
│   ├── config/
│   │   ├── constants.py        # Enums, MAX_TOOL_CALLS, permission levels
│   │   └── settings.py         # .env settings parser
│   ├── core/
│   │   ├── normalizer.py       # Input normalizer & alias engine
│   │   ├── matcher.py          # Tier 0 pattern matcher
│   │   ├── capabilities.py     # Tool capability registry
│   │   ├── state_machine.py    # Execution state machine
│   │   ├── privacy.py          # Sensitive-data filter for Gemini
│   │   └── router.py           # Multi-tier intent router
│   ├── security/
│   │   ├── confirmation.py     # UI confirmation modals
│   │   ├── permissions.py      # Permission policy engine
│   │   └── validator.py        # Argument validator & dry-run planner
│   ├── tools/
│   │   ├── base.py             # BaseTool & ToolResult
│   │   ├── registry.py         # Central ToolRegistry singleton
│   │   ├── executor.py         # Safe ToolExecutor
│   │   ├── applications/       # launcher, manager
│   │   ├── files/              # search, manager
│   │   ├── media/              # controller
│   │   ├── power/              # lock, sleep, restart, shutdown
│   │   ├── system/             # system_info, volume, display
│   │   ├── terminal/           # whitelisted CLI executor
│   │   └── windows/            # win32 window management
│   ├── needle/                 # Needle 3 local model integration
│   ├── gemini/                 # Gemini client + multi-tool loop
│   ├── memory/                 # SQLite database & trace logger
│   ├── voice/                  # STT, TTS, VAD pipeline
│   └── ui/                     # PySide6 desktop interface
├── data/                       # SQLite storage
├── tests/                      # Unit tests
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

</details>

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m "Add amazing feature"`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

<div align="center">

<br>

**Built by [Omkar Pawar](https://github.com/omkarpawar201)**

⭐ Star this repo if you find it useful!

</div>
