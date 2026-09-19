# Vector — Desktop AI Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)

**Vector** is an ultra-fast, local-first desktop AI assistant designed for Windows. It provides natural language voice and text control over your computer, prioritizing fast local execution, strict tool-based OS control, privacy, safety permissions, and cloud fallback for complex multi-step reasoning.

---

## 🚀 Key Features

* **⚡ Tier 0 Reflex Matcher (< 5–10 ms)**: Instant direct matching for deterministic commands (volume, media controls, system stats, power) without waiting for LLM inference.
* **🧠 Needle 2 Local Intent Engine**: Sub-second local intent recognition for privacy-focused desktop control.
* **☁️ Gemini API Fallback & Multi-Tool Loops**: Cloud reasoning fallback capable of multi-step autonomous tool execution (`MAX_TOOL_CALLS = 10` cap) for complex tasks.
* **🔒 Strict Security Boundary**: Neither Needle nor Gemini directly execute OS commands. All actions flow through a central **Tool Registry**, **Permission Engine** (`SAFE`, `CONFIRM`, `DANGEROUS`, `BLOCKED`), and execution state machine.
* **🛡️ Privacy Filter**: Automatically redacts sensitive environment variables, passwords, API keys, and SSH credentials before context reaches cloud models.
* **🎙️ Opt-In Voice Pipeline**: Offline Speech-to-Text via `faster-whisper` (`int8` quantized) and instant zero-RAM native Windows TTS (SAPI5).
* **🖥️ Asynchronous PySide6 UI**: Responsive dark-mode interface powered by Qt worker thread pools to guarantee 60 FPS UI performance.

---

## 📐 System Architecture

```text
                         VECTOR
                            │
                   Text / Voice Input
                            │
                            ▼
                  ┌──────────────────┐
                  │ Input Normalizer │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Tier 0 Matcher   │
                  │ Rules / Aliases  │
                  └────────┬─────────┘
                           │ miss
                           ▼
                  ┌──────────────────┐
                  │    Needle 2      │
                  │ Local Intent AI  │
                  └────────┬─────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
          High confidence        Low confidence /
          supported tool        complex request
                │                     │
                │                     ▼
                │                  Gemini
                │                     │
                │              multi-tool loop
                │                     │
                └──────────┬──────────┘
                           ▼
                  ┌──────────────────┐
                  │  Tool Registry   │
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │ Validation +     │
                  │ Permission Check │
                  └────────┬─────────┘
                           ▼
                     Tool Executor
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
            OS            Apps          Files
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                      ToolResult
                           │
                           ▼
                    Response / TTS
```

---

## 🧰 Technology Stack

* **Core**: Python 3.11+, PySide6 (GUI), SQLite (`vector.db` with WAL mode)
* **Intent Engines**: Needle 2 (Local Model), Google Gemini API (`google-genai`), optional Ollama
* **Voice**: `faster-whisper` (STT), `pyttsx3` / SAPI5 (TTS), `silero-vad` (VAD)
* **System Automation**: `psutil`, `pywin32` (`win32gui`, `win32con`), `pycaw` (Windows Audio), `PyAutoGUI`

---

## 📂 Project Structure

```text
vector/
│
├── app/
│   ├── main.py                    # Lightweight PySide6 app startup
│   ├── config/
│   │   ├── constants.py           # Enums, MAX_TOOL_CALLS, permission levels
│   │   └── settings.py            # Fast settings parser from .env
│   ├── core/
│   │   ├── normalizer.py          # Input text normalizer & alias engine
│   │   ├── matcher.py             # Tier 0 direct pattern matcher (<5-10ms)
│   │   ├── capabilities.py       # Tool capability registry
│   │   ├── state_machine.py       # Execution state machine
│   │   ├── privacy.py             # Gemini privacy & sensitive-data filter
│   │   └── router.py              # Multi-tier Intent Router
│   ├── security/
│   │   ├── confirmation.py       # UI confirmation modals
│   │   ├── permissions.py        # Permission policy engine
│   │   └── validator.py          # Argument validator & dry-run planner
│   ├── tools/
│   │   ├── base.py               # BaseTool & ToolResult data structures
│   │   ├── registry.py           # Central ToolRegistry singleton
│   │   ├── executor.py           # Safe ToolExecutor
│   │   ├── applications/         # launcher.py, manager.py
│   │   ├── files/                # search.py, manager.py
│   │   ├── media/                # controller.py
│   │   ├── power/                # power.py (lock, sleep, restart, shutdown)
│   │   ├── system/               # system_info.py, volume.py, display.py
│   │   ├── terminal/             # executor.py (whitelisted CLI)
│   │   └── windows/              # manager.py (win32 window management)
│   ├── needle/                   # Needle 2 local model integration
│   ├── gemini/                   # Gemini API client with multi-tool loop
│   ├── memory/                   # SQLite database & trace logger
│   ├── voice/                    # STT, TTS, VAD voice pipeline
│   └── ui/                       # PySide6 desktop interface
│
├── data/                         # SQLite database storage
├── tests/                        # Automated unit test suite
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
* **Python 3.11+** installed on Windows.
* **Git** installed.

### 2. Clone Repository
```bash
git clone https://github.com/omkarpawar201/Vector---Desktop-AI-Assistant.git
cd Vector---Desktop-AI-Assistant
```

### 3. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env` and set your configuration options:
```env
GEMINI_API_KEY=your_gemini_api_key_here
NEEDLE_CONFIDENCE_THRESHOLD=0.85
ENABLE_GEMINI=true
ENABLE_LOCAL_LLM=false
ENABLE_VOICE=false
REQUIRE_CONFIRMATION_FOR_DANGEROUS=true
```

---

## 🚦 Usage Examples

* **System & Stats**: *"How much RAM am I using?"*, *"What's my CPU usage?"*, *"How much disk space is left?"*
* **Volume Control**: *"Set volume to 40"*, *"Mute audio"*, *"Unmute"*
* **Media Player**: *"Pause music"*, *"Next track"*, *"Play"*
* **Applications & Files**: *"Open Chrome"*, *"Close Spotify"*, *"Find resume.pdf"*
* **Window Control**: *"Maximize window"*, *"Minimize active window"*
* **Power**: *"Lock PC"*, *"Sleep PC"* *(Requires confirmation dialog)*
* **Terminal**: *"git status"*, *"python --version"* *(Strictly whitelisted commands)*

---

## 🛡️ Security & Privacy

1. **No Arbitrary Code Execution**: LLMs cannot execute unvalidated shell commands.
2. **Explicit User Confirmations**: Destructive or power actions (`shutdown`, `restart`, `delete`) trigger explicit UI confirmation dialogs.
3. **Data Redaction**: Sensitive files (`.env`, SSH keys) are automatically masked before any cloud call.
4. **Local-First Privacy**: Toggling `ENABLE_GEMINI=false` runs Vector entirely offline.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
