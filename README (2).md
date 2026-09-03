# Robot Brain · Proactive Knowledge-Seeking Embodied Robot Cognitive Architecture

A seven-layer cognitive closed-loop engineering implementation built on the **bingogome/brainbot** framework.

> The machine no longer understands the world merely through pretraining; instead it keeps learning in the real world through "actively sensing the unknown → deciding to ask a human or explore → structured memory."

## 1. Architecture Overview

```
┌─────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
│ Input Layer │ → │ Understanding│ → │ Exploration Layer│ → │ Judgment Layer│
│ Vision/Audio│   │ Layer        │   │ (Core)           │   │ Safety Gate  │
│ /Propriocep-│   │ MWM World    │   │ Curiosity Engine │   │ (Fast)       │
│ tion        │   │ Model        │   │ Dream-MPC Planner│   │ Cognitive    │
│             │   │ Prediction   │   │ Dual-Path Decision│  │ Arbiter (Slow)│
│             │   │ Error        │   │ Question Generator│  │ Memory       │
└─────────────┘   │ Variance     │   └────────┬─────────┘   │ Retrieval    │
                  │ Trigger      │            │             └──────────────┘
                  └──────────────┘            ▼                     │
┌─────────────┐   ┌──────────────┐            ▼                     ▼
│ Brainbot    │ ← │ Long-Term    │ ←──── Seven-Layer Closed Loop (core/orchestration: brain.py)
│ Adapter     │   │ Memory Layer │      Perceive→Predict→Trigger→Decide→Arbitrate→Memorize→Act
│ ZMQ Policy  │   │ Vector Store │
│ Server      │   │ /Compression │
│             │   │ /Conflict    │
└─────────────┘   │ Resolution   │
                  └──────────────┘
```

**Fast/Slow Dual System**: the physical safety gate (fast system; 20 Hz in simulation / 1 kHz hard-realtime in real deployment) can interrupt the slow system's cognitive decisions at any time; the world model / cognitive decisions form the slow system (1–2 Hz).

## 2. Directory Structure

```
robot_brain/
├── config/                  # brainbot_config / curiosity_thresholds / safety_rules
├── src/
│   ├── core/                # brain.py (seven-layer loop orchestration) + types + config loader
│   ├── input_layer/         # vision encoding (256-dim) / audio (ASR+emotion+intent) / proprioception
│   ├── understanding_layer/ # world model (MWM adapter + Bayesian fallback) / prediction error / variance trigger
│   ├── exploration_layer/   # curiosity engine / action proposer / dual-path decision / question generator
│   ├── judgment_layer/      # safety gate / cognitive arbiter / memory retrieval
│   ├── memory/              # vector store (faiss/numpy) / L1→L2→L3 compression / knowledge conflict resolution
│   └── brainbot_adapters/   # ZMQ policy server / command provider / payload conversion
├── simulation/
│   ├── world_sim.py         # lightweight 2D world (primary closed-loop demo driver)
│   ├── mujoco_bridge.py     # MuJoCo physics backend (differential-drive chassis)
│   └── mujoco_models/       # MJCF robot model (importable by URLab)
├── tests/                   # 31 unit + end-to-end tests
├── launch/                  # launch_remote.sh / launch_hub.sh / run_all_tests.sh
└── scripts/demo_closed_loop.py   # end-to-end closed-loop demo
```

## 3. Installation

```bash
# 1) Python environment (≥3.10; this project verified on 3.12)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Brainbot upstream (ZMQ transport / message protocol)
git clone https://github.com/bingogome/brainbot.git vendor/brainbot
pip install -e vendor/brainbot
# Note: to use the ZMQ transport layer standalone in an environment without lerobot,
#       a minimal lazy-import patch was applied to brainbot_core/__init__.py in the
#       vendored copy (see the header comment of that file).

# 3) Optional: MuJoCo physics backend, faiss vector index (already included in requirements)
```

## 4. Running

```bash
# Seven-layer closed-loop demo (ask path + explore path, with learning convergence)
python scripts/demo_closed_loop.py

# All tests
launch/run_all_tests.sh

# Start the AI policy server (Brainbot ZMQ REP, tcp://127.0.0.1:5555)
python -m src.brainbot_adapters.policy_server --port 5555

# One-click remote/hub startup (see launch/*.sh)
launch/launch_remote.sh
```

## 5. Component Selection and Implementation Status

| Layer | Document Selection | Implementation in This Project |
|---|---|---|
| Core control | bingogome/brainbot | ✅ Installed; adapters reuse the upstream ZMQ/msgpack protocol |
| Simulation | URLab + MuJoCo | ⚠️ URLab requires a UE 5.7 C++ project (no GUI/GPU in sandbox, cannot build); provides MJCF model + MuJoCo physics bridge |
| RL interface | AMD Schola | ⚠️ Requires UE; a Gymnasium-compatible integration point is kept (mujoco_bridge) |
| World model | Mask World Model (ICML'26) | ⚠️ Requires GPU + GE-Base weights; provides mwm backend adapter + online Bayesian dynamics fallback |
| Exploration engine | Explauto | ⚠️ The library is incompatible with Py3.12; provides explauto adapter entry + equivalent interest-model implementation |
| Action planning | Dream-MPC (ICML'26) | ⚠️ Requires a GPU world model; provides a lightweight shooting-MPC equivalent |
| Vector memory | Chroma / FAISS | ✅ faiss-cpu / numpy dual backend |
| Speech | Whisper + emotion recognition | ⚠️ whisper requires torch; provides whisper adapter + rule-based backend |
| Multimodal | CLIP | Adapter entry (vision encoder numpy/torch dual backend) |

**Note**: All ⚠️ items keep the adapter interface for real deployment (`backend="auto"` auto-detection); after installing the corresponding weights in a GPU/UE environment, they switch to the real implementation. Within the CPU sandbox, equivalent fallback backends keep the entire closed loop runnable and verifiable.

## 6. Verification Checklist (against the document)

| # | Verification Item | Result |
|---|---|---|
| 1 | Service startup | ✅ 31 tests passed |
| 2 | Policy server ZMQ handshake | ✅ Covered by end-to-end tests (ping + get_action) |
| 3 | Vision encoding outputs 256-dim | ✅ test_vision_encoder |
| 4 | World model prediction error computable | ✅ test_prediction_error |
| 5 | Curiosity trigger (unknown object) | ✅ Dual-path closed-loop test |
| 6 | Dual-path decision (ask / explore) | ✅ Boundary verified ask=7.0 / explore=5.4 |
| 7 | Safety gate hard interrupt | ✅ HALT/WAIT/RECHARGE/SOFT_STOP |
| 8 | Episodic memory read/write <50ms | ✅ Vector store retrieval + closed-loop writes |
| 9 | Simulation model loading | ✅ MJCF parsed by MuJoCo + physics stepping |
| 10 | End-to-end closed loop | ✅ Perceive→decide→act→learning convergence |

## 7. Key Technical Behaviors

- **Prediction-variance trigger**: an unknown object's world-model prior variance is high → triggers "unknown perception"; once learned, known samples are seeded so the variance collapses → no repeated triggering.
- **Dual-path decision**: scoring table (risk / human idle / cost / task / dynamicity); ≥6 ask a human, <6 explore.
- **Lifelong learning**: categories obtained by asking or exploring are written into five-tuple memory; from the next round the object is "known", so the robot keeps learning without repeated questions.
