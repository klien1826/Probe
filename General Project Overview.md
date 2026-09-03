# Robot Brain · Active Knowledge-Seeking Embodied Robot Cognitive Architecture

A seven-layer cognitive closed-loop engineering implementation based on the **bingogome/brainbot** framework.

> A robot no longer only understands the world through pretrained models — it continuously learns in the real world through **actively perceiving the unknown → deciding to ask humans or explore → structuring memory**.

## 1. Architecture Overview

```
┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐   ┌──────────────────┐
│   INPUT LAYER    │ → │  UNDERSTANDING   │ → │ EXPLORATION (CORE) │ → │  JUDGMENT LAYER  │
│ Vision / Audio / │   │  MWM World Model │   │  Curiosity Engine  │   │  Safety Gate     │
│ Proprioception   │   │  Prediction Err. │   │  Dream-MPC Propose │   │  Cognitive Arb.  │
│                  │   │  Variance Trigger│   │  Dual-Path Decide  │   │  Memory Retrieval│
└──────────────────┘   └──────────────────┘   │  Question Gen.     │   └──────────────────┘
                                               └─────────┬──────────┘                 │
                                                         ▼                            ▼
┌──────────────────┐   ┌──────────────────┐  ◄───────── Seven-Layer Closed Loop (orchestrator: src/core/brain.py)
│   BRAINBOT       │ ← │ LONG-TERM MEMORY │       perceive → predict → trigger → decide →
│   ADAPTERS       │   │      LAYER       │       arbitrate → remember → act
│ ZMQ Policy Svr   │   │ Vector Store /   │
│ Command Provider │   │ L1→L2→L3 Compress│
└──────────────────┘   │ Conflict Resolve │
                       └──────────────────┘
```

**Fast/Slow Dual-System**: the physical safety gate (fast system — 20 Hz in simulation / 1 kHz hard real-time in deployment) can interrupt the slow system's cognitive decisions at any time; the world model / cognitive decisions run as the slow system (1–2 Hz).

## 2. Directory Structure

```
robot_brain/
├── config/                  # brainbot_config / curiosity_thresholds / safety_rules
├── src/
│   ├── core/                # brain.py (seven-layer closed-loop orchestration) + types + config loader
│   ├── input_layer/         # vision encoding (256-d) / audio (ASR + emotion + intent) / proprioception
│   ├── understanding_layer/ # world model (MWM adapter + Bayesian fallback) / prediction error / variance trigger
│   ├── exploration_layer/   # curiosity engine / action proposer / dual-path decision / question generation
│   ├── judgment_layer/      # safety gate / cognitive arbiter / memory retrieval
│   ├── memory/              # vector store (faiss/numpy) / L1→L2→L3 compression / knowledge conflict resolution
│   └── brainbot_adapters/   # ZMQ policy server / command provider / payload conversion
├── simulation/
│   ├── world_sim.py         # lightweight 2D world (primary closed-loop demo driver)
│   ├── mujoco_bridge.py     # MuJoCo physics backend (differential-drive base)
│   └── mujoco_models/       # MJCF robot model (importable into URLab)
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
# Note: to use the ZMQ transport layer standalone without lerobot,
#       a minimal lazy-import patch was applied to the vendored copy's
#       brainbot_core/__init__.py (see the header comment in that file).

# 3) Optional: MuJoCo physics backend, faiss vector index (already in requirements)
```

## 4. Running

```bash
# Seven-layer closed-loop demo (ask path + explore path, with learning convergence)
python scripts/demo_closed_loop.py

# Run all tests
launch/run_all_tests.sh

# Start the AI policy server (Brainbot ZMQ REP, tcp://127.0.0.1:5555)
python -m src.brainbot_adapters.policy_server --port 5555

# One-click launch of remote/hub hosts (see launch/*.sh)
launch/launch_remote.sh
```

## 5. Component Selection & Implementation Status

| Layer | Documented choice | This project's implementation |
|---|---|---|
| Core control | bingogome/brainbot | ✅ Installed; adapters reuse the upstream ZMQ/msgpack protocol |
| Simulation | URLab + MuJoCo | ⚠️ URLab requires a UE 5.7 C++ project (not buildable in a GUI/GPU-less sandbox); provides MJCF model + MuJoCo physics bridge |
| RL interface | AMD Schola | ⚠️ Requires UE; a Gymnasium-compatible integration point is kept (mujoco_bridge) |
| World model | Mask World Model (ICML'26) | ⚠️ Requires GPU + GE-Base weights; provides an `mwm` backend adapter + online Bayesian dynamics fallback |
| Exploration engine | Explauto | ⚠️ Not compatible with Python 3.12; provides an explauto adapter entry + equivalent interest-model implementation |
| Action planning | Dream-MPC (ICML'26) | ⚠️ Requires a GPU world model; provides a lightweight shooting-MPC equivalent |
| Vector memory | Chroma / FAISS | ✅ faiss-cpu / numpy dual backend |
| Speech | Whisper + emotion recognition | ⚠️ Whisper requires torch; provides a whisper adapter + rule-based backend |
| Multimodal | CLIP | Adapter entry (vision encoder with numpy/torch dual backend) |

**Note**: every ⚠️ item keeps a real-deployment adapter interface (`backend="auto"` auto-detection). Once the corresponding weights are installed in a GPU/UE environment, the implementation switches to the real one automatically; inside the CPU sandbox, an equivalent fallback backend keeps the whole closed loop runnable and verifiable.

## 6. Verification Checklist (mapped to the spec)

| # | Verification item | Result |
|---|---|---|
| 1 | Service startup | ✅ 31 tests passing |
| 2 | Policy-server ZMQ handshake | ✅ covered by end-to-end test (ping + get_action) |
| 3 | Vision encoder outputs 256-d vector | ✅ test_vision_encoder |
| 4 | World-model prediction error computable | ✅ test_prediction_error |
| 5 | Curiosity trigger (unknown object) | ✅ dual-path closed-loop tests |
| 6 | Dual-path decision (ask / explore) | ✅ ask=7.0 / explore=5.4 boundary verified |
| 7 | Safety-gate hard interrupt | ✅ HALT/WAIT/RECHARGE/SOFT_STOP |
| 8 | Episodic memory read/write <50ms | ✅ vector-store retrieval + closed-loop writes |
| 9 | Simulation model loading | ✅ MJCF parsed by MuJoCo + physics stepping |
| 10 | End-to-end closed loop | ✅ perceive → decide → act → learning convergence |

## 7. Key Technical Behaviors

- **Prediction-variance trigger**: an unknown object's world-model prior variance is high → triggers "awareness of the unknown"; once learned, known samples are seeded so the variance collapses → no repeated triggering.
- **Dual-path decision**: a scoring table (risk / human idle time / cost / task / object dynamicity); ≥6 asks the human, <6 explores autonomously.
- **Lifelong learning**: categories acquired via asking or exploring are written into five-tuple episodic memory, and are "known" from the next round onward — the robot keeps learning without re-asking.
