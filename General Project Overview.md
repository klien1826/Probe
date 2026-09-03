Probe

Proactive Exploration &amp; Inquiry Engine for Embodied AI
Robots that know what they don't know — and actively seek to learn.

Probe is an open-source cognitive architecture that fundamentally reimagines how robots learn. Instead of passively executing pre-trained policies, Probe empowers robots with intrinsic curiosity — the ability to perceive their own uncertainty, decide whether to ask a human or explore autonomously, and permanently retain what they learn.

Why Probe?
Today's embodied AI systems are passive: they execute what they're trained on, and fail silently when encountering the unknown. Probe flips this paradigm.
<img width="4850" height="7958" alt="deepseek_mermaid_20260902_ffd1fc" src="https://github.com/user-attachments/assets/6d7eea3b-fb49-40c7-b198-9c0916b76d5c" />


Traditional Robots	Probe
Learning mode	Offline pre-training	Online, lifelong learning
Unknown objects	Hallucinate or fail	Trigger uncertainty, then ask or explore
Memory	None or static	Structured, compressible, evolving
Safety	Afterthought	1kHz hard real-time safety gate
Core Architecture (7 Layers)
Multimodal Interaction Layer — Bidirectional human-robot communication with active questioning

Cognitive Decision Core — Uncertainty trigger + curiosity generation + dual-path decision engine (ask / explore)

World Model Layer — Mask World Model for semantic prediction with prediction error computation

Multimodal Perception Layer — Vision, audio, proprioception, force-torque sensing

Exploration & Motion Planning Layer — Explauto-driven curiosity management + Dream-MPC imaginary planning

Safety & Arbitration Layer — 1kHz hard real-time safety gate + cognitive safety仲裁

Long-term Memory & Evolution Layer — 5-tuple vector memory + L1→L2→L3 hierarchical compression + knowledge updating
<img width="6320" height="5075" alt="deepseek_mermaid_20260902_6732a3" src="https://github.com/user-attachments/assets/f2d13ae4-89ea-487a-8ab1-ba6fea01a170" />

Key Innovations
Uncertainty Trigger (not "self-awareness" — engineering, not philosophy): Prediction variance exceeds threshold → triggers curiosity


Dual-path Decision Engine (not "information gain maximization" — a rule-based scorecard): 5-dimension scoring (risk, human availability, exploration cost, task priority, object dynamics) → decide: Ask human or Explore autonomously

Fast/Slow Dual-system: 10-20Hz fast reflex (safety) + 1-2Hz slow cognition (reasoning)

Hierarchical Memory Compression: L1 detail cache → L2 abstract rules → L3 core knowledge — no "forgetting," just compression


<img width="3144" height="5724" alt="deepseek_mermaid_20260902_98b64d" src="https://github.com/user-attachments/assets/d1a08053-93b7-4431-b81d-064d77079562" />

Tech Stack
Component	Technology
Core Framework	bingogome/brainbot
Simulation	Unreal Robotics Lab (URLab) + MuJoCo
World Model	Mask World Model (ICML 2026)
Curiosity Engine	Explauto (Inria FLOWERS)
Motion Planning	Dream-MPC
Vector Memory	Chroma / FAISS
Speech	Whisper + Emotion Recognition
Multimodal Alignment	CLIP
Quick Start
bash
git clone https://github.com/[your-username]/Probe.git
cd Probe
conda create -n probe python=3.10 -y
conda activate probe
pip install -e .
# See /launch/ for startup scripts
Status
🚧 Active Development — Architecture specification complete, core modules under implementation.

License
Apache 2.0

Citation
If you use Probe in your research, please cite:

text
@software{Probe2026,
  author = {[Your Name]},
  title = {Probe: Proactive Exploration and Inquiry Engine for Embodied AI},
  year = {2026},
  url = {https://github.com/[your-username]/Probe}
}








探知 (Probe)
主动求知型具身智能引擎

让机器人知道自己不知道，并主动去学。

探知 是一套开源的机器人认知架构，从根本上重新定义了机器人的学习方式。它不再被动执行预训练策略，而是赋予机器人内在好奇心——感知自身的不确定性，自主决定是问人还是自己去探索，并把学到的知识永久记住、持续迭代。

为什么做探知？
今天的具身智能系统都是 “被动型” 的：训练什么就会什么，遇到没见过的就硬猜或直接出错。探知彻底翻转了这个范式。

传统机器人	探知 (Probe)
学习方式	离线预训练	在线、终身学习
遇到未知	幻觉或失败	触发不确定性，主动问或探索
记忆系统	无或静态	结构化、可压缩、可进化
安全保障	事后补救	1kHz 硬实时安全门禁
核心架构（七层）
多模态交互层 — 双向人机交互，机器人可主动发起提问

认知决策核心层 — 不确定性触发 + 好奇心生成 + 双路径决策（问人/探索）

世界模型理解层 — 语义掩码预测 + 预测误差计算

多模态感知编码层 — 视觉、听觉、本体感觉、力觉

自主探索与运动规划层 — Explauto 好奇管理 + Dream-MPC 想象规划

安全与仲裁层 — 1kHz 硬实时安全门禁 + 认知安全仲裁

长效记忆与进化层 — 五元组向量记忆 + L1→L2→L3 分层压缩 + 知识更新

三大独家创新
不确定性触发器（不是“自我觉知”——是工程，不是哲学）：预测方差超阈值 → 触发好奇

双路径决策引擎（不是“信息增益最大化”——是规则评分表）：5维度评分（风险、人类空闲度、探索成本、任务优先级、物体动态性）→ 决策：问人 or 自主探索

快慢双系统：10-20Hz 快反射（安全）+ 1-2Hz 慢认知（推理）

技术栈
组件	技术选型
核心框架	bingogome/brainbot
仿真环境	Unreal Robotics Lab + MuJoCo
世界模型	Mask World Model (ICML 2026)
好奇引擎	Explauto (Inria FLOWERS)
运动规划	Dream-MPC
向量记忆	Chroma / FAISS
语音交互	Whisper + 情绪识别
多模态对齐	CLIP

















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
