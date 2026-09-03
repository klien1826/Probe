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








# Robot Brain · 主动求知型具身机器人认知架构

基于 **bingogome/brainbot** 框架的七层认知闭环工程实现。

> 机器不再只靠预训练看懂世界，而是通过「主动感知未知 → 决策问人或探索 → 结构化记忆」在真实世界中持续学习。

## 1. 架构总览

```
┌─────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
│  输入层      │ → │   理解层      │ → │    探索层(核心)   │ → │   判断层      │
│ 视觉/音频/  │   │ MWM 世界模型  │   │ 好奇心引擎        │   │ 安全门禁(快)  │
│ 本体感知    │   │ 预测误差      │   │ Dream-MPC 预案    │   │ 认知仲裁(慢)  │
│             │   │ 方差触发      │   │ 双路径决策        │   │ 记忆检索      │
└─────────────┘   └──────────────┘   │ 提问生成          │   └──────────────┘
                                      └────────┬─────────┘          │
┌─────────────┐   ┌──────────────┐             ▼                    ▼
│ Brainbot    │ ← │ 长效记忆层    │ ←──────── 七层闭环(核心编排 brain.py)
│ 适配器      │   │ 向量库/压缩/  │    感知→预测→触发→决策→仲裁→记忆→动作
│ ZMQ 策略服务│   │ 冲突消解      │
└─────────────┘   └──────────────┘
```

**快/慢双系统**：物理安全门禁（快系统，仿真 20Hz / 真实部署 1kHz 硬实时）可随时中断慢系统的认知决策；世界模型/认知决策为慢系统（1–2Hz）。

## 2. 目录结构

```
robot_brain/
├── config/                  # brainbot_config / curiosity_thresholds / safety_rules
├── src/
│   ├── core/                # brain.py（七层闭环编排）+ types + config 加载
│   ├── input_layer/         # 视觉编码(256维) / 音频(ASR+情绪+意图) / 本体感知
│   ├── understanding_layer/ # 世界模型(MWM适配+贝叶斯降级) / 预测误差 / 方差触发器
│   ├── exploration_layer/   # 好奇心引擎 / 动作预案 / 双路径决策 / 提问生成
│   ├── judgment_layer/      # 安全门禁 / 认知仲裁 / 记忆检索
│   ├── memory/              # 向量库(faiss/numpy) / L1→L2→L3压缩 / 知识冲突消解
│   └── brainbot_adapters/   # ZMQ 策略服务器 / 命令提供者 / 载荷转换
├── simulation/
│   ├── world_sim.py         # 轻量 2D 世界（闭环演示主载体）
│   ├── mujoco_bridge.py     # MuJoCo 物理后端（差速底盘）
│   └── mujoco_models/       # MJCF 机器人模型（URLab 可导入）
├── tests/                   # 31 个单元 + 端到端测试
├── launch/                  # launch_remote.sh / launch_hub.sh / run_all_tests.sh
└── scripts/demo_closed_loop.py   # 端到端闭环演示
```

## 3. 安装

```bash
# 1) Python 环境（≥3.10，本工程验证于 3.12）
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Brainbot 上游（ZMQ 传输/消息协议）
git clone https://github.com/bingogome/brainbot.git vendor/brainbot
pip install -e vendor/brainbot
# 说明: 为在无 lerobot 环境独立使用 ZMQ 传输层，对 vendor 副本的
#       brainbot_core/__init__.py 做了最小惰性导入改造（详见该文件头注释）。

# 3) 可选：MuJoCo 物理后端、faiss 向量索引（requirements 已含）
```

## 4. 运行

```bash
# 七层闭环演示（提问路径 + 探索路径，含学习收敛）
python scripts/demo_closed_loop.py

# 全部测试
launch/run_all_tests.sh

# 启动 AI 策略服务器（Brainbot ZMQ REP，tcp://127.0.0.1:5555）
python -m src.brainbot_adapters.policy_server --port 5555

# 一键启动 remote/hub（见 launch/*.sh）
launch/launch_remote.sh
```

## 5. 组件选型与落地状态

| 层级 | 文档选型 | 本工程落地 |
|---|---|---|
| 核心控制 | bingogome/brainbot | ✅ 已安装，适配器复用上游 ZMQ/msgpack 协议 |
| 仿真 | URLab + MuJoCo | ⚠️ URLab 需 UE 5.7 C++ 工程（沙箱无 GUI/GPU 无法构建）；提供 MJCF 模型 + MuJoCo 物理桥 |
| RL 接口 | AMD Schola | ⚠️ 需 UE；保留 Gymnasium 兼容接入点（mujoco_bridge） |
| 世界模型 | Mask World Model (ICML'26) | ⚠️ 需 GPU+GE-Base 权重；提供 mwm 后端适配 + 在线贝叶斯动力学降级 |
| 探索引擎 | Explauto | ⚠️ 该库不兼容 Py3.12；提供 explauto 适配入口 + 兴趣模型等价实现 |
| 动作规划 | Dream-MPC (ICML'26) | ⚠️ 需 GPU 世界模型；提供 shooting-MPC 轻量等价实现 |
| 向量记忆 | Chroma / FAISS | ✅ faiss-cpu / numpy 双后端 |
| 语音 | Whisper + 情绪识别 | ⚠️ whisper 需 torch；提供 whisper 适配 + 规则后端 |
| 多模态 | CLIP | 适配入口（视觉编码器 numpy/torch 双后端） |

**说明**：所有 ⚠️ 项均保留了真实部署的适配接口（`backend="auto"` 自动探测），在 GPU/UE 环境安装对应权重后即切换到真实实现；CPU 沙箱内以等价降级后端保证整条闭环可运行、可验证。

## 6. 验证清单（对照文档）

| # | 验证项 | 结果 |
|---|---|---|
| 1 | 服务启动 | ✅ 31 项测试通过 |
| 2 | 策略服务器 ZMQ 握手 | ✅ 端到端测试覆盖（ping + get_action） |
| 3 | 视觉编码输出 256 维 | ✅ test_vision_encoder |
| 4 | 世界模型预测误差可计算 | ✅ test_prediction_error |
| 5 | 好奇触发（未知物体） | ✅ 双路径闭环测试 |
| 6 | 双路径决策（问人/探索） | ✅ ask=7.0 / explore=5.4 分界验证 |
| 7 | 安全门禁硬中断 | ✅ HALT/WAIT/RECHARGE/SOFT_STOP |
| 8 | 情景记忆读写 <50ms | ✅ 向量库检索 + 闭环写入 |
| 9 | 仿真模型加载 | ✅ MJCF 经 MuJoCo 解析 + 物理步进 |
| 10 | 端到端闭环 | ✅ 感知→决策→执行→学习收敛 |

## 7. 关键技术行为

- **预测方差触发**：未知物体的世界模型先验方差高 → 触发"未知感知"；学会后播种已知样本使方差坍缩 → 不再重复触发。
- **双路径决策**：评分表（风险/人类空闲/成本/任务/动态性），≥6 问人，<6 探索。
- **终身学习**：提问/探索获得的类别写入五元组记忆，下一轮即"已知"，机器人持续学习且不重复提问。
