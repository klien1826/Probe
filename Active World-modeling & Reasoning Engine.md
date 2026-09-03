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
