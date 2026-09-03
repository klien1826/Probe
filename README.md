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
