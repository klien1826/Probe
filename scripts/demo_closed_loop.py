#!/usr/bin/env python3
"""七层认知闭环演示。

演示两个完整闭环（对应双路径决策）:
  A. 提问路径: 人类在场且空闲 → 未知物体 → 主动提问 → 人类回答 → 终身学习
  B. 探索路径: 人类不在场 + 带标签物体 → 自主探索 → inspect 读标签 → 终身学习

输出: 结构化决策日志 + 记忆成长汇总（供可视化）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.brain import CognitiveBrain
from src.core.config import Config
from simulation.world_sim import WorldSim

STEPS = 40


def run_scenario(name, brain, sim, steps=STEPS):
    print(f"\n{'='*64}\n场景: {name}\n{'='*64}")
    obs = sim.reset()
    trace = []
    for i in range(steps):
        action = brain.think(obs)
        trace.append({
            "step": i,
            "mode": brain._last_mode.value,
            "uncertainty": round(float(brain._last_uncertainty), 3),
            "decision": brain._last_decision.path.value if brain._last_decision else None,
            "question": action.text,
            "action": action.actions.get("action"),
            "target": action.target_id,
        })
        obs = sim.step(action)
        # 打印关键事件
        if action.text and "safety" not in action.text:
            print(f"  [step {i:02d}] 决策={trace[-1]['decision']:<9} "
                  f"模式={trace[-1]['mode']:<16} 不确定性={trace[-1]['uncertainty']:.2f} "
                  f"→ {action.text}")
    mem = brain.memory_summary()
    print(f"\n  记忆成长: 总量={mem['total']} (L1={mem['L1']}, L2={mem['L2']}, L3={mem['L3']})")
    return {"scenario": name, "trace": trace, "memory": mem}


def main():
    # ---- 场景 A: 提问路径 ----
    brain_a = CognitiveBrain(Config())
    sim_a = WorldSim(seed=1, human_present=True, human_idle=True)
    res_a = run_scenario("A · 提问路径（人类在场 → 主动提问学习）", brain_a, sim_a)

    # ---- 场景 B: 探索路径 ----
    brain_b = CognitiveBrain(Config())
    sim_b = WorldSim(seed=2, human_present=False, human_idle=False, objects=[
        {"id": "obj_labelled", "category": "bottle", "color": (0.1, 0.3, 0.9), "size": 0.3,
         "state": "static", "danger": 0.1, "touchable": True, "pos": (1.0, 0.0),
         "label_visible": True},
    ])
    res_b = run_scenario("B · 探索路径（人类不在场 → 自主探索读标签）", brain_b, sim_b)

    out = {"scenarios": [res_a, res_b]}
    path = Path(__file__).resolve().parent.parent / "scripts" / "closed_loop_trace.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 轨迹已保存: {path}")

    # 汇总: 提问次数与学习收敛
    qa = [t for t in res_a["trace"] if t["question"] and t["question"].startswith("请问")]
    print(f"\n  场景A 提问次数: {len(qa)}（速率限制 ≤3/分钟，学成后不再重复提问）")


if __name__ == "__main__":
    main()
