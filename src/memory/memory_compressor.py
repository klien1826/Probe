"""长效记忆层 · 分层压缩（替代"遗忘"）。

L1 细节缓存: 最近 100 条原始五元组
L2 抽象归纳: K-Means 聚类 → 抽象规则 (上限 1000 条)
L3 核心认知: 高置信度长期知识 (验证次数 >= 3)

压缩周期: 每 24 小时执行一次"睡眠压缩"（可手动触发 / 定时器触发）。
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..core.config import Config
from .vector_store import VectorStore

L1_CAP = 100
L2_CAP = 1000
L3_VERIFY_THRESHOLD = 3


class MemoryCompressor:
    def __init__(self, store: VectorStore, cfg: Config | None = None):
        self.store = store
        self.last_compress_ts = time.time()
        self.last_summary: dict[str, Any] = {}

    def maybe_compress(self, force: bool = False) -> dict[str, Any]:
        """每 24h 自动触发一次睡眠压缩；force=True 立即执行。"""
        if not force and (time.time() - self.last_compress_ts) < 24 * 3600:
            return {"skipped": True}
        self.last_compress_ts = time.time()
        return self.compress()

    def compress(self) -> dict[str, Any]:
        l1 = self.store.all(tier="L1")
        summary: dict[str, Any] = {
            "before": self.store.count(),
            "l1_dedup": 0,
            "l2_clusters": 0,
            "l3_promoted": 0,
        }

        # ---- 1) L1 上限裁剪 + 聚类归纳为 L2 ----
        if len(l1) > L1_CAP:
            overflow = l1[: len(l1) - L1_CAP]     # 最旧部分
            summary["l1_dedup"] = len(overflow)
            self._cluster_to_l2(overflow)
            for m in overflow:
                self.store.remove(m["id"])

        # ---- 2) L2 验证达标 → 提升为 L3 核心认知 ----
        for m in self.store.all(tier="L2"):
            if m["verify_count"] >= L3_VERIFY_THRESHOLD:
                self.store.update_memory(m["id"], tier="L3")
                summary["l3_promoted"] += 1

        # ---- 3) L2 上限保护 ----
        l2 = self.store.all(tier="L2")
        if len(l2) > L2_CAP:
            for m in l2[: len(l2) - L2_CAP]:
                self.store.remove(m["id"])

        summary["after"] = self.store.count()
        self.last_summary = summary
        return summary

    def _cluster_to_l2(self, memories: list[dict[str, Any]]):
        """把一批 L1 记忆聚类为抽象规则（L2）。"""
        if not memories:
            return
        from sklearn.cluster import KMeans
        mat = np.stack([m["scene_vector"] for m in memories])
        n = min(len(memories), max(1, len(memories) // 5))
        n = min(n, len(memories))
        labels = KMeans(n_clusters=n, n_init=4, random_state=0).fit_predict(mat)
        for c in range(n):
            members = [m for m, lb in zip(memories, labels) if lb == c]
            if not members:
                continue
            centroid = np.mean(np.stack([m["scene_vector"] for m in members]), axis=0)
            # 类别/答案投票
            cats = [m["object_attrs"].get("category") for m in members if m["object_attrs"].get("category")]
            cat = max(set(cats), key=cats.count) if cats else None
            answers = [m["qa_knowledge"].get("answer") for m in members if m["qa_knowledge"].get("answer")]
            answer = max(set(answers), key=answers.count) if answers else None
            touchable = sum(1 for m in members if m["safety_rules"].get("touchable"))
            self.store.add({
                "object_id": f"L2_cluster_{c}",
                "scene_vector": centroid,
                "object_attrs": {"category": cat, "color": None, "size": None, "state": "abstract"},
                "qa_knowledge": {"question": "abstract", "answer": answer, "timestamp": time.time()},
                "safety_rules": {"touchable": touchable >= len(members) / 2, "keep_distance": 0.0},
                "usage_scenario": {"task_type": "abstract", "environment": "abstract"},
                "tier": "L2",
                "verify_count": len(members),
            })
