"""判断层 · 情景记忆检索。

输入: 场景特征向量 (512维)
输出: Top-K 相似记忆 (K=5)
检索延迟: ≤ 50ms
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.config import Config
from ..memory.vector_store import VectorStore


class MemoryRetriever:
    def __init__(self, store: VectorStore, cfg: Config | None = None, top_k: int = 5):
        self.store = store
        self.top_k = top_k

    def retrieve(
        self,
        scene_vector: np.ndarray,
        object_id: Optional[str] = None,
        k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """按场景特征检索 Top-K 情景记忆。"""
        k = k or self.top_k
        return self.store.query(scene_vector, k=k)

    def lookup_object(self, object_id: str) -> Optional[dict[str, Any]]:
        """精确查某物体是否有已存记忆（判断是否"已知"）。"""
        return self.store.get_by_object_id(object_id)

    def is_known(self, object_id: str, scene_vector: np.ndarray,
                 similarity_threshold: float = 0.85) -> tuple[bool, float]:
        """物体是否已知：优先精确 ID 匹配，其次向量相似度。"""
        if self.store.get_by_object_id(object_id) is not None:
            return True, 1.0
        hits = self.store.query(scene_vector, k=1)
        if hits:
            return True, float(hits[0]["similarity"])
        return False, 0.0
