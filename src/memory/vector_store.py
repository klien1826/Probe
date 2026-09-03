"""长效记忆层 · 五元组向量记忆库（Chroma/FAISS 封装 + numpy 降级）。

记忆条目结构:
{
  "id": str,
  "object_id": str,
  "scene_vector": np.ndarray (512,),
  "object_attrs": {"category","color","size","state"},
  "qa_knowledge": {"question","answer","timestamp"},
  "safety_rules": {"touchable","keep_distance"},
  "usage_scenario": {"task_type","environment"},
  "tier": "L1"|"L2"|"L3",
  "verify_count": int,
  "timestamp": float,
}

双后端:
  - backend="faiss": faiss-cpu，归一化 IP（余弦）
  - backend="numpy": 确定性暴力余弦检索（无额外依赖）
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional

import numpy as np

from ..core.config import Config


class VectorStore:
    def __init__(self, dim: int = 512, backend: str = "auto"):
        self.dim = dim
        self.backend = backend if backend != "auto" else self._pick_backend()
        self._memories: list[dict[str, Any]] = []
        self._faiss = None
        self._index = None
        if self.backend == "faiss":
            self._init_faiss()

    def _pick_backend(self) -> str:
        try:
            import faiss  # noqa: F401
            return "faiss"
        except Exception:
            return "numpy"

    def _init_faiss(self):
        import faiss
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(self.dim)

    # ------------------------------------------------------------------
    def add(self, memory: dict[str, Any]) -> str:
        """写入一条记忆。scene_vector 需为 self.dim 维。"""
        mem_id = memory.get("id") or f"mem_{uuid.uuid4().hex[:8]}"
        vec = np.asarray(memory.get("scene_vector"), dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            vec = np.resize(vec, self.dim)
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm > 1e-9 else vec
        memory = dict(memory)
        memory.update({
            "id": mem_id,
            "scene_vector": vec,
            "tier": memory.get("tier", "L1"),
            "verify_count": int(memory.get("verify_count", 0)),
            "timestamp": float(memory.get("timestamp", time.time())),
        })
        self._memories.append(memory)
        if self._index is not None:
            self._index.add(vec[None, :])
        return mem_id

    def query(self, vector: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        """按向量检索 Top-K，返回带 similarity(0~1) 的记忆。"""
        if not self._memories:
            return []
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            vec = np.resize(vec, self.dim)
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm > 1e-9 else vec
        k = max(1, min(k, len(self._memories)))

        if self._index is not None:
            scores, idxs = self._index.search(vec[None, :], k)
            idxs = idxs[0].tolist()
            scores = scores[0].tolist()
        else:
            mat = np.stack([m["scene_vector"] for m in self._memories])
            scores = mat @ vec
            idxs = np.argsort(-scores)[:k].tolist()
            scores = scores[idxs].tolist()

        results = []
        for i, s in zip(idxs, scores):
            mem = dict(self._memories[i])
            mem["similarity"] = float(max(0.0, min(1.0, s)))
            results.append(mem)
        return results

    def get_by_object_id(self, object_id: str) -> Optional[dict[str, Any]]:
        for m in self._memories:
            if m.get("object_id") == object_id:
                return dict(m)
        return None

    def get(self, mem_id: str) -> Optional[dict[str, Any]]:
        for m in self._memories:
            if m.get("id") == mem_id:
                return dict(m)
        return None

    def all(self, tier: Optional[str] = None) -> list[dict[str, Any]]:
        if tier is None:
            return [dict(m) for m in self._memories]
        return [dict(m) for m in self._memories if m.get("tier") == tier]

    def remove(self, mem_id: str) -> bool:
        for i, m in enumerate(self._memories):
            if m.get("id") == mem_id:
                self._memories.pop(i)
                self._rebuild_index()
                return True
        return False

    def update_memory(self, mem_id: str, **fields) -> bool:
        for m in self._memories:
            if m.get("id") == mem_id:
                m.update(fields)
                if "scene_vector" in fields:
                    self._rebuild_index()
                return True
        return False

    def count(self) -> int:
        return len(self._memories)

    def _rebuild_index(self):
        if self._index is not None:
            self._index.reset()
            if self._memories:
                mat = np.stack([m["scene_vector"] for m in self._memories])
                self._index.add(mat)
