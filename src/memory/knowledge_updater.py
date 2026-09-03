"""长效记忆层 · 知识更新与冲突消解。

触发: 新观测与 L3 核心认知矛盾
流程: 记录冲突 → 主动提问确认 → 更新 L3 → 记录变更日志
"""
from __future__ import annotations

import time
from typing import Any, Optional

from .vector_store import VectorStore


class KnowledgeUpdater:
    def __init__(self, store: VectorStore):
        self.store = store
        self.conflicts: list[dict[str, Any]] = []
        self.changelog: list[dict[str, Any]] = []

    def check_conflict(self, object_id: str, new_attrs: dict[str, Any]) -> Optional[dict[str, Any]]:
        """检测新观测与既有 L3 核心认知是否矛盾。"""
        stored = self.store.get_by_object_id(object_id)
        if stored is None or stored.get("tier") != "L3":
            return None
        old_cat = stored.get("object_attrs", {}).get("category")
        new_cat = new_attrs.get("category")
        if old_cat and new_cat and old_cat != new_cat:
            conflict = {
                "object_id": object_id,
                "old": old_cat,
                "new": new_cat,
                "timestamp": time.time(),
                "resolved": False,
            }
            self.conflicts.append(conflict)
            return conflict
        return None

    def confirm_update(
        self,
        object_id: str,
        new_attrs: dict[str, Any],
        answer: str = "",
        scene_vector: Optional[Any] = None,
    ) -> Optional[dict[str, Any]]:
        """用户确认后更新 L3 核心认知，并写变更日志。"""
        stored = self.store.get_by_object_id(object_id)
        if stored is None:
            return None
        old = dict(stored)
        self.store.update_memory(
            stored["id"],
            object_attrs={**stored.get("object_attrs", {}), **new_attrs},
            qa_knowledge={
                "question": stored.get("qa_knowledge", {}).get("question", "conflict"),
                "answer": answer or stored.get("qa_knowledge", {}).get("answer"),
                "timestamp": time.time(),
            },
        )
        # 标记相关冲突已解决
        for c in self.conflicts:
            if c["object_id"] == object_id:
                c["resolved"] = True
        entry = {
            "timestamp": time.time(),
            "object_id": object_id,
            "old": old,
            "new": new_attrs,
            "reason": "user_confirmed",
        }
        self.changelog.append(entry)
        return entry
