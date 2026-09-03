"""长效记忆层单元测试：向量库 / 分层压缩 / 知识冲突消解。"""
import numpy as np

from src.core.config import Config
from src.memory.knowledge_updater import KnowledgeUpdater
from src.memory.memory_compressor import L1_CAP, L3_VERIFY_THRESHOLD, MemoryCompressor
from src.memory.vector_store import VectorStore


def _mem(obj_id, vec, cat="cup", tier="L1", verify=1):
    return {
        "object_id": obj_id,
        "scene_vector": vec,
        "object_attrs": {"category": cat, "color": (1, 0, 0), "size": 0.5, "state": "static"},
        "qa_knowledge": {"question": "q", "answer": cat, "timestamp": 0.0},
        "safety_rules": {"touchable": True, "keep_distance": 0.0},
        "usage_scenario": {"task_type": "t", "environment": "e"},
        "tier": tier,
        "verify_count": verify,
    }


def test_vector_store_add_query():
    store = VectorStore(dim=512, backend="numpy")
    store.add(_mem("a", np.ones(512), cat="cup"))
    store.add(_mem("b", np.ones(512) * 0.9, cat="box"))
    assert store.count() == 2
    hits = store.query(np.ones(512), k=2)
    assert hits[0]["object_id"] == "a"
    assert store.get_by_object_id("a")["object_attrs"]["category"] == "cup"


def test_memory_compressor_promotes_l3():
    store = VectorStore(dim=512, backend="numpy")
    for i in range(120):
        store.add(_mem(f"o{i}", np.ones(512) + i * 0.001))
    comp = MemoryCompressor(store, Config())
    summary = comp.compress()
    assert summary["l1_dedup"] >= 20              # L1 裁剪
    # 聚类产生抽象规则（L2 或已提升至 L3）
    assert store.all(tier="L2") or store.all(tier="L3")
    assert comp.last_summary["after"] <= store.count()


def test_memory_compressor_l3_threshold():
    store = VectorStore(dim=512, backend="numpy")
    store.add(_mem("core", np.ones(512), cat="cup", tier="L2", verify=L3_VERIFY_THRESHOLD))
    comp = MemoryCompressor(store, Config())
    comp.compress()
    assert store.get_by_object_id("core")["tier"] == "L3"


def test_knowledge_updater_conflict_and_update():
    store = VectorStore(dim=512, backend="numpy")
    store.add(_mem("obj", np.ones(512), cat="cup", tier="L3", verify=3))
    updater = KnowledgeUpdater(store)
    conflict = updater.check_conflict("obj", {"category": "mug"})
    assert conflict is not None and not conflict["resolved"]
    entry = updater.confirm_update("obj", {"category": "mug"}, answer="它是马克杯")
    assert entry is not None
    assert store.get_by_object_id("obj")["object_attrs"]["category"] == "mug"
    assert updater.changelog and updater.conflicts[0]["resolved"] is True
