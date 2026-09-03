"""理解层单元测试：世界模型 / 预测误差 / 预测方差触发器。"""
import numpy as np

from src.core.config import Config
from src.core.types import ObjectPercept
from src.understanding_layer.prediction_error import PredictionError
from src.understanding_layer.uncertainty_trigger import UncertaintyTrigger
from src.understanding_layer.world_model import WorldModel, build_latent


def _obj(obj_id="obj_x", feature=None):
    f = feature if feature is not None else np.ones(16, dtype=np.float32)
    return ObjectPercept(object_id=obj_id, category=None, color=(1, 0, 0),
                         size=0.5, state="static", feature=f)


def test_world_model_unknown_object_high_variance():
    wm = WorldModel(Config(), backend="lightweight")
    z = np.ones(32, dtype=np.float32)
    a = np.zeros(3, dtype=np.float32)
    # 未知物体：高先验方差 → 预测方差大
    _, var_unknown = wm.predict(z, a, "obj_unknown")
    # 无物体（场景）：普通先验 → 方差小
    _, var_scene = wm.predict(z, a, None)
    assert var_unknown > var_scene * 3, (var_unknown, var_scene)


def test_world_model_learns_known_object():
    wm = WorldModel(Config(), backend="lightweight")
    z = np.ones(32, dtype=np.float32) * 0.5
    a = np.ones(3, dtype=np.float32) * 0.1
    # 学习：静态物体 → delta 近 0
    for _ in range(20):
        wm.update(z, a, z.copy(), "obj_known", commit=True)
    _, var = wm.predict(z, a, "obj_known")
    assert wm.familiarity("obj_known") > 0.5
    assert var < 1.0


def test_prediction_error_collision_weight():
    pe = PredictionError(Config())
    z = np.zeros(32)
    r1 = pe.compute(z, z + 0.5, variance=1.0, collision=False)
    r2 = pe.compute(z, z + 0.5, variance=1.0, collision=True)
    assert r2.prediction_error >= r1.prediction_error
    assert r1.prediction_error <= 1.0


def test_uncertainty_trigger_thresholds():
    trig = UncertaintyTrigger(Config())
    from src.understanding_layer.prediction_error import PredictionError
    pe = PredictionError(Config())
    pred = pe.compute(np.zeros(32), np.zeros(32), variance=0.0, collision=False)
    # 低误差 + 已知 → 不触发
    s = trig.evaluate(pred, object_known=True, object_id="x")
    assert not s.trigger
    # 低误差 + 未知物体 → 触发（对象新颖性）
    s2 = trig.evaluate(pred, object_known=False, object_id="y")
    assert s2.trigger
    assert s2.reason == "object_unknown_in_memory"
    # 高误差 → 触发（预测误差路径）
    pred_hi = pe.compute(np.zeros(32), np.ones(32) * 2.0, variance=0.0, collision=False)
    s3 = trig.evaluate(pred_hi, object_known=True, object_id="x")
    assert s3.trigger


def test_build_latent_dim():
    z = build_latent(np.zeros(5), np.zeros(16), dim=32)
    assert z.shape == (32,)
