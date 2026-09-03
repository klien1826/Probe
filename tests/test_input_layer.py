"""输入层单元测试。"""
import numpy as np

from src.core.config import Config
from src.input_layer.audio_processor import AudioProcessor
from src.input_layer.proprioception import Proprioception
from src.input_layer.vision_encoder import VisionEncoder


def _synthetic_rgb(size=(64, 64)):
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 255, size=(size[0], size[1], 3), dtype=np.uint8)
    # 画一个"物体"色块
    rgb[20:44, 20:44] = [200, 30, 30]
    return rgb


def test_vision_encoder_outputs_256dim():
    enc = VisionEncoder(Config(), backend="numpy")
    vec, props = enc.encode_scene(_synthetic_rgb())
    assert vec.shape == (256,)
    assert np.isfinite(vec).all()
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3)
    assert isinstance(props, list)


def test_vision_encoder_deterministic():
    enc = VisionEncoder(Config(), backend="numpy")
    v1, _ = enc.encode_scene(_synthetic_rgb())
    v2, _ = enc.encode_scene(_synthetic_rgb())
    assert np.allclose(v1, v2)


def test_audio_processor_rules():
    ap = AudioProcessor(Config(), backend="rule")
    r = ap.process(text="请问这个红色的物体是什么？")
    assert r["text"] == "请问这个红色的物体是什么？"
    assert r["intent"] == "question"
    assert r["emotion"] == "curious"


def test_proprioception_read():
    pr = Proprioception(Config())
    s = pr.read({"battery_pct": 55.0, "collision": True, "pose": (1, 2, 0.5)})
    assert s.battery_pct == 55.0
    assert s.collision is True
    assert s.pose == (1, 2, 0.5)
