"""输入层 · 音频处理器。

输入: 麦克风音频 (16kHz 单声道)
输出: {"text": str, "emotion": str, "intent": str, "confidence": float}

双后端:
  - backend="whisper": 真实 Whisper ASR + 轻量情绪/意图分类（需 GPU 或较慢 CPU）
  - backend="rule": 确定性规则后端（关键词匹配）—— 无 GPU 环境的可验证替代
频率: 按需触发
"""
from __future__ import annotations

from typing import Any, Optional

from ..core.config import Config

EMOTION_KEYWORDS = {
    "angry": ["生气", "愤怒", "气死", "angry", "mad"],
    "happy": ["开心", "高兴", "太好了", "happy", "great"],
    "sad": ["难过", "伤心", "失望", "sad"],
    "curious": ["是什么", "怎么", "为什么", "what", "how", "why", "好奇"],
    "neutral": [],
}

INTENT_KEYWORDS = {
    "inform": ["这是", "那是", "叫做", "是猫", "是狗", "it is", "it's", "called"],
    "command": ["去", "拿", "放", "走", "来", "go", "take", "come", "bring"],
    "question": ["?", "吗", "什么", "怎么", "why", "what", "how"],
    "answer": ["是", "对", "不是", "yes", "no"],
}


class AudioProcessor:
    def __init__(self, cfg: Config | None = None, backend: str = "auto"):
        cfg = cfg or Config()
        acfg = cfg.get("input_layer")["audio"]
        self.whisper_model = acfg.get("whisper_model", "tiny")
        self.backend = backend if backend != "auto" else self._pick_backend()
        self._model = None
        if self.backend == "whisper":
            self._load_whisper()

    def _pick_backend(self) -> str:
        try:
            import whisper  # noqa: F401
            return "whisper"
        except Exception:
            return "rule"

    def _load_whisper(self):
        import whisper
        self._model = whisper.load_model(self.whisper_model)

    # ------------------------------------------------------------------
    def process(self, audio: Optional[np.ndarray] = None, text: Optional[str] = None) -> dict[str, Any]:
        """处理一段音频（16kHz）。text 可直接注入（测试/仿真捷径）。"""
        if self.backend == "whisper" and audio is not None and self._model is not None:
            import numpy as np
            audio = np.asarray(audio, dtype=np.float32)
            res = self._model.transcribe(audio, fp16=False)
            text = res.get("text", "")
            conf = float(res.get("no_speech_prob", 0.0))
        elif text is None:
            text = ""

        emotion = self._classify_emotion(text)
        intent = self._classify_intent(text)
        return {
            "text": text,
            "emotion": emotion,
            "intent": intent,
            "confidence": 1.0 if self.backend == "rule" else 0.9,
        }

    def _classify_emotion(self, text: str) -> str:
        low = text.lower()
        for emo, kws in EMOTION_KEYWORDS.items():
            if any(k in low for k in kws):
                return emo
        return "neutral"

    def _classify_intent(self, text: str) -> str:
        low = text.lower()
        for intent, kws in INTENT_KEYWORDS.items():
            if any(k in low for k in kws):
                return intent
        return "none"
