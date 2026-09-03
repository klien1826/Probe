"""输入层 · 视觉编码器。

输入: RGB-D 图像 (224x224x3) + 深度 (224x224)
输出:
  - scene_vector: 256 维场景特征向量
  - object_proposals: 由深度/颜色连通域得到的候选物体列表

双后端:
  - backend="torch": 若环境装有 torch + torchvision，使用 ResNet18 骨干
  - backend="numpy": 确定性手工感知特征（色彩直方图 + 空间金字塔 + 梯度
      方向直方图 + 深度统计 + 纹理）→ 固定随机投影到 256 维
  - backend="auto": 优先 torch，否则 numpy
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.config import Config
from ..core.types import ObjectPercept


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / n if n > 1e-9 else x


class VisionEncoder:
    FEATURE_DIM = 256

    def __init__(self, cfg: Config | None = None, backend: str = "auto"):
        cfg = cfg or Config()
        vcfg = cfg.get("input_layer")["vision"]
        self.backend = backend if backend != "auto" else self._pick_backend()
        self.size = tuple(vcfg.get("input_size", [224, 224]))
        self.feature_dim = vcfg.get("feature_dim", 256)
        # 固定随机投影：180 维感知特征 → 256 维（确定性）
        rng = np.random.default_rng(20260903)
        self._projection = rng.standard_normal((180, self.feature_dim)).astype(np.float32)
        if self.backend == "torch":
            self._init_torch()

    def _pick_backend(self) -> str:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            return "torch"
        except Exception:
            return "numpy"

    def _init_torch(self):
        import torch
        import torchvision.models as models
        self._torch = torch
        self._resnet = models.resnet18(weights=None)
        # 去掉最后的分类头，取 pool 前特征
        self._resnet.fc = torch.nn.Identity()
        self._resnet.eval()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def encode_scene(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, list[ObjectPercept]]:
        """返回 (scene_vector(256,), object_proposals)。"""
        rgb = self._to_rgb(rgb)
        if depth is None:
            depth = np.full(rgb.shape[:2], 1.0, dtype=np.float32)
        if self.backend == "torch":
            vec = self._torch_embed(rgb)
        else:
            vec = self._numpy_embed(rgb, depth)
        proposals = self._segment_objects(rgb, depth)
        return vec.astype(np.float32), proposals

    # ------------------------------------------------------------------
    # numpy 感知特征后端
    # ------------------------------------------------------------------
    def _numpy_embed(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        rgb = cv2_resize(rgb, self.size)
        depth = cv2_resize(depth, self.size)
        rgb_f = rgb.astype(np.float32) / 255.0
        gray = rgb_f.mean(axis=2)
        h, w = self.size

        feats: list[np.ndarray] = []

        # 1) 全局色彩直方图：RGB 各 8 bin → 24
        for c in range(3):
            feats.append(np.histogram(rgb_f[:, :, c], bins=8, range=(0, 1))[0] / (h * w))

        # 2) 空间金字塔色彩均值：4x4 网格 × 3 通道 → 48
        grid = 4
        for gy in range(grid):
            for gx in range(grid):
                cell = rgb_f[gy * h // grid:(gy + 1) * h // grid,
                             gx * w // grid:(gx + 1) * w // grid]
                feats.append(cell.reshape(-1, 3).mean(axis=0))   # 3

        # 3) 梯度方向直方图：16 bin（幅值加权）→ 16
        gyv, gxv = np.gradient(gray)
        mag = np.hypot(gxv, gyv)
        ang = (np.arctan2(gyv, gxv) + np.pi) / (2 * np.pi)        # 0~1
        ghist = np.histogram(ang, bins=16, weights=mag, range=(0, 1))[0]
        ghist = ghist / (ghist.sum() + 1e-9)
        feats.append(ghist)

        # 4) 边缘密度：4x4 网格 → 16
        edge = (mag > mag.mean() + mag.std()).astype(np.float32)
        for gy in range(grid):
            for gx in range(grid):
                cell = edge[gy * h // grid:(gy + 1) * h // grid,
                            gx * w // grid:(gx + 1) * w // grid]
                feats.append(np.array([cell.mean()]))

        # 5) 深度统计：4x4 网格 mean/std/min/max → 64
        dnorm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
        for gy in range(grid):
            for gx in range(grid):
                cell = dnorm[gy * h // grid:(gy + 1) * h // grid,
                             gx * w // grid:(gx + 1) * w // grid]
                feats.append(np.array([cell.mean(), cell.std(), cell.min(), cell.max()]))

        # 6) 纹理：灰度共生统计简化版（邻域差分符号直方图 8 bin）→ 8
        diff = np.diff(gray, axis=1)[:, :-1].reshape(-1)
        tex = np.histogram(diff, bins=8, range=(-0.2, 0.2))[0]
        feats.append(tex / (tex.sum() + 1e-9))

        # 7) 全局统计：亮度/对比度/饱和度/深度范围 → 4
        sat = rgb_f.max(axis=2) - rgb_f.min(axis=2)
        feats.append(np.array([
            gray.mean(), gray.std(),
            sat.mean(), float(depth.max() - depth.min()),
        ]))

        raw = np.concatenate([f.flatten() for f in feats])
        assert raw.shape[0] == 180, raw.shape
        return _l2(self._projection.T @ raw)

    def _torch_embed(self, rgb: np.ndarray) -> np.ndarray:
        import numpy as np
        t = self._torch
        rgb = cv2_resize(rgb, (224, 224))
        x = t.from_numpy(rgb.transpose(2, 0, 1)[None]).float() / 255.0
        x = t.nn.functional.interpolate(x, size=(224, 224), mode="bilinear")
        with t.no_grad():
            emb = self._resnet(x).squeeze(0).numpy()
        return _l2(emb.astype(np.float32))

    # ------------------------------------------------------------------
    # 深度/颜色连通域 → 候选物体
    # ------------------------------------------------------------------
    def _segment_objects(self, rgb: np.ndarray, depth: np.ndarray) -> list[ObjectPercept]:
        import cv2
        h, w = rgb.shape[:2]
        depth_res = cv2_resize(depth, (w, h))
        # 深度近平面（有效物体）二值化
        valid = (depth_res > 0.1).astype(np.uint8)
        valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(valid, connectivity=8)
        props: list[ObjectPercept] = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < h * w * 0.01:          # 忽略过小区域
                continue
            mask = (labels == i)
            color = rgb[mask].mean(axis=0) / 255.0
            region_depth = depth_res[mask].mean()
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            size = float(np.sqrt(area)) / 100.0
            props.append(ObjectPercept(
                object_id=f"obj_{i}",
                category=None,
                color=tuple(color),
                size=size,
                state="static",
                touchable=True,
                danger_level=0.0,
                feature=np.zeros(16, dtype=np.float32),
                position=(float(cx) / w, float(cy) / h),
            ))
        return props

    def _to_rgb(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return np.stack([img] * 3, axis=-1)
        if img.shape[-1] == 4:
            return img[..., :3]
        return img[..., :3]


def cv2_resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2
    if img.shape[:2] != tuple(size):
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img
