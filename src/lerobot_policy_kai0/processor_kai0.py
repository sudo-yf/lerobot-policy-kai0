from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .configuration_kai0 import Kai0Config

logger = logging.getLogger(__name__)


class _SerializableCallableProcessor:
    def __init__(self, fn, name: str, meta: dict | None = None):
        self._fn = fn
        self._name = name
        self._meta = meta or {}

    def __call__(self, data):
        return self._fn(data)

    def save_pretrained(self, save_directory: str | Path):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{self._name}.json"
        out.write_text(
            json.dumps({"name": self._name, "meta": self._meta}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, save_directory: str | Path, name: str):
        save_dir = Path(save_directory)
        data = json.loads((save_dir / f"{name}.json").read_text(encoding="utf-8"))
        fn = _build_processor_fn(name=name, meta=data.get("meta", {}))
        return cls(fn=fn, name=name, meta=data.get("meta", {}))


class _TemporalEnsemblingChunkBroker:
    def __init__(self, action_horizon: int):
        self.action_horizon = int(action_horizon)
        self._chunks: list[tuple[torch.Tensor | np.ndarray, int]] = []

    def add_and_get_action(self, action_chunk):
        self._chunks.append((action_chunk, 0))
        candidates = self._collect_step_candidates()
        if not candidates:
            raise RuntimeError("No valid action step available in temporal broker")
        return self._average_candidates(candidates)

    def _collect_step_candidates(self):
        candidates, updated = [], []
        for chunk, step_index in self._chunks:
            step = self._step_from_chunk(chunk, step_index)
            if step is None:
                continue
            candidates.append(step)
            if step_index + 1 < self._chunk_horizon(chunk):
                updated.append((chunk, step_index + 1))
        self._chunks = updated
        logger.debug("[DEBUG] temporal broker active_chunks=%s candidates=%s", len(self._chunks), len(candidates))
        return candidates

    @staticmethod
    def _step_from_chunk(chunk, step_index: int):
        if torch.is_tensor(chunk):
            if step_index >= chunk.shape[-2]:
                return None
            return chunk.select(dim=-2, index=step_index)
        arr = np.asarray(chunk)
        if step_index >= arr.shape[-2]:
            return None
        return np.take(arr, step_index, axis=-2)

    @staticmethod
    def _chunk_horizon(chunk) -> int:
        return int(chunk.shape[-2] if torch.is_tensor(chunk) else np.asarray(chunk).shape[-2])

    @staticmethod
    def _average_candidates(candidates):
        if torch.is_tensor(candidates[0]):
            return torch.stack(candidates, dim=0).mean(dim=0)
        return np.stack(candidates, axis=0).mean(axis=0)


class _TorchImageResizer:
    def __init__(self, target_h: int, target_w: int):
        self.target_h = int(target_h)
        self.target_w = int(target_w)

    def resize_with_pad(self, img):
        if torch.is_tensor(img):
            return self._resize_tensor(img)
        return self._resize_numpy(img)

    def _resize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        canonical, was_chw = self._to_hwc_layout(tensor)
        resized = self._resize_hwc_tensor(canonical)
        return resized.movedim(-1, -3) if was_chw else resized

    def _resize_numpy(self, array_like) -> np.ndarray:
        arr = np.asarray(array_like)
        was_chw = arr.ndim >= 3 and arr.shape[-1] not in (1, 3, 4) and arr.shape[-3] in (1, 3, 4)
        if was_chw:
            arr = np.moveaxis(arr, -3, -1)
        out = self._resize_hwc_tensor(torch.from_numpy(arr)).cpu().numpy()
        return np.moveaxis(out, -1, -3) if was_chw else out

    @staticmethod
    def _to_hwc_layout(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
        was_chw = tensor.ndim >= 3 and tensor.shape[-1] not in (1, 3, 4) and tensor.shape[-3] in (1, 3, 4)
        return (tensor.movedim(-3, -1), True) if was_chw else (tensor, False)

    def _resize_hwc_tensor(self, image_hwc: torch.Tensor) -> torch.Tensor:
        if image_hwc.ndim < 3:
            raise ValueError(f"Expected image ndim>=3, got {image_hwc.ndim}")
        h, w = image_hwc.shape[-3], image_hwc.shape[-2]
        lead = image_hwc.shape[:-3]
        n = int(np.prod(lead)) if lead else 1
        orig_dtype = image_hwc.dtype

        nchw = image_hwc.reshape(n, h, w, image_hwc.shape[-1]).permute(0, 3, 1, 2).contiguous()
        nchw = self._to_interpolation_dtype(nchw)
        resized = self._interpolate_keep_ratio(nchw, h, w)
        padded = self._pad_to_target(resized)
        hwc = padded.permute(0, 2, 3, 1).reshape(*lead, self.target_h, self.target_w, image_hwc.shape[-1])
        out = self._to_original_dtype(hwc, orig_dtype)

        logger.debug(
            "[DEBUG] image resize_with_pad: in_shape=%s out_shape=%s dtype=%s",
            tuple(image_hwc.shape),
            tuple(out.shape),
            str(orig_dtype),
        )
        return out

    @staticmethod
    def _to_interpolation_dtype(nchw: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(nchw):
            return nchw.to(torch.float32)
        return nchw.to(torch.float32) / 255.0

    def _interpolate_keep_ratio(self, nchw: torch.Tensor, h: int, w: int) -> torch.Tensor:
        scale = min(self.target_h / float(h), self.target_w / float(w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        return F.interpolate(nchw, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)

    def _pad_to_target(self, resized: torch.Tensor) -> torch.Tensor:
        pad_h = self.target_h - resized.shape[-2]
        pad_w = self.target_w - resized.shape[-1]
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

    @staticmethod
    def _to_original_dtype(tensor: torch.Tensor, original_dtype: torch.dtype) -> torch.Tensor:
        if original_dtype == torch.uint8:
            return (tensor * 255.0).clamp(0, 255).round().to(torch.uint8)
        return tensor.to(original_dtype)


def make_kai0_pre_post_processors(config: Kai0Config, dataset_stats=None):
    _ = dataset_stats
    required_views = [
        "observation.images.top_rgb",
        "observation.images.left_rgb",
        "observation.images.right_rgb",
    ]
    resizer = _TorchImageResizer(config.image_resize_height, config.image_resize_width)
    broker = _TemporalEnsemblingChunkBroker(action_horizon=config.action_horizon)

    def pre_process(batch):
        for key in required_views:
            if key not in batch:
                raise KeyError(f"Missing required image key: {key}")
            batch[key] = resizer.resize_with_pad(batch[key])
        return batch

    def post_process(action_chunk):
        return broker.add_and_get_action(action_chunk)

    pre = _SerializableCallableProcessor(
        pre_process,
        name="policy_preprocessor",
        meta={
            "type": "kai0",
            "resize": [config.image_resize_height, config.image_resize_width],
            "required_views": required_views,
        },
    )
    post = _SerializableCallableProcessor(
        post_process,
        name="policy_postprocessor",
        meta={"type": "kai0", "action_horizon": int(config.action_horizon)},
    )
    return pre, post


def _build_processor_fn(name: str, meta: dict):
    if name == "policy_preprocessor":
        resize = meta.get("resize", [224, 224])
        required_views = meta.get(
            "required_views",
            ["observation.images.top_rgb", "observation.images.left_rgb", "observation.images.right_rgb"],
        )
        resizer = _TorchImageResizer(int(resize[0]), int(resize[1]))

        def _pre(batch):
            for key in required_views:
                if key not in batch:
                    raise KeyError(f"Missing required image key: {key}")
                batch[key] = resizer.resize_with_pad(batch[key])
            return batch

        return _pre

    if name == "policy_postprocessor":
        broker = _TemporalEnsemblingChunkBroker(action_horizon=int(meta.get("action_horizon", 50)))

        def _post(action_chunk):
            return broker.add_and_get_action(action_chunk)

        return _post

    raise ValueError(f"Unknown processor name: {name}")


def load_kai0_pre_post_processors(save_directory: str | Path):
    pre = _SerializableCallableProcessor.from_pretrained(save_directory, "policy_preprocessor")
    post = _SerializableCallableProcessor.from_pretrained(save_directory, "policy_postprocessor")
    return pre, post
