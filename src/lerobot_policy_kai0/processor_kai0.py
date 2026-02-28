from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .configuration_kai0 import Kai0Config
from openpi_client.image_tools import resize_with_pad

try:
    from openpi_client.action_chunk_broker import ActionChunkBroker as _OpenPIActionChunkBroker
except Exception:
    _OpenPIActionChunkBroker = None

try:
    from openpi.shared import image_tools as _openpi_image_tools
except Exception:
    _openpi_image_tools = None


class _SerializableCallableProcessor:
    """Callable wrapper that can be saved by LeRobot checkpoints."""

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
        meta = data.get("meta", {})
        fn = _build_processor_fn(name=name, meta=meta)
        return cls(fn=fn, name=name, meta=meta)


class _LocalChunkBroker:
    """Compatibility broker for action-chunk tensors/arrays."""

    def __init__(self, action_horizon: int):
        self.action_horizon = int(action_horizon)
        self._cur_step = 0
        self._last_chunk = None

    def add_and_get_action(self, action_chunk):
        if self._last_chunk is None:
            self._last_chunk = action_chunk
            self._cur_step = 0

        out = self._last_chunk[self._cur_step]
        self._cur_step += 1

        if self._cur_step >= self.action_horizon:
            self._last_chunk = None
            self._cur_step = 0

        return out


def _build_chunk_broker(config: Kai0Config):
    if _OpenPIActionChunkBroker is not None:
        try:
            sig = inspect.signature(_OpenPIActionChunkBroker.__init__)
            params = sig.parameters
            if "policy" not in params:
                broker = _OpenPIActionChunkBroker(action_horizon=config.action_horizon)
                if hasattr(broker, "add_and_get_action"):
                    return broker
        except Exception:
            pass
    return _LocalChunkBroker(action_horizon=config.action_horizon)


def _resize_image_like_kai0(img, target_h=224, target_w=224):
    """Accept torch/numpy, CHW or HWC, preserve input type/layout with minimal distortion."""
    is_torch = torch.is_tensor(img)
    was_chw = False

    if is_torch:
        t = img
        if t.ndim >= 3 and t.shape[-1] not in (1, 3, 4) and t.shape[-3] in (1, 3, 4):
            t = t.movedim(-3, -1)
            was_chw = True

        # Prefer torch-native resize path to avoid float->uint8 quantization.
        if _openpi_image_tools is not None and hasattr(_openpi_image_tools, "resize_with_pad_torch"):
            t_resized = _openpi_image_tools.resize_with_pad_torch(t, target_h, target_w)
        else:
            arr = t.detach().cpu().numpy()
            arr = _resize_numpy_fallback(arr, target_h, target_w)
            t_resized = torch.from_numpy(arr).to(device=t.device, dtype=t.dtype)

        if was_chw:
            t_resized = t_resized.movedim(-1, -3)
        return t_resized

    arr = np.asarray(img)
    if arr.ndim >= 3 and arr.shape[-1] not in (1, 3, 4) and arr.shape[-3] in (1, 3, 4):
        arr = np.moveaxis(arr, -3, -1)
        was_chw = True

    arr = _resize_numpy_fallback(arr, target_h, target_w)

    if was_chw:
        arr = np.moveaxis(arr, -1, -3)
    return arr


def _resize_numpy_fallback(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Fallback PIL-based resize, with best-effort dtype/domain preservation."""
    orig_dtype = arr.dtype
    scale01 = False

    if np.issubdtype(arr.dtype, np.floating):
        finite = np.isfinite(arr)
        vmax = float(arr[finite].max()) if finite.any() else 1.0
        vmin = float(arr[finite].min()) if finite.any() else 0.0
        if vmin >= 0.0 and vmax <= 1.5:
            scale01 = True
            arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr_u8 = arr.astype(np.uint8, copy=False)

    out = resize_with_pad(arr_u8, target_h, target_w)

    if np.issubdtype(orig_dtype, np.floating):
        out = out.astype(orig_dtype)
        if scale01:
            out = out / 255.0
    return out


def make_kai0_pre_post_processors(config: Kai0Config, dataset_stats=None):
    _ = dataset_stats

    view_to_openpi = {
        "observation.images.top_rgb": "base_0_rgb",
        "observation.images.left_rgb": "left_wrist_0_rgb",
        "observation.images.right_rgb": "right_wrist_0_rgb",
    }

    def pre_process(batch):
        # 三视角都做同样 resize，避免单视角伪装
        for key in view_to_openpi:
            if key not in batch:
                raise KeyError(f"Missing required image key: {key}")
            batch[key] = _resize_image_like_kai0(batch[key], 224, 224)
        return batch

    broker = _build_chunk_broker(config)

    def post_process(action_chunk):
        if hasattr(broker, "add_and_get_action"):
            return broker.add_and_get_action(action_chunk)
        return action_chunk

    pre = _SerializableCallableProcessor(
        pre_process,
        name="policy_preprocessor",
        meta={"type": "kai0", "resize": [224, 224], "required_views": list(view_to_openpi.keys())},
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
        h = int(resize[0])
        w = int(resize[1])
        required_views = meta.get(
            "required_views",
            ["observation.images.top_rgb", "observation.images.left_rgb", "observation.images.right_rgb"],
        )

        def _pre(batch):
            for key in required_views:
                if key not in batch:
                    raise KeyError(f"Missing required image key: {key}")
                batch[key] = _resize_image_like_kai0(batch[key], h, w)
            return batch

        return _pre

    if name == "policy_postprocessor":
        horizon = int(meta.get("action_horizon", 50))
        broker = _LocalChunkBroker(action_horizon=horizon)

        def _post(action_chunk):
            return broker.add_and_get_action(action_chunk)

        return _post

    raise ValueError(f"Unknown processor name: {name}")


def load_kai0_pre_post_processors(save_directory: str | Path):
    pre = _SerializableCallableProcessor.from_pretrained(save_directory, "policy_preprocessor")
    post = _SerializableCallableProcessor.from_pretrained(save_directory, "policy_postprocessor")
    return pre, post
