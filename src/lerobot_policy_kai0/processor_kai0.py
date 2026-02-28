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
    """Accept torch/numpy, CHW or HWC, preserve input type/layout."""
    is_torch = torch.is_tensor(img)
    was_chw = False

    if is_torch:
        device = img.device
        dtype = img.dtype
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)

    # Detect CHW family and convert to HWC for openpi_client
    if arr.ndim >= 3 and arr.shape[-1] not in (1, 3, 4) and arr.shape[-3] in (1, 3, 4):
        arr = np.moveaxis(arr, -3, -1)
        was_chw = True

    # PIL backend needs uint8/compatible dtypes. Preserve original numeric domain.
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

    arr = resize_with_pad(arr_u8, target_h, target_w)

    if np.issubdtype(orig_dtype, np.floating):
        arr = arr.astype(orig_dtype)
        if scale01:
            arr = arr / 255.0

    if was_chw:
        arr = np.moveaxis(arr, -1, -3)

    if is_torch:
        return torch.from_numpy(arr).to(device=device, dtype=dtype)
    return arr


def make_kai0_pre_post_processors(config: Kai0Config, dataset_stats=None):
    _ = dataset_stats

    def pre_process(batch):
        img = batch["observation.images.top_rgb"]
        batch["observation.images.top_rgb"] = _resize_image_like_kai0(img, 224, 224)
        return batch

    broker = _build_chunk_broker(config)

    def post_process(action_chunk):
        if hasattr(broker, "add_and_get_action"):
            return broker.add_and_get_action(action_chunk)
        return action_chunk

    pre = _SerializableCallableProcessor(
        pre_process,
        name="policy_preprocessor",
        meta={"type": "kai0", "resize": [224, 224]},
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

        def _pre(batch):
            img = batch["observation.images.top_rgb"]
            batch["observation.images.top_rgb"] = _resize_image_like_kai0(img, h, w)
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
