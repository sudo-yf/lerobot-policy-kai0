from __future__ import annotations

import inspect

import numpy as np
import torch

from .configuration_kai0 import Kai0Config
from openpi_client.image_tools import resize_with_pad

try:
    from openpi_client.action_chunk_broker import ActionChunkBroker as _OpenPIActionChunkBroker
except Exception:
    _OpenPIActionChunkBroker = None


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

    return pre_process, post_process
