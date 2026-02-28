from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_kai0 import Kai0Config

try:
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
except ImportError as exc:
    raise ImportError("请先在 Kai0 仓库执行 pip install -e . 以安装 openpi 核心库") from exc

logger = logging.getLogger(__name__)


class ProjectionAdapter(nn.Module):
    def __init__(
        self,
        state_in_dim: int,
        action_in_dim: int,
        model_state_dim: int,
        model_action_dim: int,
        recon_loss_weight: float,
    ):
        super().__init__()
        self.state_in_dim = state_in_dim
        self.action_in_dim = action_in_dim
        self.model_state_dim = model_state_dim
        self.model_action_dim = model_action_dim
        self.recon_loss_weight = recon_loss_weight

        self.state_proj = nn.Linear(state_in_dim, model_state_dim, bias=False)
        self.action_in_proj = nn.Linear(action_in_dim, model_action_dim, bias=False)
        self.action_out_proj = nn.Linear(model_action_dim, action_in_dim, bias=False)
        self._init_identity(self.state_proj)
        self._init_identity(self.action_in_proj)
        self._init_identity(self.action_out_proj)

    @staticmethod
    def _init_identity(layer: nn.Linear) -> None:
        with torch.no_grad():
            layer.weight.zero_()
            diag = min(layer.in_features, layer.out_features)
            for i in range(diag):
                layer.weight[i, i] = 1.0

    def project_state(self, state: torch.Tensor) -> torch.Tensor:
        self._check_last_dim(state, self.state_in_dim, "state")
        out = self.state_proj(state)
        logger.debug("[DEBUG] state projection: %s -> %s", tuple(state.shape), tuple(out.shape))
        return out

    def project_actions(self, actions: torch.Tensor) -> torch.Tensor:
        self._check_last_dim(actions, self.action_in_dim, "action")
        out = self.action_in_proj(actions)
        logger.debug("[DEBUG] action projection: %s -> %s", tuple(actions.shape), tuple(out.shape))
        return out

    def recover_actions(self, model_actions: torch.Tensor) -> torch.Tensor:
        self._check_last_dim(model_actions, self.model_action_dim, "model_action")
        out = self.action_out_proj(model_actions)
        logger.debug("[DEBUG] action recovery: %s -> %s", tuple(model_actions.shape), tuple(out.shape))
        return out

    def compute_recon_loss(self, projected_actions: torch.Tensor, raw_actions: torch.Tensor) -> torch.Tensor:
        recon = self.recover_actions(projected_actions)
        recon = recon[:, : raw_actions.shape[1], :]
        return self.recon_loss_weight * F.mse_loss(recon, raw_actions)

    @staticmethod
    def _check_last_dim(tensor: torch.Tensor, expected: int, name: str) -> None:
        if tensor.shape[-1] != expected:
            raise ValueError(f"{name} dim={tensor.shape[-1]} != expected {expected}")


class OpenPIBatchAdapter:
    def __init__(self, config: Kai0Config, projector: ProjectionAdapter):
        self.config = config
        self.projector = projector
        self._tokenizer = None

    def to_observation(self, batch: dict[str, torch.Tensor], model_config) -> SimpleNamespace:
        state = self._get_state(batch)
        images = self._get_images(batch)
        masks = self._make_image_masks(images, state.device)
        tokens = self._get_tokens(batch, state, model_config)
        return SimpleNamespace(images=images, image_masks=masks, state=state, **tokens)

    def to_actions(self, batch: dict[str, torch.Tensor], action_horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        raw_actions = self._get_actions(batch)
        projected = self.projector.project_actions(raw_actions)
        aligned = self._align_horizon(projected, action_horizon)
        return aligned, raw_actions

    def _get_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state = batch.get("observation.state", batch.get("state"))
        if state is None:
            raise KeyError("Missing 'observation.state' in batch")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(torch.float32)
        logger.debug("[DEBUG] raw state tensor: shape=%s dtype=%s", tuple(state.shape), state.dtype)
        return self.projector.project_state(state)

    def _get_images(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        top = self._pick(batch, "observation.images.top_rgb", ["observation.images.top"])
        left = self._pick(batch, "observation.images.left_rgb", ["observation.images.wrist_left", "observation.images.left"])
        right = self._pick(batch, "observation.images.right_rgb", ["observation.images.wrist_right", "observation.images.right"])
        return {"base_0_rgb": top, "left_wrist_0_rgb": left, "right_wrist_0_rgb": right}

    def _make_image_masks(self, images: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
        bsz = next(iter(images.values())).shape[0]
        return {k: torch.ones(bsz, dtype=torch.bool, device=device) for k in images}

    def _get_tokens(self, batch: dict[str, torch.Tensor], state: torch.Tensor, model_config) -> dict[str, torch.Tensor]:
        tokenized = batch.get("tokenized_prompt")
        token_mask = batch.get("tokenized_prompt_mask")
        if tokenized is None or token_mask is None:
            tokenized, token_mask = self._tokenize_from_text(batch, state, model_config)
        else:
            tokenized = tokenized.to(device=state.device, dtype=torch.int32)
            token_mask = token_mask.to(device=state.device, dtype=torch.bool)
        token_ar = self._optional_mask(batch, "token_ar_mask", tokenized, torch.int32)
        token_loss = self._optional_mask(batch, "token_loss_mask", token_mask, torch.bool)
        return {
            "tokenized_prompt": tokenized,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": token_ar,
            "token_loss_mask": token_loss,
        }

    def _tokenize_from_text(self, batch: dict[str, torch.Tensor], state: torch.Tensor, model_config):
        prompt = batch.get("prompt", batch.get("task", batch.get("task_description")))
        if prompt is None:
            raise KeyError("Missing tokenized prompts and no prompt/task/task_description provided")

        if self._tokenizer is None:
            from openpi.models.tokenizer import PaligemmaTokenizer

            max_len = int(self.config.max_token_len or getattr(model_config, "max_token_len", 192) or 192)
            self._tokenizer = PaligemmaTokenizer(max_len=max_len)
            logger.debug("[DEBUG] init tokenizer max_token_len=%s", max_len)

        prompts = self._expand_prompt(prompt, state.shape[0])
        token_list, mask_list = [], []
        for idx, text in enumerate(prompts):
            token_np, mask_np = self._tokenizer.tokenize(text, state[idx].detach().cpu().numpy())
            token_list.append(torch.as_tensor(token_np, dtype=torch.int32, device=state.device))
            mask_list.append(torch.as_tensor(mask_np, dtype=torch.bool, device=state.device))
        tokenized = torch.stack(token_list, dim=0)
        token_mask = torch.stack(mask_list, dim=0)
        logger.debug("[DEBUG] tokenized_prompt shape=%s", tuple(tokenized.shape))
        return tokenized, token_mask

    @staticmethod
    def _expand_prompt(prompt_obj, batch_size: int) -> list[str]:
        if isinstance(prompt_obj, (list, tuple)):
            prompts = [str(x) for x in prompt_obj]
        else:
            prompts = [str(prompt_obj)]
        if len(prompts) < batch_size:
            prompts = (prompts + [prompts[-1]] * batch_size)[:batch_size]
        return prompts[:batch_size]

    @staticmethod
    def _optional_mask(batch: dict[str, torch.Tensor], key: str, reference: torch.Tensor, dtype: torch.dtype):
        value = batch.get(key)
        if value is None:
            return torch.zeros_like(reference, dtype=dtype, device=reference.device)
        return value.to(device=reference.device, dtype=dtype)

    def _get_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        actions = batch.get("action", batch.get("actions"))
        if actions is None:
            raise KeyError("Missing 'action' or 'actions' in batch")
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = actions.to(torch.float32)
        logger.debug("[DEBUG] raw action tensor: shape=%s dtype=%s", tuple(actions.shape), actions.dtype)
        return actions

    @staticmethod
    def _align_horizon(projected_actions: torch.Tensor, action_horizon: int) -> torch.Tensor:
        if projected_actions.shape[1] == action_horizon:
            return projected_actions
        if projected_actions.shape[1] > action_horizon:
            out = projected_actions[:, :action_horizon, :]
            logger.debug("[DEBUG] action chunk truncated: %s -> %s", tuple(projected_actions.shape), tuple(out.shape))
            return out
        pad = action_horizon - projected_actions.shape[1]
        tail = projected_actions[:, -1:, :].expand(projected_actions.shape[0], pad, projected_actions.shape[2])
        out = torch.cat([projected_actions, tail], dim=1)
        logger.debug("[DEBUG] action chunk padded: %s -> %s", tuple(projected_actions.shape), tuple(out.shape))
        return out

    @staticmethod
    def _pick(batch: dict[str, torch.Tensor], required_key: str, aliases: list[str]) -> torch.Tensor:
        for key in [required_key, *aliases]:
            value = batch.get(key)
            if value is None:
                continue
            if value.ndim == 3:
                value = value.unsqueeze(0)
            logger.debug("[DEBUG] image key '%s' resolved from '%s' shape=%s", required_key, key, tuple(value.shape))
            return value
        raise KeyError(f"Missing required camera input: {required_key} (aliases={aliases})")


class Kai0Policy(PreTrainedPolicy):
    config_class = Kai0Config
    name = "kai0"

    def __init__(self, config: Kai0Config, dataset_stats: dict = None, dataset_meta=None, **kwargs):
        super().__init__(config, dataset_stats=dataset_stats, dataset_meta=dataset_meta, **kwargs)
        self.model = PI0Pytorch(config.to_openpi_config())
        self.projector = ProjectionAdapter(
            state_in_dim=config.state_dim,
            action_in_dim=config.action_dim,
            model_state_dim=config.model_state_dim,
            model_action_dim=config.model_action_dim,
            recon_loss_weight=config.adapter_recon_loss_weight,
        )
        self.adapter = OpenPIBatchAdapter(config, self.projector)
        self._load_pretrained_if_needed(config)

    def _load_pretrained_if_needed(self, config: Kai0Config) -> None:
        path = getattr(config, "pretrained_path", None)
        if not path:
            return
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"pretrained_path is not a file: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

    def get_optim_params(self) -> dict:
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def reset(self):
        return

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        observation = self.adapter.to_observation(batch, self.model.config)
        projected_actions, raw_actions = self.adapter.to_actions(batch, int(self.model.config.action_horizon))
        loss_tensor = self.model(observation, projected_actions)
        if not torch.is_tensor(loss_tensor):
            raise TypeError(f"Kai0 forward 返回异常类型: {type(loss_tensor)}")
        main_loss = loss_tensor.mean()
        recon_loss = self.projector.compute_recon_loss(projected_actions, raw_actions)
        total_loss = main_loss + recon_loss
        return total_loss, {
            "mse_loss": float(main_loss.detach().cpu()),
            "adapter_recon_loss": float(recon_loss.detach().cpu()),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        _ = kwargs
        self.eval()
        observation = self.adapter.to_observation(batch, self.model.config)
        device = next(self.model.parameters()).device
        if hasattr(self.model, "sample_actions"):
            return self.model.sample_actions(device, observation)
        return self.model.generate(observation)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        _ = kwargs
        action_chunk = self.predict_action_chunk(batch).to(torch.float32)
        return self.projector.recover_actions(action_chunk)
