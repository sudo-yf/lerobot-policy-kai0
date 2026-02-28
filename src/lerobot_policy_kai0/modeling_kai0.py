from __future__ import annotations

from types import SimpleNamespace

import torch
from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_kai0 import Kai0Config

# 动态引入 Kai0 核心
try:
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
except ImportError:
    raise ImportError("请先在 Kai0 仓库执行 pip install -e . 以安装 openpi 核心库")


class Kai0Policy(PreTrainedPolicy):
    config_class = Kai0Config
    name = "kai0"

    def __init__(
        self,
        config: Kai0Config,
        dataset_stats: dict = None,
        dataset_meta=None,
        **kwargs,
    ):
        super().__init__(config, dataset_stats=dataset_stats, dataset_meta=dataset_meta, **kwargs)
        self.model = PI0Pytorch(config.to_openpi_config())

        if hasattr(config, "pretrained_path") and config.pretrained_path:
            state_dict = torch.load(config.pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

    def get_optim_params(self) -> dict:
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def reset(self):
        return

    def _pad_last_dim(self, x: torch.Tensor, target_dim: int = 32) -> torch.Tensor:
        if x.shape[-1] == target_dim:
            return x
        if x.shape[-1] > target_dim:
            return x[..., :target_dim]
        pad = target_dim - x.shape[-1]
        return torch.nn.functional.pad(x, (0, pad))

    def _build_openpi_observation(self, batch: dict[str, torch.Tensor]) -> SimpleNamespace:
        # state
        state = batch.get("observation.state", batch.get("state"))
        if state is None:
            raise KeyError("Missing 'observation.state' in batch")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = self._pad_last_dim(state.to(torch.float32), 32)

        # images: map LeHome top camera to OpenPI camera keys
        top = batch.get("observation.images.top_rgb")
        if top is None:
            raise KeyError("Missing 'observation.images.top_rgb' in batch")
        if top.ndim == 3:
            top = top.unsqueeze(0)

        images = {
            "base_0_rgb": top,
            "left_wrist_0_rgb": top,
            "right_wrist_0_rgb": top,
        }

        bsz = state.shape[0]
        device = state.device
        image_masks = {k: torch.ones(bsz, dtype=torch.bool, device=device) for k in images}

        # tokenized prompt fallback
        tokenized_prompt = batch.get("tokenized_prompt")
        tokenized_prompt_mask = batch.get("tokenized_prompt_mask")
        token_ar_mask = batch.get("token_ar_mask")
        token_loss_mask = batch.get("token_loss_mask")

        if tokenized_prompt is None or tokenized_prompt_mask is None:
            tokenized_prompt = torch.zeros((bsz, 1), dtype=torch.int32, device=device)
            tokenized_prompt_mask = torch.ones((bsz, 1), dtype=torch.bool, device=device)
        else:
            tokenized_prompt = tokenized_prompt.to(device=device, dtype=torch.int32)
            tokenized_prompt_mask = tokenized_prompt_mask.to(device=device, dtype=torch.bool)

        if token_ar_mask is None:
            token_ar_mask = torch.zeros_like(tokenized_prompt, dtype=torch.int32, device=device)
        else:
            token_ar_mask = token_ar_mask.to(device=device, dtype=torch.int32)

        if token_loss_mask is None:
            token_loss_mask = torch.zeros_like(tokenized_prompt_mask, dtype=torch.bool, device=device)
        else:
            token_loss_mask = token_loss_mask.to(device=device, dtype=torch.bool)

        return SimpleNamespace(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
        )

    def _build_openpi_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        actions = batch.get("action", batch.get("actions"))
        if actions is None:
            raise KeyError("Missing 'action' or 'actions' in batch")

        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = actions.to(torch.float32)
        actions = self._pad_last_dim(actions, 32)

        # Ensure horizon matches model config.
        horizon = int(getattr(self.model.config, "action_horizon", actions.shape[1]))
        if actions.shape[1] < horizon:
            pad = horizon - actions.shape[1]
            tail = actions[:, -1:, :].expand(actions.shape[0], pad, actions.shape[2])
            actions = torch.cat([actions, tail], dim=1)
        elif actions.shape[1] > horizon:
            actions = actions[:, :horizon, :]

        return actions

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        observation = self._build_openpi_observation(batch)
        actions = self._build_openpi_actions(batch)
        loss_tensor = self.model(observation, actions)

        if not torch.is_tensor(loss_tensor):
            raise TypeError(f"Kai0 forward 返回异常类型: {type(loss_tensor)}")

        loss = loss_tensor.mean()
        return loss, {"mse_loss": float(loss.detach().cpu())}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        _ = kwargs
        self.eval()
        observation = self._build_openpi_observation(batch)
        device = next(self.model.parameters()).device
        return self.model.sample_actions(device, observation)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        _ = kwargs
        action_chunk = self.predict_action_chunk(batch)
        # Export competition action dim only.
        return action_chunk[..., : self.config.action_dim]
