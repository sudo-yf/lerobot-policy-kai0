from __future__ import annotations

from pathlib import Path
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
        self._tokenizer = None

        if hasattr(config, "pretrained_path") and config.pretrained_path:
            ckpt_path = Path(config.pretrained_path)
            if ckpt_path.is_file():
                state_dict = torch.load(ckpt_path, map_location="cpu")
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
        state = batch.get("observation.state", batch.get("state"))
        if state is None:
            raise KeyError("Missing 'observation.state' in batch")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = self._pad_last_dim(state.to(torch.float32), 32)

        def _get_img(required_key: str, aliases: list[str]) -> torch.Tensor:
            for key in [required_key, *aliases]:
                value = batch.get(key)
                if value is not None:
                    if value.ndim == 3:
                        return value.unsqueeze(0)
                    return value
            raise KeyError(f"Missing required camera input: {required_key} (aliases={aliases})")

        # 三个视角严格对齐，不做 top 复用伪装
        top = _get_img("observation.images.top_rgb", ["observation.images.top"])
        left = _get_img("observation.images.left_rgb", ["observation.images.wrist_left", "observation.images.left"])
        right = _get_img("observation.images.right_rgb", ["observation.images.wrist_right", "observation.images.right"])

        images = {
            "base_0_rgb": top,
            "left_wrist_0_rgb": left,
            "right_wrist_0_rgb": right,
        }

        bsz = state.shape[0]
        device = state.device
        image_masks = {k: torch.ones(bsz, dtype=torch.bool, device=device) for k in images}

        tokenized_prompt = batch.get("tokenized_prompt")
        tokenized_prompt_mask = batch.get("tokenized_prompt_mask")
        token_ar_mask = batch.get("token_ar_mask")
        token_loss_mask = batch.get("token_loss_mask")

        if tokenized_prompt is None or tokenized_prompt_mask is None:
            prompt_obj = batch.get("prompt", batch.get("task", batch.get("task_description")))
            if prompt_obj is None:
                raise KeyError(
                    "Missing tokenized_prompt/tokenized_prompt_mask and no prompt/task/task_description provided."
                )

            if self._tokenizer is None:
                from openpi.models.tokenizer import PaligemmaTokenizer

                max_len = int(getattr(self.model.config, "max_token_len", 200) or 200)
                self._tokenizer = PaligemmaTokenizer(max_len=max_len)

            if isinstance(prompt_obj, (list, tuple)):
                prompts = [str(x) for x in prompt_obj]
            else:
                prompts = [str(prompt_obj)] * bsz
            if len(prompts) != bsz:
                prompts = (prompts + [prompts[-1]])[:bsz]

            token_list = []
            mask_list = []
            for i, text in enumerate(prompts):
                token_np, mask_np = self._tokenizer.tokenize(text, state[i].detach().cpu().numpy())
                token_list.append(torch.as_tensor(token_np, dtype=torch.int32, device=device))
                mask_list.append(torch.as_tensor(mask_np, dtype=torch.bool, device=device))
            tokenized_prompt = torch.stack(token_list, dim=0)
            tokenized_prompt_mask = torch.stack(mask_list, dim=0)
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

        if hasattr(self.model, "sample_actions"):
            return self.model.sample_actions(device, observation)
        return self.model.generate(observation)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        _ = kwargs
        action_chunk = self.predict_action_chunk(batch)
        return action_chunk[..., : self.config.action_dim]
