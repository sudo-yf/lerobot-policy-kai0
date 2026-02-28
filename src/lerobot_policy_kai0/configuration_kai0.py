from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("kai0")
@dataclass
class Kai0Config(PreTrainedConfig):
    model_type: str = "pi0"
    vision_backbone: str = "paligemma_2b"

    action_horizon: int = 50
    action_dim: int = 12
    state_dim: int = 12

    max_token_len: int = 192
    model_action_dim: int = 32
    model_state_dim: int = 32

    image_resize_height: int = 224
    image_resize_width: int = 224

    adapter_recon_loss_weight: float = 0.05

    input_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "observation.images.top_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.left_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.right_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(12,)),
        }
    )

    output_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(12,)),
        }
    )

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 2.5e-6

    def to_openpi_config(self):
        from openpi.models.pi0_config import Pi0Config

        return Pi0Config(
            action_dim=self.model_action_dim,
            action_horizon=self.action_horizon,
            max_token_len=self.max_token_len,
        )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self):
        return None

    def validate_features(self) -> None:
        required_visual = {
            "observation.images.top_rgb",
            "observation.images.left_rgb",
            "observation.images.right_rgb",
        }
        missing_visual = [k for k in required_visual if k not in self.input_features]
        if missing_visual:
            raise ValueError(f"Missing required visual features: {missing_visual}")

        if "observation.state" not in self.input_features:
            raise ValueError("Missing required state feature: observation.state")

        state_shape = tuple(self.input_features["observation.state"].shape)
        if len(state_shape) != 1 or state_shape[0] != self.state_dim:
            raise ValueError(
                f"State feature shape {state_shape} is inconsistent with state_dim={self.state_dim}"
            )

        action_shape = tuple(self.output_features["action"].shape)
        if len(action_shape) != 1 or action_shape[0] != self.action_dim:
            raise ValueError(
                f"Action feature shape {action_shape} is inconsistent with action_dim={self.action_dim}"
            )

        if self.image_resize_height <= 0 or self.image_resize_width <= 0:
            raise ValueError("image_resize_height and image_resize_width must be positive")

        if self.model_action_dim <= 0 or self.model_state_dim <= 0:
            raise ValueError("model_action_dim and model_state_dim must be positive")

        if self.max_token_len <= 0:
            raise ValueError("max_token_len must be positive")
