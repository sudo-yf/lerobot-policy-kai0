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
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
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
        return
