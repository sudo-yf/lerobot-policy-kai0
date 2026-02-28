from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("kai0")
@dataclass
class Kai0Config(PreTrainedConfig):
    model_type: str = "pi0"
    vision_backbone: str = "paligemma_2b"
    action_horizon: int = 50
    action_dim: int = 12
    pretrained_path: str | None = None

    input_features: dict = field(
        default_factory=lambda: {
            "observation.images.top_rgb": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.state": {"type": "STATE", "shape": [12]},
        }
    )

    output_features: dict = field(
        default_factory=lambda: {
            "action": {"type": "ACTION", "shape": [12]},
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be > 0")

    def to_openpi_config(self):
        try:
            from openpi.models_pytorch.pi0_pytorch import PI0Config
        except ImportError:
            from openpi.models_pytorch.pi0_pytorch import Pi0Config as PI0Config

        return PI0Config(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
        )
