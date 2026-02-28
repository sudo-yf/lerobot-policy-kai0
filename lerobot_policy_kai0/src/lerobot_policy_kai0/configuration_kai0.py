from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig

@PreTrainedConfig.register_subclass("kai0")
@dataclass
class Kai0Config(PreTrainedConfig):
    # 模型架构
    model_type: str = "pi0"
    vision_backbone: str = "paligemma_2b"
    action_horizon: int = 50
    
    # 比赛特定的动作空间 (LeHome 推荐双臂 12 维)
    action_dim: int = 12
    
    # 输入特征配置
    input_features: dict = field(default_factory=lambda: {
        "observation.images.top_rgb": {"type": "VISUAL", "shape": [3, 480, 640]},
        "observation.state": {"type": "STATE", "shape": [12]},
    })
    
    output_features: dict = field(default_factory=lambda: {
        "action": {"type": "ACTION", "shape": [12]}
    })

    def to_openpi_config(self):
        # 这里的转换逻辑必须与 Kai0 的 pi0_pytorch.py 预期一致
        from openpi.models_pytorch.pi0_pytorch import PI0Config
        return PI0Config(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            # 其他 Kai0 原生参数...
        )
