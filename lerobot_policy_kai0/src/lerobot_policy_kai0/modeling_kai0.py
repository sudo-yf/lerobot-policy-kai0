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

    def __init__(self, config: Kai0Config, dataset_stats: dict = None):
        super().__init__(config, dataset_stats)
        # 100% 复原：使用 Kai0 原生初始化
        self.model = PI0Pytorch(config.to_openpi_config())
        
        # 加载预训练权重 (适配 LeHome 比赛权重)
        if hasattr(config, "pretrained_path") and config.pretrained_path:
            state_dict = torch.load(config.pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """训练模式下的前向传播"""
        # 将 LeRobot Batch 转换为 Kai0 推理格式
        return self.model(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """推理模式：LeHome 比赛实机运行核心"""
        self.eval()
        # 调用 Kai0 的 generate 逻辑，包含其内部的 Action Chunk 处理
        actions = self.model.generate(batch) 
        return actions
