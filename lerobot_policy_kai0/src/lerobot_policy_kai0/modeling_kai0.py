import torch
from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_kai0 import Kai0Config

try:
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
except ImportError as exc:
    raise ImportError(
        "请先在 Kai0 仓库执行 pip install -e . 以安装 openpi 核心库"
    ) from exc


class Kai0Policy(PreTrainedPolicy):
    config_class = Kai0Config
    name = "kai0"

    def __init__(self, config: Kai0Config, dataset_stats: dict | None = None, **kwargs):
        super().__init__(config, dataset_stats=dataset_stats, **kwargs)
        self.model = PI0Pytorch(config.to_openpi_config())

        if config.pretrained_path:
            state_dict = torch.load(config.pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

    def forward(self, batch: dict[str, torch.Tensor]):
        return self.model(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        return self.model.generate(batch)
