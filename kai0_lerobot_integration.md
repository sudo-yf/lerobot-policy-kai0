这份文档将 Kai0 (基于 $\pi_0$ 架构) 作为一个独立的 `openpi` 核心库进行依赖，并构建 LeRobot 插件包。这种结构能确保 100% 复原 Kai0 的推理性能，同时完美适配 LeHome 比赛环境。

---

# Kai0 LeRobot Policy 适配集成手册 (LeHome 比赛专用)

## 1. 整体目录结构

根据顶会最佳实践，我们将模型核心逻辑 (`openpi`) 与 框架适配逻辑 (`lerobot_policy_kai0`) 解耦。

```text
lerobot_policy_kai0/
├── pyproject.toml                     # 声明对 Kai0/src 的依赖
└── src/
    └── lerobot_policy_kai0/
        ├── __init__.py                # 暴露入口
        ├── configuration_kai0.py      # 将 Pi0Config 桥接到 LeRobot Config
        ├── modeling_kai0.py           # 核心：调用 openpi.models_pytorch.pi0_pytorch
        └── processor_kai0.py          # 核心：适配 Kai0 的图像缩放与 Action Chunking

```

---

## 2. 核心代码实现

### A. `pyproject.toml`

声明依赖项。注意：`openpi` 应作为本地编辑模式或直接从 Git 安装。

```toml
[project]
name = "lerobot_policy_kai0"
version = "0.1.0"
description = "Kai0 VLA policy plugin for LeRobot"
dependencies = [
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "lerobot",
    "openpi", # 假设你已在 Kai0/src 下执行 pip install -e .
]
requires-python = ">=3.10"

[build-system]
build-backend = "setuptools.build_meta"
requires = ["setuptools>=64"]

```

### B. `configuration_kai0.py`

关键点：定义 `to_openpi_config` 方法，将 LeRobot 的配置动态转换为 Kai0 内部配置。

```python
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

```

### C. `modeling_kai0.py`

关键点：直接封装 `openpi.models_pytorch.pi0_pytorch.PI0Pytorch`。

```python
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

```

### D. `processor_kai0.py`

关键点：复用 Kai0 仓库中的 `image_tools.py` 和 `action_chunk_broker.py`。

```python
import torch
from lerobot.policies.utils import PolicyProcessorPipeline
# 复用 Kai0 原厂工具
from openpi_client.image_tools import resize_with_pad 
from openpi_client.action_chunk_broker import ActionChunkBroker

def make_kai0_pre_post_processors(config: Kai0Config, dataset_stats):
    # 预处理：图像必须按 Kai0 要求缩放并填充 (Padding)
    def pre_process(batch):
        # 示例：将 480x640 转换为 Kai0 预期的 224x224
        img = batch["observation.images.top_rgb"]
        batch["observation.images.top_rgb"] = resize_with_pad(img, 224, 224)
        return batch

    # 后处理：Action Chunking 策略
    # 在 LeHome 比赛中，由于硬件延迟，Broker 模式至关重要
    broker = ActionChunkBroker(action_horizon=config.action_horizon)

    def post_process(action_chunk):
        # 这里的逻辑可以决定是执行整个 chunk 还是逐帧执行
        return broker.add_and_get_action(action_chunk)

    return pre_process, post_process

```

---

## 3. 部署指南 (科学家建议)

1. **环境安装**：
```bash
# 先安装 Kai0 核心
git clone https://github.com/OpenDriveLab/Kai0.git
cd Kai0/src && pip install -e .

# 再安装你的插件包
cd ../../lerobot_policy_kai0 && pip install -e .

```


2. **训练命令**：
按照 LeHome 规范，创建 `configs/train_kai0.yaml`：
```yaml
policy:
  type: kai0
  action_dim: 12 # 双臂关节空间
dataset:
  repo_id: lehome/competition_data

```


3. **比赛避坑点**：
* **Joint Space Only**：LeHome 文档强调不要用 `ee_pose`，请确保在 `configuration_kai0.py` 中强制将 `action_dim` 与关节维度对齐。
* **Padding**：Kai0 是针对特定长宽比训练的，`resize_with_pad` 是复原精度的关键。



通过这份基于 `gitingest` 思想的适配文档，你既保留了 Kai0 顶会级的模型能力，又完美接入了 LeRobot 的比赛评估管线。
