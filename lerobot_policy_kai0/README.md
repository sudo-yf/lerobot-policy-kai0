# lerobot_policy_kai0

这是一个用于 LeRobot/LeHome 的 BYOP（Bring Your Own Policy）最小插件示例。

目标不是复现完整论文效果，而是帮你彻底掌握：
- 如何注册一个新策略类型
- 如何让 `lerobot-train` 识别并调用它
- 如何从 YAML 驱动训练

## 1. 安装

在本目录执行：

```bash
pip install -e .
```

## 2. 使用

在训练配置里设置：

```yaml
policy:
  type: kai0
```

然后从项目根目录运行：

```bash
lerobot-train --config_path=configs/train_kai0.yaml
```

## 3. 学习建议

请配合阅读项目根目录文档：

- `docs/从零构建新策略_实战教程.md`
- `docs/完全按Kai0训练逻辑运行.md`

该文档包含完整的学习路径、排错流程和练习任务。

## 4. 复制粘贴封装证明（Kai0 对齐）

为满足“尽量复制粘贴、尽量不改原代码”的要求，本插件包含从 `kai0` 原样复制的关键源码：

- `src/lerobot_policy_kai0/copied_from_kai0/agilex_policy.py`
- `src/lerobot_policy_kai0/copied_from_kai0/arx_policy.py`
- `src/lerobot_policy_kai0/copied_from_kai0/pi0_config.py`
- `src/lerobot_policy_kai0/copied_from_kai0/pi0_pytorch.py`

运行下面命令可验证复制文件与源文件 SHA256 完全一致：

```bash
bash lerobot_policy_kai0/scripts/verify_copied_sources.sh
```

详细说明见：

- `docs/复制粘贴封装证明.md`

## 5. 完全按 Kai0 训练逻辑（桥接脚本）

如果你要保留 Kai0 原生训练逻辑（data flow、tokenization、loss、checkpoint），请直接使用桥接脚本调用 `kai0/scripts/*.py`：

```bash
# JAX
bash lerobot_policy_kai0/scripts/train_openpi_exact.sh pi05_flatten_fold_normal run1 jax

# PyTorch
bash lerobot_policy_kai0/scripts/train_openpi_exact.sh ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD adv_run pytorch

# Serve
bash lerobot_policy_kai0/scripts/serve_openpi_exact.sh pi05_flatten_fold_normal /path/to/checkpoint 8000
```

可以用下面命令证明桥接目标确实指向 `kai0/` 原生入口：

```bash
bash lerobot_policy_kai0/scripts/prove_openpi_bridge.sh
```

## 6. 配置部分（configuration_kai0.py）

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
