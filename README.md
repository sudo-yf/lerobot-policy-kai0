# lerobot-policy-kai0

这个仓库现在以“根目录即可用”为目标组织：你克隆后不需要再点进子目录，直接在仓库根目录就能看到并使用 Kai0 的 LeRobot 插件。

## 仓库根目录内容

- `pyproject.toml`：插件打包定义
- `src/lerobot_policy_kai0/`：插件源码
- `train_kai0.yaml`：可直接用的训练配置
- `README.md`：启动方法 + 全部映射与适配说明

## 一键启动（LeHome）

假设你的训练工程目录是 `lehome-challenge`：

```bash
cd /root/data/lehome-challenge
source .venv/bin/activate
pip install -e /root/data/lerobot-policy-kai0
lerobot-train --config_path=/root/data/lerobot-policy-kai0/train_kai0.yaml
```

建议把日志落盘：

```bash
mkdir -p logs
lerobot-train --config_path=/root/data/lerobot-policy-kai0/train_kai0.yaml 2>&1 | tee logs/train_kai0_$(date +%m-%d_%H-%M-%S).log
```

## 这套封装做了什么适配

### 1. `configuration_kai0.py` 适配

目标：对齐 LeRobot 抽象配置接口，并桥接 Kai0 配置对象。

- 注册策略类型：`@PreTrainedConfig.register_subclass("kai0")`
- `to_openpi_config()` 双路径兼容：
  - 优先 `openpi.models_pytorch.pi0_pytorch.PI0Config`
  - 回退 `openpi.models.pi0_config.Pi0Config`
- 补齐 LeRobot 新版要求的抽象项：
  - `get_optimizer_preset`
  - `get_scheduler_preset`
  - `observation_delta_indices`
  - `action_delta_indices`
  - `reward_delta_indices`
  - `validate_features`

### 2. `modeling_kai0.py` 适配

目标：把 LeRobot batch 转成 Kai0 `PI0Pytorch` 真实可吃的输入。

- 对齐 LeRobot policy 抽象方法：
  - `get_optim_params`
  - `predict_action_chunk`
  - `reset`
- `__init__` 兼容工厂注入参数：`dataset_meta` / `**kwargs`
- 训练路径适配：
  - LeRobot 传入 `batch`
  - 适配后调用 `self.model(observation, actions)`（而不是直接 `self.model(batch)`）
- 推理路径适配：
  - 优先 `sample_actions(device, observation)`
  - 若版本提供 `generate` 则自动回退兼容
- 维度映射：
  - 外部比赛动作 `12` 维 -> 内部 Kai0 计算通道 `32` 维（pad/crop）
  - `action_horizon` 自动对齐（不足复制末帧补齐，过长截断）
- 视觉键映射（单相机最小可跑策略）：
  - `observation.images.top_rgb` ->
    - `base_0_rgb`
    - `left_wrist_0_rgb`
    - `right_wrist_0_rgb`

### 3. `processor_kai0.py` 适配

目标：统一处理 openpi_client 版本差异和图像类型差异。

- `ActionChunkBroker` 版本兼容：
  - 老版本：`ActionChunkBroker(action_horizon=...)`
  - 新版本：`ActionChunkBroker(policy, action_horizon)`
  - 处理方法：检测签名，若不兼容则使用本地 `_LocalChunkBroker` 保持 `add_and_get_action` 行为
- 图像 resize 适配：
  - 支持 `torch.Tensor` 和 `numpy.ndarray`
  - 支持 `CHW/HWC`
  - float 图像先转 `uint8` 走 PIL resize，再转回原 dtype/域
  - 固定 resize 到 `224x224`（Kai0 期望分辨率）

## 已解决的典型崩溃点

- `ActionChunkBroker.__init__() missing required positional argument: 'policy'`
- `PI0Pytorch.forward() missing 1 required positional argument: 'actions'`
- `AttributeError: 'Tensor' object has no attribute '__array_interface__'`
- `TypeError: Cannot handle this data type: (1, 1, 3), <f4`
- `Can't instantiate abstract class Kai0Config ...`
- `Kai0Policy.__init__() got an unexpected keyword argument 'dataset_meta'`

## 训练配置说明（`train_kai0.yaml`）

- 数据默认指向本地：`Datasets/example/top_long_merged`
- `policy.type: kai0`
- `action_dim: 12`
- `action_horizon: 50`
- `wandb.enable: false`（你可以自行改成 `true`）

## 备注

这份插件是“适配层”实现：
- 保留 Kai0 模型主逻辑（`PI0Pytorch`）
- 在 LeRobot 与 openpi_client 版本漂移的边界做防护和映射
- 目标是让训练命令稳定跑通，而不是改写 Kai0 本体算法
