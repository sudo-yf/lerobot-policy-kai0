# lerobot-policy-kai0

Kai0 (`PI0Pytorch`) 的 LeRobot 插件，按 LeHome 环境做最小适配，保持 Kai0 推理/训练接口一致。

## 目录

- `pyproject.toml`
- `src/lerobot_policy_kai0/__init__.py`
- `src/lerobot_policy_kai0/configuration_kai0.py`
- `src/lerobot_policy_kai0/modeling_kai0.py`
- `src/lerobot_policy_kai0/processor_kai0.py`
- `train_kai0.yaml`

## 核心适配

### 1) 配置层 (`configuration_kai0.py`)

- 注册 `kai0`：`@PreTrainedConfig.register_subclass("kai0")`
- `to_openpi_config()` -> `openpi.models.pi0_config.Pi0Config`
- `input_features`/`output_features` 使用 LeRobot `PolicyFeature` 类型（不是裸 dict）
- 默认三路视觉 + 状态：
  - `observation.images.top_rgb`
  - `observation.images.left_rgb`
  - `observation.images.right_rgb`
  - `observation.state`

### 2) 模型层 (`modeling_kai0.py`)

- 直接封装 `PI0Pytorch`
- LeRobot batch -> Kai0 observation/actions 适配
- 三视角严格映射（不做 top 伪装）：
  - `observation.images.top_rgb` -> `base_0_rgb`
  - `observation.images.left_rgb` -> `left_wrist_0_rgb`
  - `observation.images.right_rgb` -> `right_wrist_0_rgb`
- 动作维度对齐：外部 `action_dim=12`，内部 pad/crop 到 32 通道供 Kai0 计算
- prompt 处理：优先用 batch 里的 token 字段；缺失时走 `PaligemmaTokenizer`

### 3) 处理器层 (`processor_kai0.py`)

- 预处理：三路图像都做 `resize_with_pad -> 224x224`
- 后处理：`ActionChunkBroker` 版本兼容（不兼容时回退本地 broker）
- 返回可序列化 processor（含 `save_pretrained/from_pretrained`），保证 checkpoint/save/load 与评估脚本兼容

## 训练（LeHome）

先安装插件（编辑模式）：

```bash
pip install -e /root/data/lerobot_policy_kai0
```

建议在 `lehome-challenge` 下运行，并把日志写到 `logs/`：

```bash
cd /root/data/lehome-challenge
source .venv/bin/activate
mkdir -p logs
lerobot-train --config_path=configs/train_kai0_run2.yaml \
  2>&1 | tee logs/train_kai0_$(date +%m-%d_%H-%M-%S).log
```

注意：`--config_path` 请用 `lehome-challenge` 内的相对路径（如 `configs/...`），不要直接传绝对路径给 `lerobot-train`，否则会被当成 HuggingFace repo id。

## 当前已验证

- 训练可正常进入 loop
- `save_pretrained` 路径可用
- smoke 跑通并成功落地 checkpoint（`000050`）

## 已知现象

- 日志中会出现 tokenizer truncation 警告（`max_token_len=48`），不阻塞训练；若你希望保留更长文本语义，可再调大该长度。
