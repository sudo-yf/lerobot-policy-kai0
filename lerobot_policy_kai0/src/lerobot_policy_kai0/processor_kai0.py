import torch
from lerobot.policies.utils import PolicyProcessorPipeline
from .configuration_kai0 import Kai0Config
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
