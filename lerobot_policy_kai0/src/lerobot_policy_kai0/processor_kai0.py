from __future__ import annotations

from .configuration_kai0 import Kai0Config

try:
    from openpi_client.action_chunk_broker import ActionChunkBroker
    from openpi_client.image_tools import resize_with_pad
except ImportError as exc:
    raise ImportError(
        "缺少 openpi_client 依赖，请确认 Kai0 环境已正确安装。"
    ) from exc


def make_kai0_pre_post_processors(config: Kai0Config, dataset_stats=None):
    _ = dataset_stats

    def pre_process(batch):
        img = batch["observation.images.top_rgb"]
        batch["observation.images.top_rgb"] = resize_with_pad(img, 224, 224)
        return batch

    broker = ActionChunkBroker(action_horizon=config.action_horizon)

    def post_process(action_chunk):
        return broker.add_and_get_action(action_chunk)

    return pre_process, post_process
