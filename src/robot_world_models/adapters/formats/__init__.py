"""Format adapters live here. See prompts/add-dataset.md before adding one."""
from robot_world_models.adapters.formats.lerobot_v2 import LeRobotV2Adapter
from robot_world_models.adapters.formats.lerobot_v3_collection import (
    LeRobotV3CollectionAdapter,
)

__all__ = ["LeRobotV2Adapter", "LeRobotV3CollectionAdapter"]
