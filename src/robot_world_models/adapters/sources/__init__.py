"""Source adapters live here. See prompts/add-dataset.md before adding one."""
from robot_world_models.adapters.sources.github import GitHubSparseCheckoutSource
from robot_world_models.adapters.sources.huggingface import HuggingFaceDatasetSource

__all__ = ["GitHubSparseCheckoutSource", "HuggingFaceDatasetSource"]
