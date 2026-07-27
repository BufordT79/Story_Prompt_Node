"""Story → Krea 2 Prompt Batch: ComfyUI custom node package."""

from .story_prompt_node import StoryPromptBatchGenerator

NODE_CLASS_MAPPINGS = {
    "StoryPromptBatchGenerator": StoryPromptBatchGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StoryPromptBatchGenerator": "Story → Krea 2 Prompt Batch",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
