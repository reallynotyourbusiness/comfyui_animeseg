from .simple_anime_seg import AnimeSeg
from .advanced_anime_seg import AdvancedAnimeSeg

VERSION = "1.2.0"

NODE_CLASS_MAPPINGS = {
    "SimpleAnimeSeg": AnimeSeg,
    "AdvancedAnimeSeg": AdvancedAnimeSeg,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleAnimeSeg": AnimeSeg.TITLE,
    "AdvancedAnimeSeg": AdvancedAnimeSeg.TITLE,
}

__all__ = ["VERSION", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
