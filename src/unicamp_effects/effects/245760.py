import numpy as np

from unicamp_effects.registry import register


@register(prefix="245760")
def effect_name(img: np.ndarray) -> np.ndarray:
    ...
