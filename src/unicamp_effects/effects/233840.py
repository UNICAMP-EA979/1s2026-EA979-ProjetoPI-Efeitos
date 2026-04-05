import numpy as np

from unicamp_effects.registry import register


@register(prefix="233840")
def dummy1(img: np.ndarray) -> np.ndarray:
    return img


@register(prefix="233840")
def dummy2(img: np.ndarray) -> np.ndarray:
    return img


@register(prefix="233840")
def dummy3(img: np.ndarray) -> np.ndarray:
    return img
