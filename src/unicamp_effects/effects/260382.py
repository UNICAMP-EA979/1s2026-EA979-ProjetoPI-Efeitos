import numpy as np

from unicamp_effects.registry import register
from .imports.dithering_pallet import dithering_pallet
from .imports.dg import difference_of_gaussians

@register(prefix="260382")
def effect_identity(img: np.ndarray) -> np.ndarray:
    return img

@register(prefix="260382")
def delete_later(img: np.ndarray) -> np.ndarray:
    return 255 - img

@register(prefix="260382")
def dithering_black_white(img: np.ndarray) -> np.ndarray:
    return dithering_pallet(img, [(255,255,255), (0,0,0)])

@register(prefix="260382")
def dithering_red_black_white(img: np.ndarray) -> np.ndarray:
    return dithering_pallet(img, [(255,255,255), (0,0,0), (255,0,0)])

@register(prefix="260382")
def dithering_red_yellow_black_white(img: np.ndarray) -> np.ndarray:
    return dithering_pallet(img, [(255,255,255), (0,0,0), (255,0,0), (255,230,0)])

@register(prefix="260382")
def edge_detection(img: np.ndarray) -> np.ndarray:
    return difference_of_gaussians(img, 1.2, 0.8, 3.2, 1.2, 1000, 0.8, 0.95, )

@register(prefix="260382")
def difference_of_gaussians_flow(img: np.ndarray) -> np.ndarray:
    return difference_of_gaussians(img, 5.84, 0.8, 3.2, 0.75, 120, 1.83, 0.95, )

@register(prefix="260382")
def difference_of_gaussians_flow_less(img: np.ndarray) -> np.ndarray:
    return difference_of_gaussians(img, 1.0, 0.8, 3.2, 0.75, 120, 1.83, 0.9, )


@register(prefix="260382")
def diference_of_gaussians_flow_color(img: np.ndarray) -> np.ndarray:




