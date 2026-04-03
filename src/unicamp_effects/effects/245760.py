import numpy as np
from PIL import Image

from unicamp_effects.registry import register


@register(prefix="245760")
def dithering(img: np.ndarray) -> np.ndarray:
    img_grayscale = np.array(Image.fromarray(img, mode="RGB").convert("L"),
                         dtype=np.uint8)

    bayer_matrix = np.array([[0, 8, 2, 10],
                             [12, 4, 14, 6],
                             [3, 11, 1, 9],
                             [15, 7, 13, 5]],
                            dtype=np.float32)
    threshold_matrix = (bayer_matrix + 0.5) * (255.0 / bayer_matrix.size)

    height, width = img_grayscale.shape
    tiled_thresholds = np.tile(threshold_matrix,
                               (height // 4 + 1, width // 4 + 1))
    tiled_thresholds = tiled_thresholds[:height, :width]

    dithered = np.where(img_grayscale > tiled_thresholds, 255, 0).astype(np.uint8)

    return np.repeat(dithered[:, :, np.newaxis], 3, axis=2)
