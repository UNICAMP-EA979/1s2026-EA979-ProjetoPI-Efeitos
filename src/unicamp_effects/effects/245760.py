import cv2
import numpy as np
from PIL import Image

from unicamp_effects.registry import register

# --- Funções auxiliares para aberração cromática
def _shift_channel(channel: np.ndarray, shift_x: int) -> np.ndarray:
    h, w = channel.shape

    if shift_x == 0:
        return channel

    if shift_x > 0:
        padded = np.pad(channel, ((0, 0), (shift_x, 0)), mode="edge")
        return padded[:, :w]

    offset = -shift_x
    padded = np.pad(channel, ((0, 0), (0, offset)), mode="edge")
    return padded[:, offset:offset + w]

def _mascara_roxo(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    h_min, h_max = 120, 170
    s_min, s_max = 70, 255

    # mascara binaria 
    mask = np.zeros(h.shape, dtype=np.uint8)
    mask[
        (h >= h_min) & (h <= h_max) &
        (s >= s_min) & (s <= s_max)] = 255

    # fundo em grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # background = np.repeat(gray[:, :, np.newaxis], 3, axis=2)

    return mask

@register(prefix="245760")
def aberracao_cromatica(img: np.ndarray) -> np.ndarray:
    mask = _mascara_roxo(img)

    img_segmentada = np.zeros_like(img)
    img_segmentada[mask == 255] = img[mask == 255]
    
    # aberracao cromatica
    shift = 8
    red = _shift_channel(img_segmentada[:, :, 0], shift)
    green = img_segmentada[:, :, 1]
    blue = _shift_channel(img_segmentada[:, :, 2], -shift)
    aberrated = np.stack([red, green, blue], axis=2)

    # fundo em grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    background = np.repeat(gray[:, :, np.newaxis], 3, axis=2)

    result = np.zeros_like(img)
    result[mask == 0] = background[mask == 0]
    result[mask == 255] = aberrated[mask == 255]

    return result.astype(np.uint8)

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
