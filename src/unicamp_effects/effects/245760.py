import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from unicamp_effects.registry import register

# --- Funções auxiliares para detecção de borda ---
def _mascara_tijolos(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    h_min, h_max = 4, 20
    s_min, s_max = 80, 255

    # mascara binaria 
    base_mask = np.zeros(h.shape, dtype=np.uint8)
    base_mask[
        (h >= h_min) & (h <= h_max) &
        (s >= s_min) & (s <= s_max) 
        ] = 255

    opened = cv2.morphologyEx(
        base_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    cleaned = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)),
    )

    return cleaned

def _magSobel(img: np.ndarray) -> np.ndarray:
    Sv = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Sv = Sv.reshape(3, 3)
    
    Sh = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Sh = Sh.reshape(3, 3)

    # Kernel do filtro gaussiano
    kernel_gaussiano = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1], dtype=float)
    kernel_gaussiano = kernel_gaussiano * (1/16)
    kernel_gaussiano = kernel_gaussiano.reshape((3, 3))

    img_gaussian = ndi.convolve(img, kernel_gaussiano, mode="constant",cval=0.0, output=float)

    imgv = ndi.convolve(img_gaussian, Sv)
    imgh = ndi.convolve(img_gaussian, Sh)

    img_filtered = np.sqrt(np.pow(imgv, 2) + np.pow(imgh, 2))
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)

    return(img_filtered)

# --- Funções auxiliares para aberração cromática ---
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

    return mask

# --- Efeitos implementados ---
@register(prefix="245760")
def deteccao_borda(img: np.ndarray) -> np.ndarray:

    # Imagem segmentada com mascara
    mask = _mascara_tijolos(img)
    img1 = np.zeros_like(img)
    img1[mask == 255] = img[mask == 255]

    img2 = _magSobel(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
    img2 = np.repeat(img2[:, :, np.newaxis], 3, axis=2)

    result = np.zeros_like(img)
    result[mask == 0] = img[mask == 0]
    result[mask == 255] = img2[mask == 255]

    return result.astype(np.uint8)

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
