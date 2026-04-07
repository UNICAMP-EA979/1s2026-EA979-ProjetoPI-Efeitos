import numpy as np
from scipy.ndimage import convolve

from unicamp_effects.registry import register


@register(prefix="241163")
def edge_detection(img: np.ndarray) -> np.ndarray:
    # Converte para grayscale
    gray = np.dot(img, np.array([0.299, 0.587, 0.114]))

    # Máscaras de Sobel
    Sv = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Sh = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    # Convolucoes
    fv = convolve(gray, Sv)
    fh = convolve(gray, Sh)

    # Magnitude do gradiente
    mag = np.sqrt(fh**2 + fv**2)

    # Normalização
    mag = (mag / mag.max() * 255).astype(np.uint8)

    # Replicar para RGB
    return np.stack([mag, mag, mag], axis=-1)


@register(prefix="241163")
def chromatic_aberration(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    offset = max(1, w // 100)  # deslocamento proporcional

    # Separar canais
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Criar cópias deslocadas
    R_shifted = np.roll(R, shift=offset, axis=1)
    B_shifted = np.roll(B, shift=-offset, axis=1)

    # Evitar wrap-around (preencher bordas)
    R_shifted[:, :offset] = 0
    B_shifted[:, -offset:] = 0

    # Recombinar
    return np.stack([R_shifted, G, B_shifted], axis=-1)


@register(prefix="241163")
def pixelation(img: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    block_size = max(2, min(h, w) // 50)

    # Ajustar dimensões para múltiplos do bloco
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    img_cropped = img[:h_crop, :w_crop]

    # Reorganizar para blocos
    reshaped = img_cropped.reshape(
        h_crop // block_size, block_size,
        w_crop // block_size, block_size,
        c
    )

    # Média por bloco
    block_means = reshaped.mean(axis=(1, 3), keepdims=True)

    # Expandir de volta
    pixelated = np.repeat(np.repeat(block_means, block_size, axis=1),
                          block_size, axis=3)

    return pixelated.reshape(h_crop, w_crop, c).astype(np.uint8)