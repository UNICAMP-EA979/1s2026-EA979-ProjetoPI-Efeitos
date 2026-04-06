import numpy as np
from scipy.ndimage import convolve, map_coordinates
from unicamp_effects.registry import register
import numpy as np
from scipy.ndimage import convolve, map_coordinates
from unicamp_effects.registry import register

def _convolucao_2d(imagem_canal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Função auxiliar para aplicar convolução 2D sem usar scipy.
    Utiliza fatiamento (slicing) do NumPy para ser rápida e evitar loops pixel a pixel.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Preenche as bordas repetindo os pixels vizinhos (equivalente ao mode='edge' ou 'nearest')
    padded = np.pad(imagem_canal, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    resultado = np.zeros_like(imagem_canal, dtype=np.float32)
    
    # Desliza o kernel sobre a imagem somando as matrizes multiplicadas
    for i in range(kh):
        for j in range(kw):
            resultado += padded[i:i+imagem_canal.shape[0], j:j+imagem_canal.shape[1]] * kernel[i, j]
            
    return resultado

@register(prefix="177884")
def normal_blur(img: np.ndarray) -> np.ndarray:
    # Kernel da média
    tamanho = 15
    kernel_media = np.ones((tamanho, tamanho), dtype=np.float32) / (tamanho ** 2)

    imagem_float = img.astype(np.float32)
    imagem_borrada = np.zeros_like(imagem_float)

    # Aplica a convolução canal por canal
    for canal in range(3):
        imagem_borrada[:, :, canal] = _convolucao_2d(imagem_float[:, :, canal], kernel_media)

    return np.clip(imagem_borrada, 0, 255).astype(np.uint8)

@register(prefix="177884")
def pincushion_distortion(img: np.ndarray) -> np.ndarray:
    altura, largura = img.shape[:2]
    k = 0.0000025

    y, x = np.mgrid[0:altura, 0:largura]

    x_centralizado = x - largura / 2.0
    y_centralizado = y - altura / 2.0

    r_quadrado = x_centralizado**2 + y_centralizado**2
    fator_distorcao = 1.0 + k * r_quadrado

    x_distorcido = x_centralizado / fator_distorcao + largura / 2.0
    y_distorcido = y_centralizado / fator_distorcao + altura / 2.0

    # Evita que as coordenadas vazem para fora da imagem antes da interpolação
    x_distorcido = np.clip(x_distorcido, 0, largura - 1.001)
    y_distorcido = np.clip(y_distorcido, 0, altura - 1.001)

    # Interpolação Bilinear implementada manualmente com NumPy
    x0 = np.floor(x_distorcido).astype(int)
    y0 = np.floor(y_distorcido).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x_distorcido - x0
    dy = y_distorcido - y0

    # Expandir dimensões para permitir broadcast com os 3 canais de cor
    dx = dx[..., np.newaxis]
    dy = dy[..., np.newaxis]

    # Pesos da interpolação
    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy

    imagem_float = img.astype(np.float32)
    
    # Amostra os 4 pixels vizinhos e multiplica pelos seus respectivos pesos
    imagem_distorcida = (imagem_float[y0, x0] * w00 + 
                         imagem_float[y0, x1] * w10 + 
                         imagem_float[y1, x0] * w01 + 
                         imagem_float[y1, x1] * w11)

    return np.clip(imagem_distorcida, 0, 255).astype(np.uint8)

@register(prefix="177884")
def sobel_filter(img: np.ndarray) -> np.ndarray:
    imagem_float = img.astype(np.float32)

    # Conversão para tons de cinza
    cinza = np.dot(imagem_float[..., :3], [0.299, 0.587, 0.114])

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    Gx = _convolucao_2d(cinza, Kx)
    Gy = _convolucao_2d(cinza, Ky)

    magnitude = np.hypot(Gx, Gy)

    # Normaliza a magnitude
    valor_maximo = np.max(magnitude)
    if valor_maximo > 0:
        magnitude = magnitude / valor_maximo

    magnitude_uint8 = (magnitude * 255).astype(np.uint8)

    # Replicação do canal monocromático para 3 canais (R=G=B)
    return np.stack([magnitude_uint8, magnitude_uint8, magnitude_uint8], axis=-1)