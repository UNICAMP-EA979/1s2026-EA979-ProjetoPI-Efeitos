import numpy as np
from scipy.ndimage import convolve, map_coordinates
from unicamp_effects.registry import register
import numpy as np
from scipy.ndimage import convolve, map_coordinates
from unicamp_effects.registry import register

def _convolucao_2d(imagem_canal: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    padded = np.pad(imagem_canal, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    resultado = np.zeros_like(imagem_canal, dtype=np.float32)
    
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

    x_distorcido = np.clip(x_distorcido, 0, largura - 1.001)
    y_distorcido = np.clip(y_distorcido, 0, altura - 1.001)

    x0 = np.floor(x_distorcido).astype(int)
    y0 = np.floor(y_distorcido).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x_distorcido - x0
    dy = y_distorcido - y0

    dx = dx[..., np.newaxis]
    dy = dy[..., np.newaxis]

    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy

    imagem_float = img.astype(np.float32)
    
    imagem_distorcida = (imagem_float[y0, x0] * w00 + 
                         imagem_float[y0, x1] * w10 + 
                         imagem_float[y1, x0] * w01 + 
                         imagem_float[y1, x1] * w11)

    return np.clip(imagem_distorcida, 0, 255).astype(np.uint8)

@register(prefix="177884")
def sobel_filter(img: np.ndarray) -> np.ndarray:

    img = img.astype(np.float32)
    
    Sv = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Sh = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    img_sh = convolve(img, Sh)
    img_sv = convolve(img, Sv)

    magnitude = np.sqrt(img_sh**2 + img_sv**2)
    magnitude =  (magnitude/magnitude.max() *255).astype('uint8')

    return np.stack([magnitude, magnitude, magnitude], axis = -1)