import numpy as np
from scipy.ndimage import convolve

from unicamp_effects.registry import register


@register(prefix="175807")
def maria_e_sobel(img: np.ndarray) -> np.ndarray:
    # Criação dos filtros
    x = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])
    y = x.T

    # Transformando a imagem em greyscale usando média ponderada
    pesos = [0.299, 0.587, 0.114]
    f_grey = np.average(img, weights=pesos, axis=2).astype('uint8')

    # Aplicação de threshold 160 para retirar informações indesejadas
    T1 = np.arange(256).astype('uint8')
    tmp = (T1 > 160).astype(float) # threshold 160
    T4 = ((tmp-tmp.min())*255/(tmp.max()-tmp.min())).astype('uint8')  # Normalização
    f_grey = T4[f_grey]

    # Aplicação dos filtros horizontal e vertical
    foto_x = convolve(f_grey, x)
    foto_y = convolve(f_grey, y)

    # Cálculo da magnitude
    foto_mag = np.sqrt(foto_x**2 + foto_y**2).astype('uint8')
    foto_mag = foto_mag / foto_mag.max() * 255
    foto_mag = foto_mag.astype('uint8')

    foto_mag_3 = np.concatenate([foto_mag[:,:,None],
                                 foto_mag[:,:,None],
                                 foto_mag[:,:,None]], axis=2)
    
    return foto_mag_3

@register(prefix="175807")
def pixelular(img: np.ndarray) -> np.ndarray:
    # Adiciono padding para que a imagem possa ser quebrada em pixels de 16x16
    # Adiciono 4 pixels na direita e 9 pixels inferiores para totalizar 480x528 pixels
    h, w, _ = img.shape
    bloco = h // 10

    pad_h = (bloco - h % bloco) % bloco
    pad_w = (bloco - w % bloco) % bloco

    img_pad = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode='edge')

    H, W, _ = img_pad.shape

    img_bloco = img_pad.reshape(H//bloco, bloco, W//bloco, bloco, 3).transpose(0,2,1,3,4)

    means = img_bloco.mean(axis=(2,3), keepdims=True)

    img_pxl = np.repeat(np.repeat(means, bloco, axis=2), bloco, axis=3)

    img_out = img_pxl.transpose(0,2,1,3,4).reshape(H, W, 3)

    return img_out[:h, :w].astype(np.uint8)

@register(prefix="175807")
def futuro(img: np.ndarray) -> np.ndarray:
    h, w, canal = img.shape
    
    # Aberração Cromática
    img_r = img[:,:,0]
    img_g = img[:,:,1]
    img_b = img[:,:,2]
    d_1 = w // 10
    d_2 = w - d_1
    mask_r = np.zeros((h,w))
    mask_r[:,0:d_2:] = img_r[:,d_1:w:] # desloca 60 para esquerda
    mask_b = np.zeros((h,w))
    mask_b[:,d_1:w:] = img_b[:,0:d_2:] # desloca 60 para a direita
    img = np.concatenate([mask_r[:,:,None],
                          img_g[:,:,None],
                          mask_b[:,:,None]], axis=2)
    img = img.astype('uint8')

    # Aplicação de vinheta
    # Criação de máscara para aplicar na imagem
    y, x = np.indices((h, w))

    cy, cx = h//2, w//2

    dist = (x - cx)**2 + (y - cy)**2
    dist = dist.astype(float)

    dist = (dist - dist.min()) / (dist.max() - dist.min()) * 255

    matriz_3c = np.concatenate([dist[:,:,None],
                                dist[:,:,None],
                                dist[:,:,None]], axis=2)
    
    # Aplicação do filtro com cuidado com overflow
    img_vinheta = img.astype(float) - matriz_3c
    img_vinheta = np.clip(img_vinheta, 0, 255).astype('uint8')

    return img_vinheta

