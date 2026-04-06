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
    foto_pad = np.pad(img, ((0,9),(0,4),(0,0)), mode='edge')

    # Quebro a imagem em bloquinhos de 16x16, tomando cuidado para pegar as posições corretas de pixels
    foto_bloco = foto_pad.reshape(30, 16, 33, 16, 3).transpose(0,2,1,3,4)

    # Tiro a média dos pixels separados em rgb dentro desses bloquinhos
    means = foto_bloco.mean(axis=(2,3), keepdims=True)

    # Expansão da média
    foto_pxl = np.repeat(np.repeat(means, 16, axis=2), 16, axis=3)

    # Volta ao shape original
    foto_ori = foto_pxl.transpose(0,2,1,3,4).reshape(480,528,3).astype('uint8')

    foto_crop = foto_ori[0:471:,0:524:,:]
    return foto_crop

@register(prefix="175807")
def futuro(img: np.ndarray) -> np.ndarray:
    # Aberração Cromática
    img_r = img[:,:,0]
    img_g = img[:,:,1]
    img_b = img[:,:,2]
    mask_r = np.zeros((3000,4000))
    mask_r[:,0:3940:] = img_r[:,60:4000:] # desloca 60 para esquerda
    mask_b = np.zeros((3000,4000))
    mask_b[:,60:4000:] = img_b[:,0:3940:] # desloca 60 para a direita
    img = np.concatenate([mask_r[:,:,None],
                          img_g[:,:,None],
                          mask_b[:,:,None]], axis=2)
    img = img.astype('uint8')

    # Aplicação de vinheta
    # Criação de máscara para aplicar na imagem
    aux = np.indices((1500,2000))
    aux = (aux[0]**2 + aux[1]**2)
    aux = aux.astype(float)
    aux = (aux - aux.min()) / (aux.max() - aux.min()) * 255
    aux = aux.astype('uint8')
    matriz_11 = aux
    matriz_00 = aux[::-1,::-1]
    matriz_01 = aux[::-1,::1]
    matriz_10 = aux[::1,::-1]
    matriz = np.zeros((3000,4000))
    matriz[0:1500:,0:2000:] = matriz_00
    matriz[0:1500:,2000:4000:] = matriz_01
    matriz[1500:3000:,0:2000:] = matriz_10
    matriz[1500:3000:,2000:4000:] = matriz_11
    matriz_3c = np.concatenate([matriz[:,:,None],
                                matriz[:,:,None],
                                matriz[:,:,None]], axis=2)
    # Aplicação do filtro com cuidado com overflow
    img_vinheta = img.astype(float) - matriz_3c
    img_vinheta = np.clip(img_vinheta, 0, 255).astype('uint8')
    return img_vinheta
