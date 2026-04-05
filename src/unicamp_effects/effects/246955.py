import numpy as np

from unicamp_effects.registry import register

# funções auxiliares
def rgb2hsv(img: np.ndarray) -> np.ndarray:
    '''converte imagem rgb para hsv com dtype float64: h de 0 a 360, s e v de 0 a 1'''
    img_float = img / 255.0

    r = img_float[:, :, 0]
    g = img_float[:, :, 1]
    b = img_float[:, :, 2]

    c_max = np.max(img_float, axis=2)
    c_min = np.min(img_float, axis=2)
    v = c_max # value é intensidade do canal mais forte

    delta = c_max - c_min
    s_mask = c_max != 0

    s = np.zeros(v.shape)
    s[s_mask] = delta[s_mask] / c_max[s_mask] # s é diferença entre canal mais forte e mais fraco dividido por valor do canal mais forte

    r_mask = (c_max == r) & (delta != 0)
    g_mask = (c_max == g) & (delta != 0)
    b_mask = (c_max == b) & (delta != 0)

    # calculo de hue baseado em fórmula
    h = np.zeros(v.shape)
    h[r_mask] = (60 * ((g[r_mask] - b[r_mask])/delta[r_mask])) % 360
    h[g_mask] = (60 * ((b[g_mask] - r[g_mask])/delta[g_mask] + 2)) % 360
    h[b_mask] = (60 * ((r[b_mask] - g[b_mask])/delta[b_mask] + 4)) % 360

    return np.dstack((h, s, v))

def hsv2rgb(img: np.ndarray) -> np.ndarray:
    '''converte imagem hsv (no formato retornado por rgb2hsv) em imagem rgb uint8'''
    # programa baseado em fórmula de conversão
    h = img[:, :, 0]
    s = img[:, :, 1]
    v = img[:, :, 2]

    c = v * s
    x = c * (1 - np.abs(((h/60) % 2) - 1))
    m = v - c

    r = np.zeros(v.shape)
    g = np.zeros(v.shape)
    b = np.zeros(v.shape)

    mask1 = h < 60
    mask2 = (h >= 60) & (h < 120)
    mask3 = (h >= 120) & (h < 180)
    mask4 = (h >= 180) & (h < 240)
    mask5 = (h >= 240) & (h < 300)
    mask6 = (h >= 300) & (h < 360)

    r[mask1] = c[mask1]
    g[mask1] = x[mask1]

    r[mask2] = x[mask2]
    g[mask2] = c[mask2]

    g[mask3] = c[mask3]
    b[mask3] = x[mask3]

    g[mask4] = x[mask4]
    b[mask4] = c[mask4]

    b[mask5] = c[mask5]
    r[mask5] = x[mask5]

    b[mask6] = x[mask6]
    r[mask6] = c[mask6]

    r += m
    g += m
    b += m

    r *= 255
    g *= 255
    b *= 255

    return np.dstack((r, g, b)).astype(np.uint8)

def vignette(img: np.ndarray, i:int) -> np.ndarray:
    '''aplica efeito de vinheta em imagem. i -> intensidade do efeito'''
    # cria grid com (0, 0) no centro
    x = np.arange(-img.shape[1]//2, img.shape[1]//2)
    y = np.arange(-img.shape[0]//2, img.shape[0]//2)
    ym, xm = np.meshgrid(y, x, indexing='ij')

    xm_norm = xm*2/(xm.max() - xm.min())
    ym_norm = ym*2/(xm.max() - xm.min()) # normaliza tudo por maior valor da imagem

    # matriz de distâncias do centro
    r = np.sqrt(ym_norm**2 + xm_norm**2)

    # mascara para escurecer pixels linearmente
    mask = 1 - np.clip(r * i, 0, 1)

    # aplica mascara e altera dtype para uint8
    ret = np.zeros(img.shape)
    ret[:, :, 0] = img[:, :, 0] * mask
    ret[:, :, 1] = img[:, :, 1] * mask
    ret[:, :, 2] = img[:, :, 2] * mask

    return ret.astype(np.uint8)

@register(prefix="246955")
def restricted_chromatic_aberration_and_vignette(img: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(img)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]

    # seleciona bituqueira
    # (OBS: tentei selecionar analisando hue e saturation, porém resultado não ficou tão bom quanto esperado
    # (chão também foi selecionado))
    seg = ((h < 45) | (h > 180)) & (s < 0.3)
    bituqueira = img * seg[..., None]

    # aberração cromática apenas nos pixels selecionados
    bituqueira[:, :, 0] = np.roll(bituqueira[:, :, 0], (40, -40), (0, 1))
    bituqueira[:, :, 2] = np.roll(bituqueira[:, :, 2], (-40, 40), (0, 1))

    res_r = np.where(bituqueira[:, :, 0], bituqueira[:, :, 0], img[:, :, 0])
    res_g = img[:, :, 1]
    res_b = np.where(bituqueira[:, :, 2], bituqueira[:, :, 2], img[:, :, 2])

    abrr = (np.dstack((res_r, res_g, res_b)))

    # por fim, aplica vinheta
    return vignette(abrr, 0.8)

@register(prefix="246955")
def color_mapping(img: np.ndarray) -> np.ndarray:
    # muda representação de imagem para HSV
    hsv = rgb2hsv(img)

    h = hsv[:, :, 0].astype(np.uint16) # uint16 pois h pertence a [0, 359]

    # vetor de mapeamento de hue, apenas shiftado
    T = np.arange(360)
    T = np.roll(T, 120)

    # aplica transformação de "intensidade" por mapeamento no canal de hue
    hsv[:, :, 0] = T[h].astype(np.float64)

    # retorna imagem em formato RGB
    return hsv2rgb(hsv)

import scipy # usado para mapeamento de coordenadas com interpolação

@register(prefix="246955")
def fish_eye(img: np.ndarray) -> np.ndarray:
    # cria grid com 0, 0 no centro da imagem
    x = np.arange(-img.shape[1]//2, img.shape[1]//2)
    y = np.arange(-img.shape[0]//2, img.shape[0]//2)
    ym, xm = np.meshgrid(y, x, indexing='ij')

    # normaliza de -1 a 1
    xm_norm = xm*2/(xm.max() - xm.min())
    ym_norm = ym*2/(xm.max() - xm.min()) # normaliza tudo por maior valor da imagem

    # converte para coordenadas polares
    r = np.sqrt(xm_norm**2 + ym_norm**2)
    theta = np.arctan2(ym_norm, xm_norm)

    # modelo polinomial para distorção
    k = 0.25
    rd = r * (1 + k * r**2)

    # aplica zoom para encaixar imagem no quadro (remover quinas pretas)
    rd_max = r.max() * (1 + k * r.max()**2)
    rd_zoom = rd * r.max()/rd_max

    # passa de coordenadas polares para cartesianas após distorção
    x_eye_norm = rd_zoom * np.cos(theta)
    y_eye_norm = rd_zoom * np.sin(theta)

    # desnormaliza
    x_eye = x_eye_norm * (xm.max() - xm.min()) / 2
    x_eye -= xm.min() # retorna centro para origem
    y_eye = y_eye_norm * (xm.max() - xm.min()) / 2
    y_eye -= ym.min()

    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]

    # mapeia pixels da imaegm original em pixels indicados pelo grid distorcido (usando interpolação)
    distorted_red = scipy.ndimage.map_coordinates(img_r, [y_eye, x_eye])
    distorted_green = scipy.ndimage.map_coordinates(img_g, [y_eye, x_eye])
    distorted_blue = scipy.ndimage.map_coordinates(img_b, [y_eye, x_eye])

    return np.dstack((distorted_red, distorted_green, distorted_blue))