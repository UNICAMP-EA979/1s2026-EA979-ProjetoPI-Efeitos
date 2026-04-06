import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from unicamp_effects.registry import register
@register(245609)

#Para a primeira imagem, usaremos detecção de borda e inversão de cores para criar um efeito de desenho a lápis.
#função que recebe uma imagem em array e retorna a imagem com o efeito.
def borda_lapis(img: np.ndarray) -> np.ndarray:
    #conversão para escalas de cinza
    gray = np.mean(img, axis=2)
    #detecção de borda (diferença entre pixels vizinhos),
    #compara o pixel atual com o do lado (eixo X) e com o de baixo (eixo Y)
    diff_x = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    diff_y = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    bordas = diff_x + diff_y
    bordas = np.clip(bordas * 3, 0, 255) #aumentar o contraste 
    
    desenho = 255 - bordas #inversão das cores para efeito de desenho

    desenho = desenho.astype(np.uint8)
    
    out = np.stack([desenho, desenho, desenho], axis=2)
    return out

#Para a segunda imagem, utilizaremos de alguns efeitos aprendidos em processamento de imagens
#para dar uma sensação de onda de calor. A primeira técnica usada o aumento da intensidade do canal vermelho.
#Depois, utilizaremos a aberração cromática vertical para parecer que as linhas vermelhas de calor estão subinndo. 
#Depois, aplicaremos o blur para desfoque para simular a distorção causada pelo ar quente. 

@register(245609)
def onda_de_calor(img: np.ndarray) -> np.ndarray:
    img_float = img.astype(np.float32) #conversão para float
    img_float[:, :, 0] = img_float[:, :, 0] * 1.7 #aumento do canal vermelho
    img_float[:, :, 2] = img_float[:, :, 2] * 0.5 #diminuicao do canal azul para nao ficar em tom roxo
    
    #aberracao cromatica desloca os canais de luz
    out = np.zeros_like(img_float)
    deslocamento = 30
    #deslocamento vertical do canal vermelho
    out[:-deslocamento, :, 0] = img_float[deslocamento:, :, 0]
    out[-deslocamento:, :, 0] = img_float[-deslocamento:, :, 0] 
    
    out[:, :, 1] = img_float[:, :, 1]
    out[:, :, 2] = img_float[:, :, 2]

    #o efeito blur será feito utilizando o filtro da média
    k = 7 #tamanho do kernel
    pad = k // 2 #pixels da borda
    
    #tratamento de borda: método extend
    out_pad = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    janelas = sliding_window_view(out_pad, window_shape=(k, k), axis=(0, 1))
    img_blur = np.mean(janelas, axis=(-2, -1))
    out_final = np.clip(img_blur, 0, 255).astype(np.uint8)
    return out_final

#Para a terceira imagem, será feito uma simulação de lente fisheye, para que pareça que
#estamos olhando a entrada do prédio por uma porta. Também iremos separar a cor azul do céu e
#deixar o resto em tons de cinza.
@register(245609)
def fisheye(img: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    img_float = img.astype(np.float32)
    #separaçã dos canais de cor
    r = img_float[:, :, 0]
    g = img_float[:, :, 1]
    b = img_float[:, :, 2]
    #máscara de identificação do azul
    mascara_ceu = (b > r + 30) & (b > g + 5) & (b > 80)
    mascara_ceu = mascara_ceu[:, :, np.newaxis] 

    cinza = np.mean(img_float, axis=2, keepdims=True) #fundo em tons de cinza
    img_cinza = np.concatenate([cinza, cinza, cinza], axis=2)

    img_destaque = np.where(mascara_ceu, img_float, img_cinza)
    #fisheye com mapeamento reverso
    Y, X = np.ogrid[:h, :w]
    centro_y, centro_x = h / 2.0, w / 2.0
    raio_maximo = min(centro_y, centro_x) #para o fisheye ficar em um círculo
    #calculo da distancia de cada pixel do centro
    dy = (Y - centro_y) / raio_maximo
    dx = (X - centro_x) / raio_maximo
    raio_pixel = np.sqrt(dx**2 + dy**2)
    raio_seguro = np.clip(raio_pixel, 1e-6, None)
    #intensidade da distorção
    fator_distorcao = 1.5
    raio_esfera = raio_seguro ** fator_distorcao
    #aplicação do mapeamento reverso
    src_y = centro_y + (dy / raio_seguro) * raio_esfera * raio_maximo
    src_x = centro_x + (dx / raio_seguro) * raio_esfera * raio_maximo
    src_y_int = np.clip(np.round(src_y), 0, h - 1).astype(np.int32)
    src_x_int = np.clip(np.round(src_x), 0, w - 1).astype(np.int32)

    out_fisheye = img_destaque[src_y_int, src_x_int]
    #tratamento de borda: pixels fora do raio máximo ficam pretos
    mascara_borda = (raio_pixel > 1.0)[:, :, np.newaxis]
    out_fisheye = np.where(mascara_borda, 0, out_fisheye)

    return out_fisheye.astype(np.uint8)