import numpy as np

from unicamp_effects.registry import register

RA = "260363" 

@register(RA)
def borda(img: np.ndarray) -> np.ndarray:
    """
    Efeito de Detecção de Borda.
    Converte para tons de cinza e aplica uma diferença de gradiente básica para identificar as mudanças abruptas.
    """
    # convertemos para float para evitar cortes dos valores
    img_float = img.astype(float)
    
    # conversão manual para tons de cinza: Y = 0.299R + 0.587G + 0.114B
    gray = 0.2989 * img_float[:, :, 0] + 0.5870 * img_float[:, :, 1] + 0.1140 * img_float[:, :, 2]
    
    # Matrizes para armazenar os gradientes horizontais e verticais
    # Aqui guardaremos a diferença entre os píxels não mais seus valores absolutos
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    
    # Calculando a diferença entre pixels vizinhos (gradiente simples)
    # Para achar bordas verticais e horizontais
    gx[:, :-1] = np.diff(gray, axis=1)
    gy[:-1, :] = np.diff(gray, axis=0)
    
    # Magnitude ou tamanho do vetor gradiente
    # nem toda borda é só vertical ou horizontal
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Normalizando para a escala de 0 a 255 como já tínhamos antes
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max()) * 255
        
    # Concatenemos a imagem monocromática 3 vezes para manter o formato da saída
    result = np.stack([magnitude, magnitude, magnitude], axis=-1)
    
    return result.astype(np.uint8) # retornamos no tipo original u8


@register(RA)
def dof_quadro(img: np.ndarray) -> np.ndarray:
    """
    Depth of Field focado em uma região retangular (ex: quadro da aula de eletrônica).
    Máscara ainda limitada porque deve ser setada na mão no código para cada imagem diferente.
    """
    # Nesse efeito preservamos as cores
    h, w, c = img.shape
    # float novamente para não termos problemas com o tipo u8 de overflow
    img_float = img.astype(float)
    
    raio_blur = 15  # Quanto maior, mais borrado fica o fundo
    
    # Blur Horizontal
    blur_x = np.zeros_like(img_float)
    for dx in range(-raio_blur, raio_blur + 1):
        # essa função empurra todos os píxels da imagem de uma vez estamos 
        # gerrando 2 x raio_blur fotos, uma um pouco mais pro lado que as anteriores
        # dando assim a sensação de borrado
        blur_x += np.roll(img_float, shift=dx, axis=1) 
    blur_x /= (2 * raio_blur + 1) # somamos considerando a quantidade de fotos pra não estourar
    
    # Blur Vertical
    blur_y = np.zeros_like(blur_x)
    for dy in range(-raio_blur, raio_blur + 1):
        blur_y += np.roll(blur_x, shift=dy, axis=0) 
    borrada = blur_y / (2 * raio_blur + 1)
    
    # Vamos criar a máscara para só aplicar o efeito na área fora do quadro
    mask = np.zeros((h, w, 1), dtype=float)
    
    y_min, y_max = int(h * 0.35), int(h * 0.69)  # vai de 35% da altura total à 69%
    x_min, x_max = int(w * 0.02), int(w * 0.97)  # Vai de 2% da largura total até 97%
    
    # forçamos a região da mascara a ser 1 - valor máximo - branco
    mask[y_min:y_max, x_min:x_max] = 1.0
    
    # Se for a região branca, o pixel da imagem é recuperado
    # caso contrário, o píxel da região borrada é recuperado
    resultado = (img_float * mask) + (borrada * (1.0 - mask))
    
    return resultado.astype(np.uint8)

@register(RA)
def retro_tela(img: np.ndarray) -> np.ndarray:
    """
    Efeito retrô (Pixelização + Quantização) com máscara.
    Borra e reduz as cores apenas na área da tela do computador.
    """
    h, w, c = img.shape
    # ajustar o tamanho do píxel para parecer censura the sims - blocos grandes
    block_size = 64  
    
    # Pegamos 1 pixel a cada 'block_size'
    img_reduzida = img[::block_size, ::block_size]
    
    # Depois de reduzir a imagem, reduzimos a quantidade de cores a 4 tons por canal
    # Para que os tons se dividam de forma igual de 0 a 255 fazemos a seguinte operação:
    img_quantizada = (img_reduzida // 64) * (255/3)
    
    # Repetimos as linhas, e depois repetimos as colunas para criar o aspecto blocado
    # Usamos os vizinhos mais próximos para fazer essa interpolação
    retro_effect = np.repeat(np.repeat(img_quantizada, block_size, axis=0), block_size, axis=1)
    
    # Cortamos o shape exato que reduzimos antes
    retro_effect = retro_effect[:h, :w]
    
    # vamos repetir o mesmo processo para a criação da Máscara aqui
    mask = np.zeros((h, w, 1), dtype=float)
    
    y_min, y_max = int(h * 0.08), int(h * 0.55)
    x_min, x_max = int(w * 0.25), int(w * 0.70)
    
    mask[y_min:y_max, x_min:x_max] = 1.0
    
    img_float = img.astype(float)
    retro_float = retro_effect.astype(float)
    
    # Efeito retro aplicado onde a máscara é um, caso contrário não
    final = (img_float * (1.0 - mask)) + (retro_float * mask)
    
    return final.astype(np.uint8)