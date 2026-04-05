import numpy as np
from unicamp_effects.registry import register

# ==============================================================================
# FUNÇÕES AUXILIARES GERAIS
# ==============================================================================

def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """Converte uma imagem RGB para Tons de Cinza usando a proporção padrão de luminância."""
    gray = np.dot(img[..., :3],[0.2989, 0.5870, 0.1140])
    return gray

def apply_1d_kernel(img: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Aplica um kernel 1D usando vetorização do Numpy (deslocamento/roll)."""
    res = np.zeros_like(img, dtype=np.float32)
    k_center = len(kernel) // 2
    for i, val in enumerate(kernel):
        shift = i - k_center
        res += np.roll(img, shift, axis=axis) * val
    return res

def fast_gaussian_blur(img: np.ndarray) -> np.ndarray:
    """Aplica um blur Gaussiano aproximado 5x5 usando convolução separável (muito mais rápido)."""
    # Kernel gaussiano 1D
    k = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    
    # Se a imagem tiver 3 canais, aplicamos em cada canal simultaneamente
    blurred = apply_1d_kernel(img.astype(np.float32), k, axis=0) # Blur vertical
    blurred = apply_1d_kernel(blurred, k, axis=1)                # Blur horizontal
    return blurred

# ==============================================================================
# EFEITO 1: COLOR SPLASH COM MÁSCARA RADIAL
# ==============================================================================

@register(prefix="186629")
def color_splash(img: np.ndarray) -> np.ndarray:
    """
    Mantém o centro/foco da imagem colorido e transforma o resto em preto e branco.
    Usa uma máscara gaussiana para um degradê suave (sem recortes retangulares duros).
    """
    H, W = img.shape[:2]
    
    # 1. Converte para tons de cinza e replica para 3 canais
    gray = rgb_to_gray(img)
    gray_3c = np.stack([gray, gray, gray], axis=-1)
    
    # 2. Cria uma malha de coordenadas
    Y, X = np.ogrid[:H, :W]
    
    # Define o centro de foco (centro-inferior, ideal para fotos de objetos no chão como troncos)
    center_y, center_x = int(H * 0.6), W // 2 
    
    # Eixos da elipse de foco (raios)
    a, b = W // 2.5, H // 4
    
    # Calcula a distância de cada pixel para o centro da elipse
    dist_sq = ((X - center_x) / a)**2 + ((Y - center_y) / b)**2
    
    # 3. Cria a máscara usando decaimento exponencial (Gaussiano) para bordas suaves
    mask = np.exp(-dist_sq)
    mask = np.clip(mask, 0, 1)[..., np.newaxis] # Ajusta dimensões para multiplicar
    
    # 4. Mescla as imagens: onde a máscara é 1 fica colorido, onde é 0 fica cinza
    result = (img * mask + gray_3c * (1.0 - mask))
    
    return result.astype(np.uint8)

# ==============================================================================
# EFEITO 2: DESFOQUE COM ABERRAÇÃO CROMÁTICA (BOKEH ANALÓGICO)
# ==============================================================================

@register(prefix="186629")
def chromatic_aberration_blur(img: np.ndarray) -> np.ndarray:
    """
    Aplica um desfoque (blur) na imagem e separa/desloca os canais de cor 
    para simular uma lente analógica com defeito em luzes noturnas.
    """
    # 1. Aplica o desfoque Gaussiano na imagem inteira
    blurred = fast_gaussian_blur(img)
    
    # 2. Inicializa a imagem final vazia
    result = np.zeros_like(img, dtype=np.float32)
    
    # Define a intensidade do deslocamento (aberração)
    shift_val = 8
    
    # 3. Deslocamento dos canais (Aberração Cromática)
    # Canal Vermelho (R) vai para diagonal superior esquerda
    result[..., 0] = np.roll(blurred[..., 0], shift=(-shift_val, -shift_val), axis=(0, 1))
    
    # Canal Verde (G) fica no centro (ancora a imagem)
    result[..., 1] = blurred[..., 1]
    
    # Canal Azul (B) vai para diagonal inferior direita
    result[..., 2] = np.roll(blurred[..., 2], shift=(shift_val, shift_val), axis=(0, 1))
    
    # Garante que os valores fiquem no intervalo correto de cor
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)

# ==============================================================================
# EFEITO 3: CANNY EDGE DETECTION (IMPLEMENTAÇÃO COM NUMPY VETORIZADO)
# ==============================================================================

@register(prefix="186629")
def canny_edge_detection(img: np.ndarray) -> np.ndarray:
    """
    Extrai contornos da imagem usando o Algoritmo de Canny, implementado do zero.
    Gera um fundo escuro com linhas brancas destacando geometria.
    """
    # Passo 1: Tons de Cinza e Blur Gaussiano para remover ruídos
    gray = rgb_to_gray(img)
    blurred = fast_gaussian_blur(gray)
    
    # Passo 2: Derivadas (Diferença Central Simples - Substitui a matriz 2D que gerava o erro)
    gx = np.roll(blurred, -1, axis=1) - np.roll(blurred, 1, axis=1)
    gy = np.roll(blurred, -1, axis=0) - np.roll(blurred, 1, axis=0)
    
    # Magnitude e Direção (Ângulo)
    mag = np.hypot(gx, gy)
    mag_max = mag.max()
    if mag_max > 0:
        mag = mag / mag_max * 255.0
        
    theta = np.arctan2(gy, gx)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    
    # Passo 3: Supressão de Não-Máximos (Deixa os traços finos)
    # Recupera os pixels vizinhos deslocando as matrizes
    mag_left, mag_right = np.roll(mag, 1, axis=1), np.roll(mag, -1, axis=1)
    mag_up, mag_down = np.roll(mag, 1, axis=0), np.roll(mag, -1, axis=0)
    mag_up_left, mag_down_right = np.roll(mag_up, 1, axis=1), np.roll(mag_down, -1, axis=1)
    mag_up_right, mag_down_left = np.roll(mag_up, -1, axis=1), np.roll(mag_down, 1, axis=1)
    
    # Verifica em qual direção o gradiente aponta e compara com os vizinhos dessa direção
    nms_0 = (mag >= mag_left) & (mag >= mag_right)
    nms_45 = (mag >= mag_up_right) & (mag >= mag_down_left)
    nms_90 = (mag >= mag_up) & (mag >= mag_down)
    nms_135 = (mag >= mag_up_left) & (mag >= mag_down_right)
    
    mask_0 = ((angle < 22.5) | (angle >= 157.5)) & nms_0
    mask_45 = ((angle >= 22.5) & (angle < 67.5)) & nms_45
    mask_90 = ((angle >= 67.5) & (angle < 112.5)) & nms_90
    mask_135 = ((angle >= 112.5) & (angle < 157.5)) & nms_135
    
    Z = np.where(mask_0 | mask_45 | mask_90 | mask_135, mag, 0)
    
    # Passo 4: Limiarização (Thresholding Duplo)
    high_threshold = Z.max() * 0.15 # 15% do valor máximo
    low_threshold = high_threshold * 0.05
    
    res = np.zeros_like(Z)
    strong = 255
    weak = 75
    
    res[Z >= high_threshold] = strong
    res[(Z <= high_threshold) & (Z >= low_threshold)] = weak
    
    # Passo 5: Rastreamento de borda por histerese (Conecta pixels fracos a fortes)
    # Se um pixel fraco está perto de um forte, ele vira forte.
    for _ in range(2): # 2 iterações para expandir os contornos
        strong_dilated = (
            np.roll(res == strong, 1, axis=0) | np.roll(res == strong, -1, axis=0) |
            np.roll(res == strong, 1, axis=1) | np.roll(res == strong, -1, axis=1) |
            np.roll(np.roll(res == strong, 1, axis=0), 1, axis=1) |
            np.roll(np.roll(res == strong, 1, axis=0), -1, axis=1) |
            np.roll(np.roll(res == strong, -1, axis=0), 1, axis=1) |
            np.roll(np.roll(res == strong, -1, axis=0), -1, axis=1)
        )
        res[(res == weak) & strong_dilated] = strong
        
    # Limpa o resto e finaliza
    res[res != strong] = 0
    
    # O Canny original gera imagem P&B (1 canal). Replicamos para 3 canais (RGB).
    res_final = res.astype(np.uint8)
    return np.stack([res_final, res_final, res_final], axis=-1)