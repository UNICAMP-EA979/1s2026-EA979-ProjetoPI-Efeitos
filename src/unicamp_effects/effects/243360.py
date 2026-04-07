import numpy as np

import scipy as scp
from unicamp_effects.registry import register
from scipy.ndimage import map_coordinates
from scipy.ndimage import affine_transform

@register(prefix="243360")
def edge_detection(img: np.ndarray) -> np.ndarray:
    # Tons de Cinza (Pesos da norma ITU-R 601-2)
    # Assume que a imagem está em formato RGB (NumPy array)
    size_img = len(img)
    if len(img.shape) == 3:
        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    else:
        gray = img

    # Filtro Gaussiano (Kernel 5x5, sigma=1.0)
    def gaussian_kernel(size, sigma=1.0):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
        return g

    kernel_blur = gaussian_kernel(5, sigma=1.0)
    blurred = scp.signal.convolve(gray, kernel_blur, mode='same')

    # Gradiente (Sobel)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = scp.signal.convolve(blurred, Kx, mode='same')
    Iy = scp.signal.convolve(blurred, Ky, mode ='same')
    
    # Magnitude e Direção do Gradiente
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    # Supressão de Não-Máximos (Simplificado)
    # Aqui removemos pixels que não são o pico local na direção do gradiente
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q, r = 255, 255
            # Direção 0 (Horizontal)
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q, r = G[i, j+1], G[i, j-1]
            # Direção 45 (Diagonal)
            elif (22.5 <= angle[i,j] < 67.5):
                q, r = G[i+1, j-1], G[i-1, j+1]
            # Direção 90 (Vertical)
            elif (67.5 <= angle[i,j] < 112.5):
                q, r = G[i+1, j], G[i-1, j]
            # Direção 135 (Diagonal)
            elif (112.5 <= angle[i,j] < 157.5):
                q, r = G[i-1, j-1], G[i+1, j+1]

            if (G[i,j] >= q) and (G[i,j] >= r):
                Z[i,j] = G[i,j]
            else:
                Z[i,j] = 0

    # Histérese (Thresholding)
    low_threshold = 50
    high_threshold = 150
    
    res = np.zeros((M,N), dtype=np.int32)
    strong_i, strong_j = np.where(Z >= high_threshold)
    weak_i, weak_j = np.where((Z <= high_threshold) & (Z >= low_threshold))
    
    res[strong_i, strong_j] = 255
    res[weak_i, weak_j] = 50 # Marcado como fraco
    
    # Nota: O Canny real conecta pixels fracos se estiverem perto de fortes.
    # Para simplicidade, este código retorna o mapa de intensidades filtrado.
    res_rgb = np.stack([res]*3, axis=-1)
    return res_rgb.astype(np.uint8)
# --- SEPARAÇÃO E DISTORÇÃO GEOMÉTRICA ---
def distort_channel_manual(channel, radial_mask, center_x, center_y, scale_factor):
    h, w = channel.shape
    # Criamos o mapa de coordenadas original (y, x) para o scipy
    # O scipy espera as coordenadas de destino organizadas por eixos
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        
    # Fator de deslocamento baseado na máscara radial
    offset_factor = 1 + (scale_factor - 1) * radial_mask
        
    # Calculamos para onde cada pixel "deve olhar" na imagem original
    # (Invertemos a lógica da escala para o remapeamento)
    map_x = (x_coords - center_x) * offset_factor + center_x
    map_y = (y_coords - center_y) * offset_factor + center_y
    
    # Organizamos as coordenadas para o formato exigido pelo map_coordinates: 
    # uma matriz 2xN onde a primeira linha é Y e a segunda é X
    coords = np.array([map_y.ravel(), map_x.ravel()])
    
    # Realiza a interpolação bilinear (order=1)
    # cval=0 define a cor dos pixels que "saírem" da borda da imagem
    distorted = map_coordinates(channel, coords, order=1, mode='constant', cval=0)
    
    return distorted.reshape(h, w)

@register(prefix="243360")
def chromatic_aberration(img: np.ndarray) -> np.ndarray:
    intensity = 0.005
    if img is None: return None
    
    h, w, c = img.shape
    center_x, center_y = w / 2.0, h / 2.0

    # MÁSCARA DE INTENSIDADE RADIAL
    y, x = np.indices((h, w))
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    radial_mask = distance_from_center / max_distance
    
    # Separar canais (Assume formato HWC)
    # Se a imagem veio do OpenCV, a ordem é B, G, R
    b_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    r_channel = img[:, :, 2]

    # Aplicar as distorções
    # R: escala positiva (expande)
    r_distorted = distort_channel_manual(r_channel, radial_mask, center_x, center_y, 1.0 + intensity)
    # B: escala negativa (contrai)
    b_distorted = distort_channel_manual(b_channel, radial_mask, center_x, center_y, 1.0 - intensity)
    # G: permanece inalterado
    g_unchanged = g_channel

    # --- RECOMPOR ---
    # Stack dos canais de volta para o formato original
    result = np.stack([b_distorted, g_unchanged, r_distorted], axis=2).astype(np.uint8)

    # --- SUTILIZAÇÃO (Blend manual) ---
    # final = img * 0.3 + result * 0.7
    final_output = (img * 0.3 + result * 0.7).astype(np.uint8)

    return final_output

@register(prefix="243360")
def radial_blur(img: np.ndarray) -> np.ndarray:
    intensity = 0.10
    samples = 10
    
    if img is None:
        return None
    
    h, w, c = img.shape
    center_y, center_x = h / 2.0, w / 2.0
    
    # Imagem de acumulação em float32 para evitar estouro de bits (overflow)
    acc_img = np.zeros(img.shape, dtype=np.float32)

    # --- SIMULAÇÃO DE ZOOM RADIAL ---
    for i in range(samples):
        # Escala progressiva: 1.0 (original) até 1.0 + intensity
        scale = 1.0 + (i * intensity / (samples - 1))
        
        # No scipy.ndimage.affine_transform, a matriz define como mapear 
        # as coordenadas de SAÍDA para a ENTRADA. 
        # Para um "zoom in" (escala > 1), a matriz deve ser a inversa (1/scale).
        inv_scale = 1.0 / scale
        
        # Matriz de transformação afim para escala centrada:
        # 1. Transladar o centro para a origem (0,0)
        # 2. Escalar
        # 3. Transladar de volta para o centro original
        # A fórmula simplificada para o deslocamento (offset) é:
        offset = [
            center_y * (1 - inv_scale),
            center_x * (1 - inv_scale)
        ]
        
        # Aplicamos a transformação em cada canal de cor separadamente
        warped_sample = np.zeros_like(img, dtype=np.float32)
        for channel in range(c):
            warped_sample[:, :, channel] = affine_transform(
                img[:, :, channel],
                matrix=np.array([inv_scale, inv_scale]), # Escala nos eixos Y e X
                offset=offset,
                order=1,              # Interpolação bilinear (equivalente ao OpenCV)
                mode='nearest'        # Equivalente ao BORDER_REPLICATE
            )
        
        acc_img += warped_sample

    # Média das amostras
    blurred_layer = acc_img / samples

    # --- MÁSCARA RADIAL E BLENDING ---
    y, x = np.indices((h, w))
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Máscara: 1.0 no centro (nítido), 0.0 na borda (desfocado)
    # Expandimos a dimensão para permitir o broadcast com os 3 canais (H, W, 1)
    radial_mask = (1.0 - (dist_from_center / max_dist))[:, :, np.newaxis]
    
    # Clip para garantir que a máscara fique entre 0 e 1
    radial_mask = np.clip(radial_mask, 0, 1)

    # Mesclagem: Original * máscara + Blur * (1 - máscara)
    final_output_f = (img.astype(np.float32) * radial_mask) + (blurred_layer * (1.0 - radial_mask))

    return np.clip(final_output_f, 0, 255).astype(np.uint8)