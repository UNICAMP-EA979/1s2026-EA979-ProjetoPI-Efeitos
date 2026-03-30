import numpy as np
import cv2


from unicamp_effects.registry import register


@register(prefix="243360")
def edge_detection(img: np.ndarray) -> np.ndarray:
    # Carregar a imagem e Conversão para Tons de Cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Redução de Ruído (Filtro Gaussiano)
    # ksize=(5, 5) é comum, mas para rachaduras finas, (3, 3) pode preservar mais detalhes
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Algoritmo de Canny
    # O OpenCV agrupa o cálculo do Gradiente, Supressão e Histérese em uma função.
    # threshold1: Limiar inferior (Hysteresis)
    # threshold2: Limiar superior (Bordas fortes)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    return edges

# --- SEPARAÇÃO E DISTORÇÃO GEOMÉTRICA ---
def distort_channel(w, h, radial_mask, center_x, center_y, channel, scale_factor):
    # Criamos o mapa de coordenadas original
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        
    # O deslocamento é proporcional à máscara (0 no centro, máx na borda)
    # scale_factor > 1 expande (Vermelho), < 1 contrai (Azul)
    offset_factor = 1 + (scale_factor - 1) * radial_mask
        
    # Aplicamos a distorção radial a partir do centro
    new_map_x = ((map_x - center_x) * offset_factor + center_x).astype(np.float32)
    new_map_y = ((map_y - center_y) * offset_factor + center_y).astype(np.float32)
        
    # Remapeia o canal individualmente
    return cv2.remap(channel, new_map_x, new_map_y, cv2.INTER_LINEAR)

@register(prefix="243360")
def chromatic_aberration(img: np.ndarray) -> np.ndarray:
    intensity = 0.005
    if img is None: return print("Erro ao carregar imagem.")
    
    h, w = img.shape[:2]
    center_x, center_y = w / 2, h / 2

    # MÁSCARA DE INTENSIDADE RADIAL
    # Criamos uma matriz de distâncias do centro para servir de mapa de influência
    y, x = np.indices((h, w))
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    # Normaliza a máscara (0 no centro, 1 nas bordas)
    radial_mask = distance_from_center / max_distance
    
    # Separar canais (BGR no OpenCV)
    b_channel, g_channel, r_channel = cv2.split(img)

    # Aplicar as distorções solicitadas
    # Vermelho: escala positiva (expande)
    r_distorted = distort_channel(w, h, radial_mask, center_x, center_y, r_channel, 1.0 + intensity)
    # Azul: escala negativa (contrai)
    b_distorted = distort_channel(w, h, radial_mask, center_x, center_y, b_channel, 1.0 - intensity)
    # Verde: permanece inalterado
    g_unchanged = g_channel

    # --- RECOMPOR ---
    result = cv2.merge([b_distorted, g_unchanged, r_distorted])

    # --- SUTILIZAÇÃO (OPCIONAL) ---
    # Para sutilizar, podemos fazer um blend com a imagem original
    final_output = cv2.addWeighted(img, 0.3, result, 0.7, 0)

    return final_output

@register(prefix="243360")
def radial_blur(img: np.ndarray) -> np.ndarray:
    intensity=0.10
    samples=10
    # --- CARREGAR E ISOLAR ---
    if img is None: return print("Erro ao carregar imagem.")
    
    h, w = img.shape[:2]
    center = (w / 2, h / 2) # Ponto central onde as linhas convergem

    # Usaremos uma imagem em float32 para acumular as amostras com precisão
    acc_img = np.zeros_like(img, dtype=np.float32)

    # --- AMOSTRAGEM RADIAL (USANDO TRANSFORMAÇÃO AFIM ITERATIVA) ---
    # Em vez de calcular pixel por pixel, aplicamos pequenas escalas e 
    # acumulamos o resultado. Isso simula a coleta de amostras ao longo da linha radial.
    
    for i in range(samples):
        # A escala é progressiva: 1.0 (original) até 1.0 + intensity
        scale = 1.0 + (i * intensity / (samples - 1))
        
        # Cria a matriz de transformação para escala centrada
        M = cv2.getRotationMatrix2D(center, 0, scale)
        
        # Aplica a transformação de escala
        warped_sample = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Acumula a amostra na imagem de ponto flutuante
        acc_img += warped_sample.astype(np.float32)

    # Calcula a média das amostras
    blurred_layer = (acc_img / samples).astype(np.uint8)

    # --- INTENSIDADE RADIAL, RECOMPOSIÇÃO E AJUSTE SUTIL ---
    # Criamos uma máscara radial para garantir o centro nítido e bordas desfocadas
    y, x = np.indices((h, w))
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    max_dist = np.sqrt(center[0]**2 + center[1]**2)
    
    # Máscara radial: 1.0 no centro (original), 0.0 nas bordas (desfocado)
    radial_mask = 1.0 - (dist_from_center / max_dist)
    
    # Garante que a máscara tem 3 canais para o blend de cores
    radial_mask_3ch = cv2.merge([radial_mask, radial_mask, radial_mask])
    
    # Converte para float para multiplicação
    img_f = img.astype(np.float32)
    blurred_f = blurred_layer.astype(np.float32)

    # Mesclagem (Blending):
    # Resultado = (Imagem Original * Peso do Centro) + (Camada Desfocada * Peso da Borda)
    final_output_f = (img_f * radial_mask_3ch) + (blurred_f * (1.0 - radial_mask_3ch))

    # Converte de volta para uint8
    final_output = final_output_f.astype(np.uint8)

    return final_output
