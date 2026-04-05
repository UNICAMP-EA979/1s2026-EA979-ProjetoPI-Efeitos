import numpy as np
from scipy import ndimage as sndi

from unicamp_effects.registry import register

@register(247145)
def twirl(img: np.ndarray) -> np.ndarray:
    strength = 3*np.pi/2

    H, W = img.shape[:2]
    cy, cx = H / 2.0, W / 2.0 # centre coordinates
    max_r = np.sqrt(cx**2 + cy**2) # diagonal length
    y, x = np.ogrid[:H, :W]
    dy = y - cy
    dx = x - cx
    r = np.sqrt(dx**2 + dy**2) # radius from centre (H, W)

    theta = np.arctan2(dy, dx)     # original angle
    twist = strength * (r / max_r) 
    theta_new = theta + twist      # twisted angle

    # Source coordinates for the reverse mapping
    src_x = cx + r * np.cos(theta_new)
    src_y = cy + r * np.sin(theta_new)

    # Round to nearest‑neighbour
    src_x = np.round(src_x).astype(np.int32)
    src_y = np.round(src_y).astype(np.int32)

    # Clip to valid range
    src_x = np.clip(src_x, 0, W - 1)
    src_y = np.clip(src_y, 0, H - 1)

    twisted = img[src_y, src_x]
    return twisted

@register(247145)
def color_selection(img: np.ndarray) -> np.ndarray:
    target_rgb = (255, 0, 0)
    threshold = 140

    # Squared Euclidean distance between original and target color
    diff = img - np.array(target_rgb, dtype=np.float32)
    dist_sq = np.sum(diff**2, axis=2)

    # Pixels with color whithin an acceptable distance of the target
    keep_mask = dist_sq <= threshold**2

    # Convert entire image to grayscale
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    gray = gray.astype(np.uint8)       # 0-255

    result = np.broadcast_to(gray[..., np.newaxis], img.shape).copy()

    # Restore original colours where keep is true
    result[keep_mask] = img[keep_mask]
    return result

@register(247145)
def border_detection(img: np.ndarray) -> np.ndarray:
    threshold = 20

    # Convert intire image to grayscale
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    gray = gray.astype(np.float32)

    # Sobel filters
    sobel_v = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    sobel_h = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    borders_v = sndi.convolve(gray, sobel_v, cval=0, mode='constant')
    borders_h = sndi.convolve(gray, sobel_h, cval=0, mode='constant')

    # Combine borders with Sobel magnitude
    magnitude = np.sqrt(borders_v**2 + borders_h**2)

    # Normalise to 0-255
    max_val = magnitude.max()
    if max_val > 0:
        magnitude = (magnitude / max_val) * 255
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    # Threshold to binary edge map
    edge_mask = magnitude > threshold

    # Make a complete black image
    edges = np.zeros_like(img)

    # Paint white pixels where we have edges
    edges[edge_mask] = [255, 255, 255]
    return edges
