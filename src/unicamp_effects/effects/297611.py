import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from unicamp_effects.registry import register


def _clip_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _rgb_to_gray_float(img: np.ndarray) -> np.ndarray:
    rgb = img.astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _gray_to_rgb_uint8(gray: np.ndarray) -> np.ndarray:
    gray_uint8 = _clip_uint8(gray)
    return np.repeat(gray_uint8[..., None], 3, axis=2)


def _posterize_gray(gray: np.ndarray, levels: int) -> np.ndarray:
    scaled = gray / 255.0
    quantized = np.floor(scaled * (levels - 1) + 0.5) / (levels - 1)
    return quantized * 255.0


def _convolve2d_same(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(channel, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    windows = sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)


def _dilate_mask(mask: np.ndarray, size: int) -> np.ndarray:
    pad = size // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=False)
    windows = sliding_window_view(padded, (size, size))
    return np.any(windows, axis=(-2, -1))


def _gaussian_blur(gray: np.ndarray) -> np.ndarray:
    gaussian_kernel = np.array(
        [[1, 4, 6, 4, 1],
         [4, 16, 24, 16, 4],
         [6, 24, 36, 24, 6],
         [4, 16, 24, 16, 4],
         [1, 4, 6, 4, 1]],
        dtype=np.float64,
    ) / 256.0
    return _convolve2d_same(gray, gaussian_kernel)


def _sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kernel_x = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=np.float64,
    )
    kernel_y = np.array(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]],
        dtype=np.float64,
    )
    grad_x = _convolve2d_same(gray, kernel_x)
    grad_y = _convolve2d_same(gray, kernel_y)
    return grad_x, grad_y


def _non_maximum_suppression(magnitude: np.ndarray, angle_rad: np.ndarray) -> np.ndarray:
    angle = (np.rad2deg(angle_rad) + 180.0) % 180.0

    mag_pad = np.pad(magnitude, 1, mode="constant")
    center = mag_pad[1:-1, 1:-1]
    left = mag_pad[1:-1, :-2]
    right = mag_pad[1:-1, 2:]
    up = mag_pad[:-2, 1:-1]
    down = mag_pad[2:, 1:-1]
    up_left = mag_pad[:-2, :-2]
    up_right = mag_pad[:-2, 2:]
    down_left = mag_pad[2:, :-2]
    down_right = mag_pad[2:, 2:]

    suppressed = np.zeros_like(magnitude)

    dir_0 = ((angle < 22.5) | (angle >= 157.5)) & (center >= left) & (center >= right)
    dir_45 = (angle >= 22.5) & (angle < 67.5) & (center >= up_right) & (center >= down_left)
    dir_90 = (angle >= 67.5) & (angle < 112.5) & (center >= up) & (center >= down)
    dir_135 = (angle >= 112.5) & (angle < 157.5) & (center >= up_left) & (center >= down_right)

    keep = dir_0 | dir_45 | dir_90 | dir_135
    suppressed[keep] = magnitude[keep]
    return suppressed


def _hysteresis_threshold(magnitude: np.ndarray, low_ratio: float, high_ratio: float) -> np.ndarray:
    high_threshold = magnitude.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    strong = magnitude >= high_threshold
    weak = (magnitude >= low_threshold) & ~strong
    strong_neighborhood = _dilate_mask(strong, size=9)
    return strong | (weak & strong_neighborhood)


def _rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    rgb = img.astype(np.float64) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    c_max = np.max(rgb, axis=2)
    c_min = np.min(rgb, axis=2)
    delta = c_max - c_min

    hue = np.zeros_like(c_max)
    non_zero = delta > 0

    r_mask = non_zero & (c_max == r)
    g_mask = non_zero & (c_max == g)
    b_mask = non_zero & (c_max == b)

    hue[r_mask] = np.mod((g[r_mask] - b[r_mask]) / delta[r_mask], 6.0)
    hue[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
    hue[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0
    hue = 60.0 * hue

    saturation = np.zeros_like(c_max)
    max_non_zero = c_max > 0
    saturation[max_non_zero] = delta[max_non_zero] / c_max[max_non_zero]

    value = c_max
    return np.stack((hue, saturation, value), axis=2)


def _ordered_dither_masked(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bayer = np.array(
        [[0, 8, 2, 10],
         [12, 4, 14, 6],
         [3, 11, 1, 9],
         [15, 7, 13, 5]],
        dtype=np.float64,
    ) / 16.0
    height, width = gray.shape
    tiled = np.tile(bayer, (height // 4 + 1, width // 4 + 1))[:height, :width]
    normalized = gray.astype(np.float64) / 255.0
    dithered = np.where(normalized >= tiled, 255.0, 0.0)
    return np.where(mask, dithered, gray.astype(np.float64))


def _bilinear_sample(img: np.ndarray, sample_y: np.ndarray, sample_x: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]

    y0 = np.floor(sample_y).astype(np.int32)
    x0 = np.floor(sample_x).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)

    y0 = np.clip(y0, 0, height - 1)
    x0 = np.clip(x0, 0, width - 1)

    wy = sample_y - y0
    wx = sample_x - x0

    top_left = img[y0, x0].astype(np.float64)
    top_right = img[y0, x1].astype(np.float64)
    bottom_left = img[y1, x0].astype(np.float64)
    bottom_right = img[y1, x1].astype(np.float64)

    top = top_left * (1.0 - wx)[..., None] + top_right * wx[..., None]
    bottom = bottom_left * (1.0 - wx)[..., None] + bottom_right * wx[..., None]
    return top * (1.0 - wy)[..., None] + bottom * wy[..., None]


@register(prefix="297611")
def canny_sketch(img: np.ndarray) -> np.ndarray:
    gray = _rgb_to_gray_float(img)
    blurred = _gaussian_blur(gray)
    grad_x, grad_y = _sobel_gradients(blurred)

    magnitude = np.hypot(grad_x, grad_y)
    suppressed = _non_maximum_suppression(magnitude, np.arctan2(grad_y, grad_x))
    edges = _hysteresis_threshold(suppressed, low_ratio=0.45, high_ratio=0.08)

    edge_scale = np.percentile(suppressed, 97)
    edge_scale = edge_scale if edge_scale > 1e-12 else 1.0
    line_strength = np.clip(suppressed / edge_scale, 0.0, 1.0)

    base = _posterize_gray(0.65 * blurred + 0.35 * gray, levels=6)
    sketch = base * (1.0 - 0.72 * line_strength)
    sketch = np.where(edges, sketch * 0.18, sketch)
    sketch = np.clip(sketch + 18.0, 0.0, 255.0)
    return _gray_to_rgb_uint8(sketch)


@register(prefix="297611")
def green_railing_dither(img: np.ndarray) -> np.ndarray:
    hsv = _rgb_to_hsv(img)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]

    green_mask = (hue >= 70.0) & (hue <= 170.0) & (sat >= 0.18) & (val >= 0.18)
    neutral_mask = (sat <= 0.35) & (val >= 0.16)
    color_mask = (~green_mask) & neutral_mask

    height, width = img.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float64)
    y_norm = yy / max(height - 1, 1)
    center_x = 0.505 * (width - 1)
    left_center = center_x - width * (0.004 + 0.176 * y_norm**1.36)
    right_center = center_x + width * (0.003 + 0.056 * y_norm**1.40)
    left_half_width = width * (0.003 + 0.070 * y_norm**2.35)
    right_half_width = width * (0.002 + 0.024 * y_norm**2.05)
    left_rail = np.abs(xx - left_center) <= left_half_width
    right_rail = np.abs(xx - right_center) <= right_half_width
    gray = _rgb_to_gray_float(img)
    geometric_mask = (yy >= 0.24 * height) & (left_rail | right_rail)
    candidate_mask = geometric_mask & color_mask
    railing_mask = _dilate_mask(geometric_mask, size=3)

    if np.any(railing_mask):
        masked_values = gray[candidate_mask] if np.any(candidate_mask) else gray[railing_mask]
        low = np.percentile(masked_values, 6)
        high = np.percentile(masked_values, 94)
        scale = max(high - low, 1.0)
        enhanced_gray = np.clip((gray - low) * (255.0 / scale), 0.0, 255.0)
    else:
        enhanced_gray = gray

    step = 4
    reduced_gray = enhanced_gray[::step, ::step]
    reduced_mask = railing_mask[::step, ::step]
    dithered_small = _ordered_dither_masked(reduced_gray, reduced_mask)
    dithered_gray = np.repeat(np.repeat(dithered_small, step, axis=0), step, axis=1)
    dithered_gray = dithered_gray[:height, :width]
    dithered_binary = np.where(dithered_gray >= 127.5, 255.0, 0.0)

    dithered_rgb = _gray_to_rgb_uint8(dithered_binary)

    background = _gray_to_rgb_uint8(0.88 * gray + 14.0)
    result = np.where(railing_mask[..., None], dithered_rgb, background)
    return result.astype(np.uint8)


@register(prefix="297611")
def radial_vignette(img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    center_y = 0.38 * (height - 1)
    center_x = 0.64 * (width - 1)

    yy, xx = np.indices((height, width), dtype=np.float64)
    x = xx - center_x
    y = yy - center_y

    radius = np.hypot(x, y)
    max_radius = radius.max() if radius.max() > 0 else 1.0
    radius_norm = radius / max_radius

    distorted_radius = 0.86 * radius_norm**1.7 + 0.14 * radius_norm
    scale = np.divide(
        distorted_radius,
        radius_norm,
        out=np.ones_like(distorted_radius),
        where=radius_norm > 1e-12,
    )

    sample_x = center_x + x * scale
    sample_y = center_y + y * scale
    valid = (
        (sample_x >= 0.0) & (sample_x <= width - 1.0) &
        (sample_y >= 0.0) & (sample_y <= height - 1.0)
    )

    distorted = _bilinear_sample(
        img,
        np.clip(sample_y, 0.0, height - 1.0),
        np.clip(sample_x, 0.0, width - 1.0),
    )
    distorted = np.where(valid[..., None], distorted, img.astype(np.float64))
    vignette = np.clip(1.0 - 0.16 * radius_norm**2, 0.84, 1.0)
    result = distorted * vignette[..., None]
    return _clip_uint8(result)
