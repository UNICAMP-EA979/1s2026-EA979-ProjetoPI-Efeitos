import numpy as np
from scipy.ndimage import gaussian_filter, convolve


def bilinear_sample(img, x, y):
    """Sample feito a GPU"""
    h, w = img.shape

    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    dx = x - x0
    dy = y - y0

    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]

    return (
        v00 * (1.0 - dx) * (1.0 - dy)
        + v10 * dx * (1.0 - dy)
        + v01 * (1.0 - dx) * dy
        + v11 * dx * dy
    )


def bilinear_sample_grid(img, x, y):
    """Vectorized bilinear sampling for array coordinates."""
    h, w = img.shape

    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)

    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    dx = x - x0
    dy = y - y0

    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]

    return (
        v00 * (1.0 - dx) * (1.0 - dy)
        + v10 * dx * (1.0 - dy)
        + v01 * (1.0 - dx) * dy
        + v11 * dx * dy
    )


#Feito com o auxílio de IA generativa
def structure_tensor(img, sigma_smooth=2.0):


    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)

    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=float)

    f = img.astype(float)
    Ix = convolve(f, Kx)   # horizontal gradient
    Iy = convolve(f, Ky)   # vertical gradient


    J11 = gaussian_filter(Ix * Ix, sigma=sigma_smooth)
    J12 = gaussian_filter(Ix * Iy, sigma=sigma_smooth)
    J22 = gaussian_filter(Iy * Iy, sigma=sigma_smooth)

    return J11, J12, J22


def eigenvectors_from_structure_tensor(J11, J12, J22):

    diff = J11 - J22
    disc = np.sqrt(np.maximum(diff ** 2 + 4.0 * J12 ** 2, 0.0))

    lam_major = (J11 + J22 + disc) * 0.5   # larger eigenvalue  – gradient dir
    lam_minor = (J11 + J22 - disc) * 0.5   # smaller eigenvalue – tangent dir

    vx_minor = J12.copy()
    vy_minor = lam_minor - J11
    norm_minor = np.hypot(vx_minor, vy_minor) + 1e-8

    vx_major = J12.copy()
    vy_major = lam_major - J11
    norm_major = np.hypot(vx_major, vy_major) + 1e-8

    return (
        vx_minor / norm_minor,
        vy_minor / norm_minor,
        vx_major / norm_major,
        vy_major / norm_major,
    )


#Feito com o auxílio de IA generativa
def lic_gaussian(vx, vy, noise, steps=10, step_size=1.0, sigma=5.0):
    """
    vx:    (H, W)
    vy:    (H, W)
    noise: (H, W)

    returns:
    output: (H, W)
    """

    h, w = noise.shape
    yy, xx = np.meshgrid(
        np.arange(h, dtype=float) + 0.5,
        np.arange(w, dtype=float) + 0.5,
        indexing="ij",
    )

    acc = np.zeros((h, w), dtype=float)
    weight_sum = np.zeros((h, w), dtype=float)

    for direction in (-1.0, 1.0):
        px = xx.copy()
        py = yy.copy()
        active = np.ones((h, w), dtype=bool)

        for i in range(steps):
            in_bounds = (px >= 0.0) & (px <= (w - 1.0)) & (py >= 0.0) & (py <= (h - 1.0))
            active &= in_bounds
            if not np.any(active):
                break

            # distance along streamline
            s = i * step_size
            # Gaussian weight
            wi = np.exp(-(s * s) / (2.0 * sigma * sigma))

            val = bilinear_sample_grid(noise, px, py)
            mask = active.astype(float)
            acc += val * wi * mask
            weight_sum += wi * mask

            # vector field sampled continuously along streamline
            vx_val = bilinear_sample_grid(vx, px, py)
            vy_val = bilinear_sample_grid(vy, px, py)
            norm = np.hypot(vx_val, vy_val) + 1e-8

            dx = vx_val / norm
            dy = vy_val / norm

            px = np.where(active, px + direction * step_size * dx, px)
            py = np.where(active, py + direction * step_size * dy, py)

    output = np.zeros((h, w), dtype=float)
    valid = weight_sum > 0.0
    output[valid] = acc[valid] / weight_sum[valid]
    return output


def gaussian(x, sigma):
    exponent = -0.5 * (x / sigma)**2
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(exponent)


def blur_across_edges(vx, vy, img, sigma_e):
    if sigma_e <= 0:
        raise ValueError("sigma_e must be > 0")

    h, w = img.shape
    yy, xx = np.meshgrid(
        np.arange(h, dtype=float) + 0.5,
        np.arange(w, dtype=float) + 0.5,
        indexing="ij",
    )

    # Vector perpendicular to the edge-flow field
    p_vx = -vy
    p_vy = vx

    w0 = gaussian(0, sigma=sigma_e)
    acc = w0 * bilinear_sample_grid(img, xx, yy)
    weight_sum = w0

    for direction in (-1.0, 1.0):
        for i in range(1, 5):
            wi = gaussian(i, sigma=sigma_e)
            sx = xx + direction * i * p_vx
            sy = yy + direction * i * p_vy
            acc += wi * bilinear_sample_grid(img, sx, sy)
            weight_sum += wi

    return acc / weight_sum


