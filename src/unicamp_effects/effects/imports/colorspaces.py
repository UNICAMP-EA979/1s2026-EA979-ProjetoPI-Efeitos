import numpy as np


def to_float(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.float32 or x.dtype == np.float64:
        return x
    return x.astype(np.float64) / 255.0

def to_uint8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    return np.round(255.0 * x).astype(np.uint8)


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
#Transforma RBG para OKLAB
#OKlab é um espaço perceptualmente uniforme. Facilita tarefas como conversão para grayscale e
#Ver a distância entre duas cores.
def rgb_to_oklab(rgb):
    """
    rgb: shape (..., 3), values in [0,1]
    returns: same shape (..., 3) in Oklab
    """

    rgb = np.asarray(rgb)

    # Step 1: sRGB -> linear RGB
    rgb_lin = srgb_to_linear(rgb)

    # Step 2: linear RGB -> LMS
    l = 0.4122214708 * rgb_lin[..., 0] + 0.5363325363 * rgb_lin[..., 1] + 0.0514459929 * rgb_lin[..., 2]
    m = 0.2119034982 * rgb_lin[..., 0] + 0.6806995451 * rgb_lin[..., 1] + 0.1073969566 * rgb_lin[..., 2]
    s = 0.0883024619 * rgb_lin[..., 0] + 0.2817188376 * rgb_lin[..., 1] + 0.6299787005 * rgb_lin[..., 2]

    # Step 3: cube root
    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)

    # Step 4: LMS -> Oklab
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return np.stack([L, a, b], axis=-1)

def linear_to_srgb(rgb):
    rgb = np.asarray(rgb)
    return np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * (rgb ** (1/2.4)) - 0.055
    )

def oklab_to_rgb(lab):
    """
    lab: shape (..., 3)
    returns: RGB in [0,1] (may slightly exceed → clamp if needed)
    """

    lab = np.asarray(lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # Step 1: Oklab -> LMS (nonlinear)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    # Step 2: cube
    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    # Step 3: LMS -> linear RGB
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    rgb_lin = np.stack([r, g, b], axis=-1)

    # Step 4: linear RGB -> sRGB
    rgb = linear_to_srgb(rgb_lin)

    return rgb

