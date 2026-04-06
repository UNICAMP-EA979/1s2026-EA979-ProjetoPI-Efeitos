import numpy as np

# Arquivo com o effeito difference of Gaussians
from . import anisotropic_filters, colorspaces


def difference_of_gaussians(img_raw: np.ndarray, sigma_c, sigma_e, sigma_m, sigma_a, p, phi, epsilon) -> np.ndarray:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be a quantile in the range [0, 1]")

    img = colorspaces.to_float(img_raw)
    img_color = colorspaces.rgb_to_oklab(img)
    img = img @ [0.2126, 0.7152, 0.0722]
    del img_raw
    # print("grayscale!")
    J11, J12, J22 = anisotropic_filters.structure_tensor(
        img, sigma_smooth=sigma_c)

    v1, v2, _1, _2 = anisotropic_filters.eigenvectors_from_structure_tensor(
        J11, J12, J22)
    result = np.zeros_like(img)

    G1 = anisotropic_filters.blur_across_edges(v1, v2, img, sigma_e)
    G2 = anisotropic_filters.blur_across_edges(v1, v2, img, 1.6 * sigma_e)

    Xdog = (1.0 + p) * G1 - p * G2
    epsilon_threshold = np.quantile(Xdog, epsilon)

    # Aplicação da função de ativação
    bright_mask = Xdog >= epsilon_threshold
    Xdog[bright_mask] = 1.0
    Xdog[~bright_mask] = 1 + \
        np.tanh(phi * (Xdog[~bright_mask] - epsilon_threshold))
    # print("XDog!", flush=True)
    Fdog = anisotropic_filters.lic_gaussian(v1, v2, Xdog, sigma=sigma_m)
    Axxdog = anisotropic_filters.lic_gaussian(v1, v2, Fdog, sigma=sigma_a)
    # result = colorspaces.oklab_to_rgb(result)
    result = Axxdog

    result = np.clip(result, 0.0, 1.0)
    result = result[..., None] * np.array([1.0, 1.0, 1.0])
    return colorspaces.to_uint8(result)
