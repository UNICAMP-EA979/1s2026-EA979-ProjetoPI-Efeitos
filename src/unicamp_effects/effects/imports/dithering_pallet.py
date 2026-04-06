import numpy as np
from . import colorspaces


def dithering_pallet(img_raw: np.ndarray, pallet_raw) -> np.ndarray:

    """
    :param img: imagem a ser dithering
    :param pallet: palheta de cores a qual a quantização usará
    :return: imagem com dithering
    """''
    img_raw = colorspaces.to_float(img_raw)
    img = colorspaces.rgb_to_oklab(img_raw)
    del img_raw
    print(np.max(img), np.min(img))

    pallet = []
    for i, color in enumerate(pallet_raw):
        color = colorspaces.to_float(np.array(color))
        color = colorspaces.rgb_to_oklab(color)
        pallet.append(color)
    print("pallet", pallet)
    vec_error = np.zeros_like(img[0,0,:])
    for i in range(img.shape[0]):
        #print(i, flush=True)
        for j in range(img.shape[1]):

            best_color = None
            best_k = None
            min_dif = float('inf')
            for k, color in enumerate(pallet):
                dif = np.linalg.norm(img[i, j, :] + vec_error - color)
                # assert (type(dif) == float or
                #         type(dif) == np.float64 or
                #         type(dif) == np.float32) ,f"{type(dif)}"
                if dif < min_dif:
                    min_dif = dif
                    best_color = color
                    best_k = k
            vec_error += img[i, j, :] - best_color
            img[i, j, :] = best_color
            re = 0.0
            if i+1 < img.shape[0] and j+1 < img.shape[1]:
                img[i+1, j+1, :] += 0.2 * vec_error
                img[i + 1, j, :] += 0.4 * vec_error
                img[i, j + 1, :] += 0.4 * vec_error
                vec_error -= vec_error
    img = colorspaces.oklab_to_rgb(img)
    print(np.max(img), np.min(img))
    img = np.clip(img, 0.0, 1.0)
    return colorspaces.to_uint8(img)


