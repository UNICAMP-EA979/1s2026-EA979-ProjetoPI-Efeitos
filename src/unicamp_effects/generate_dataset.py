import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image, ImageOps

from unicamp_effects.registry import get_all_effects, register
from unicamp_effects.utils import get_all_imgs_in_dir, get_path_ra, load_photo


@register()
def origin(img: np.ndarray) -> np.ndarray:
    return img


def save_img(img: np.ndarray, path: str) -> None:
    cv.imwrite(path, cv.cvtColor(img, cv.COLOR_RGB2BGR))


def process_func(path: str) -> None:
    effects = get_all_effects()
    photo = load_photo(path)
    photo_name = os.path.basename(path).split(".")[0]

    side = max(photo.shape[0], photo.shape[1])
    if side > 500:
        factor = 500/side
        photo = cv.resize(photo, None, fx=factor, fy=factor)

    for name, effect in effects.items():
        result = effect(photo)
        result_path = os.path.join("dataset", name, f"{photo_name}.jpg")
        save_img(result, result_path)


if __name__ == "__main__":
    effects = get_all_effects()

    for name in effects:
        os.makedirs(os.path.join("dataset", name), exist_ok=True)

    executor = ProcessPoolExecutor()
    paths = get_all_imgs_in_dir("photo")
    futures = []
    for path in paths:
        f = executor.submit(process_func, path)
        futures.append(f)

    wait(futures)

    fail = 0
    for f in futures:
        try:
            f.result()
        except Exception:
            fail += 1

    print("Failed", fail)
