import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


def get_all_imgs_in_dir(dir: str) -> list[str]:
    template = os.path.join(dir, "*.{format}")

    paths = []
    for format in ["png", "jpg", "jpeg"]:
        paths += glob.glob(template.format(format=format))

    return paths


def get_path_ra(path: str) -> str:
    return os.path.basename(path).split("_")[0]


def load_photo(path: str) -> np.ndarray:
    return np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
