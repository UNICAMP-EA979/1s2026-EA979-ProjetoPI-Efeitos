import os

import cv2 as cv
import numpy as np
import tqdm

from .utils import get_all_imgs_in_dir, get_path_ra, load_photo

os.makedirs("comparison", exist_ok=True)

photos = get_all_imgs_in_dir("photo")
photo_results = get_all_imgs_in_dir("photo_result")

ras = set()

for path in photo_results:
    ra = get_path_ra(path)
    ras.add(ra)


for ra in tqdm.tqdm(ras):
    ra_results = []

    for path in photo_results:
        path_ra = get_path_ra(path)
        if path_ra == ra:
            ra_results.append(path)

    photo_map = []
    for path in ra_results:
        origin_path = os.path.join("photo", os.path.basename(path))

        origin_photo = load_photo(origin_path)
        photo = load_photo(path)

        if photo.shape[0] == origin_photo.shape[1]:
            photo = np.rot90(photo, -1)

        photo_map.append((origin_photo, photo))

    max_width_origin = 0
    max_width_result = 0
    for pair in photo_map:
        max_width_origin = max(max_width_origin, pair[0].shape[1])
        max_width_result = max(max_width_result, pair[1].shape[1])

    views = []
    for pair in photo_map:
        max_height = max(pair[0].shape[0], pair[1].shape[0])
        view = np.full((max_height, max_width_origin +
                       max_width_result, 3), (255, 0, 255), np.uint8)

        view[:pair[0].shape[0], :pair[0].shape[1], :] = pair[0]
        view[:pair[1].shape[0], pair[0].shape[1]:pair[0].shape[1]+pair[1].shape[1], :] = pair[1]

        views.append(view)

    compare_view = np.vstack(views)

    cv.imwrite(os.path.join("comparison", f"{ra}.jpg"),
               cv.cvtColor(compare_view, cv.COLOR_RGB2BGR))
