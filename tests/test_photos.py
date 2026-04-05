import os

from utils import assert_prefix, assert_valid_ra, get_all_imgs_in_dir

import unicamp_effects


def test_photos():
    package_dir = os.path.dirname(unicamp_effects.__file__)
    photo_dir = os.path.join(package_dir, "..", "..", "photo")

    paths = get_all_imgs_in_dir(photo_dir)

    assert len(paths) >= 15
    prefix = os.path.basename(os.path.normpath(paths[0])).split("_")[0]
    for path in paths:
        filename = os.path.basename(os.path.normpath(path))
        assert filename.split("_")[0] == prefix

def test_photo_results():
    package_dir = os.path.dirname(unicamp_effects.__file__)
    result_dir = os.path.join(package_dir, "..", "..", "photo_result")

    paths = get_all_imgs_in_dir(result_dir)

    assert len(paths) >= 3

    ra = os.path.basename(paths[0]).split("_")[0]
    assert_valid_ra(ra)
    assert_prefix(paths, ra)
