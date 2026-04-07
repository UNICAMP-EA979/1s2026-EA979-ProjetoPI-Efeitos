import argparse
import os

import numpy as np
from PIL import Image, ImageOps

from unicamp_effects.registry import get_effect


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_image",
                        help="Path to input image",
                        type=str)

    parser.add_argument("effect_name",
                        help="Name of the effect",
                        type=str)

    parser.add_argument("--output_path",
                        help="Path to folder to save result. Defaults is 'photo_result'",
                        type=str,
                        default="photo_result")

    args = parser.parse_args()

    input_image: str = args.input_image
    effect_name: str = args.effect_name
    output_path: str = args.output_path

    img_pil = ImageOps.exif_transpose(Image.open(input_image)).convert("RGB")
    img = np.array(img_pil)

    effect = get_effect(effect_name)

    result = effect(img)

    img_name = os.path.basename(input_image)
    result_path = os.path.join(output_path, img_name)

    result_pil = Image.fromarray(result)
    result_pil.save(result_path)


if __name__ == "__main__":
    main()
