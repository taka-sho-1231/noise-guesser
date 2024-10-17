import os
import random

import numpy as np
from PIL import Image

def get_random_image_path(path_to_images_dir: str) -> str:
    image_name_list = os.listdir(path_to_images_dir)
    image_name = image_name_list[random.randint(0, len(image_name_list) - 1)]
    path_to_random_image = os.path.join(path_to_images_dir, image_name)
    return path_to_random_image

def get_random_image(path_to_images_dir: str) -> Image:
    path_to_random_image = get_random_image_path(path_to_images_dir)
    image = Image.open(path_to_random_image)
    return image

def preprocess_image(image: Image, size: int, is_square: bool) -> np.ndarray:
    if is_square:
        image_resized = np.array(image.resize((size, size)))
        image_resized = image_resized / 255.0
        return image_resized
    if image.size[0] > image.size[1]:
        h = size
        w = int(size * image.size[0] / image.size[1])
    else:
        w = size
        h = int(size * image.size[1] / image.size[0])
    image_resized = np.array(image.resize((w, h)))
    image_resized = image_resized / 255.0
    return image_resized