import os

import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .diffuser import Diffuser
from .utils import get_random_image, preprocess_image


class NoiseGuesser:
    def __init__(self, path_to_images_dir: str, diffuser: Diffuser, image_size: int=64):
        self.path_to_images_dir = path_to_images_dir
        self.diffuser = diffuser
        self.image_size = image_size
        self.num_time_steps = diffuser.num_timesteps
        
    def play(self, path_to_image: str=None):
        
        # decide using random image or given image
        if path_to_image is None:
            image = get_random_image(self.path_to_images_dir)
        else:
            image = Image.open(path_to_image)
            
        # decide t using random value
        t = random.randint(1, self.num_time_steps+1)
        
        image = preprocess_image(image, self.image_size, False)
        
        # make_noisy_image(t, max)
        image_t, image_T, _= self.diffuser.add_noise(image, t, return_x_T=True)
        
        # show images
        self.show_image_0andT(image, image_t, image_T)
        
        # guess t
        while True:
            t_guessed = input("Guess t: ")
            if t_guessed.isdigit():
                t_guessed = int(t_guessed)
                break
            else:
                print("数字を入力してください。")
                continue
        
        # calculate difference
        diff = abs(t - t_guessed)
        diff_db = 20 * np.log10(diff)
        square_error = diff ** 2
        
        print(f"Answer: {t}, Diff: {diff}, Diff in dB: {diff_db:.3f}, Square error: {square_error}")
        
        while True:
            yn_retry = input("再挑戦しますか？(y/n): ")
        
            if yn_retry == "y":
                self.play(path_to_image)
                break
            elif yn_retry == "n":
                break
            else:
                print("y か n を入力してください。")
                continue
            
    def show_image_0andT(self, image_0, image_t, image_T):
        plt.subplot(1, 3, 1)
        plt.imshow(image_0)
        plt.title("Original Image")
        plt.axis("off") 
        
        plt.subplot(1, 3, 2)
        image_t = np.clip(image_t, 0, 1).astype(np.float32)
        plt.imshow(image_t)
        plt.title(f"step t Image")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        image_T = np.clip(image_T, 0, 1).astype(np.float32)
        plt.imshow(image_T)
        plt.title(f"step {self.num_time_steps} Image")
        plt.axis("off")
        
        plt.show()
        

