import sys
import enum

import random
import pygame
import numpy as np
from scripts.diffuser import Diffuser
from scripts.utils import preprocess_image, get_random_image

class GameState(enum.Enum):
    GUESSING = 0
    SHOWING_ANSWER = 1

PATH_TO_IMAGES_DIR = "images/"
SCREEN_SIZE = (800, 600)
NUM_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
IMAGE_SIZE = (64, 64)

def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Noise Guesser")

    diffuser = Diffuser(NUM_TIMESTEPS,
                        BETA_START, 
                        BETA_END)

    images, t = None, None

    game_state = GameState.GUESSING

    while True:
        screen.fill((0, 0, 0))

        # Get random images and t if not already set
        if images is None:
            image = get_random_image(PATH_TO_IMAGES_DIR)
            image = preprocess_image(image, IMAGE_SIZE[0], True)
            t = random.randint(1, NUM_TIMESTEPS + 1)
            image_t, image_T, _ = diffuser.add_noise(image, t, return_x_T=True)

            # Convert images to pygame surfaces
            images = [image, image_t, image_T]
            images = [(np.clip(image, 0, 1) * 255).astype(np.uint8) for image in images]
            images = [pygame.surfarray.make_surface(image) for image in images]
            labels = ["Step 0", "Step t", f"Step {NUM_TIMESTEPS}"]
            images = {label: image for label, image in zip(labels, images)}

        # Display text and images
        guess_t_text = pygame.font.Font(None, 36).render("Guess t!", True, (255, 255, 255))
        screen.blit(guess_t_text, (50, 50))
        for i, (label, image) in enumerate(images.items()):
            image = pygame.transform.scale(image, (200, 200))
            screen.blit(image, (25 + i * 250, 150))
            label_text = pygame.font.Font(None, 36).render(label, True, (255, 255, 255))
            screen.blit(label_text, (50 + i * 250, 370))

        match game_state:
            case GameState.GUESSING:
                answer_button = pygame.Rect(50, 450, 150, 50)
                answer_text = pygame.font.Font(None, 36).render("Answer", True, (255, 255, 255))
                pygame.draw.rect(screen, (100, 100, 100), answer_button)
                screen.blit(answer_text, (70, 460))
                pass

            case GameState.SHOWING_ANSWER:
                answer_text = pygame.font.Font(None, 36).render(f"Answer: {t}", True, (255, 255, 255))
                screen.blit(answer_text, (50, 450))

                retry_button = pygame.Rect(50, 500, 150, 50)
                retry_text = pygame.font.Font(None, 36).render("Retry", True, (255, 255, 255))
                pygame.draw.rect(screen, (100, 100, 100), retry_button)
                screen.blit(retry_text, (70, 510))

                quit_button = pygame.Rect(250, 500, 150, 50)
                quit_text = pygame.font.Font(None, 36).render("Quit", True, (255, 255, 255))
                pygame.draw.rect(screen, (100, 100, 100), quit_button)
                screen.blit(quit_text, (270, 510))
                pass

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if game_state == GameState.GUESSING:
                    if answer_button.collidepoint(x, y):
                        game_state = GameState.SHOWING_ANSWER
                elif game_state == GameState.SHOWING_ANSWER:
                    if retry_button.collidepoint(x, y):
                        images = None
                        game_state = GameState.GUESSING
                    elif quit_button.collidepoint(x, y):
                        pygame.quit()
                        sys.exit()

if __name__ == "__main__":
    main()