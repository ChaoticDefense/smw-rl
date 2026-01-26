import stable_retro as retro
import pygame
import numpy as np

def main():
    # Create the environment in headless mode
    env = retro.make(
        game="SuperMarioWorld-Snes-v0",
        render_mode="rgb_array"  # get frames as numpy arrays
    )

    obs, info = env.reset()

    # Initialize a pygame window
    pygame.init()
    height, width, _ = obs.shape
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("SMW - Wayland Friendly")

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Take a random action
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        # Reset if episode ended
        if terminated or truncated:
            obs, info = env.reset()

        # Convert the numpy frame to a pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(60)  # limit to 60 FPS

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
