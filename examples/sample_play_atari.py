import pygame

import srl

# --- Atari env
# Run "pip install gymnasium pygame" and also see the URL below.
# https://gymnasium.farama.org/environments/atari/
env_config = srl.EnvConfig(
    "ALE/Galaxian-v5",
    kwargs=dict(full_action_space=True),
)

key_bind = {
    "": 0,
    "z": 1,
    pygame.K_UP: 2,
    pygame.K_RIGHT: 3,
    pygame.K_LEFT: 4,
    pygame.K_DOWN: 5,
    (pygame.K_UP, pygame.K_RIGHT): 6,
    (pygame.K_UP, pygame.K_LEFT): 7,
    (pygame.K_DOWN, pygame.K_RIGHT): 8,
    (pygame.K_DOWN, pygame.K_LEFT): 9,
    (pygame.K_UP, pygame.K_z): 10,
    (pygame.K_RIGHT, pygame.K_z): 11,
    (pygame.K_LEFT, pygame.K_z): 12,
    (pygame.K_DOWN, pygame.K_z): 13,
    (pygame.K_UP, pygame.K_RIGHT, pygame.K_z): 14,
    (pygame.K_UP, pygame.K_LEFT, pygame.K_z): 15,
    (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_z): 16,
    (pygame.K_DOWN, pygame.K_LEFT, pygame.K_z): 17,
}
runner = srl.Runner(env_config, rl_config=None)
runner.play_window(key_bind=key_bind)
