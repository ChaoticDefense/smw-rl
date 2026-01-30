import argparse
import gymnasium as gym
import numpy as np
import pygame
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    DummyVecEnv
)
import stable_retro as retro

# --- YOUR CUSTOM HIJACK WRAPPERS ---

class SMWVisualAndRewardWrapper(gym.Wrapper):
    def __init__(self, env, render_logic=False):
        super().__init__(env)
        
        self.render_logic = render_logic
        self.prev_x = 0
        
        if self.render_logic:
            pygame.init()
            self.screen = pygame.display.set_mode((256, 224))
            pygame.display.set_caption("SMW Training Monitor")

    def reset(self, **kwargs):
        self.prev_x = 0
        return self.env.reset(**kwargs)

    def step(self, ac):
        ob, rew, terminated, truncated, info = self.env.step(ac)
        
        # 1. Custom Reward Hijack (X-Position)
        curr_x = info.get('x', 0)
        # We replace the default reward with progress
        custom_reward = float(curr_x - self.prev_x)
        self.prev_x = curr_x

        # 2. Render Hijack
        if self.render_logic:
            # Grab raw pixels directly from the emulator
            raw_frame = self.env.unwrapped.render()
            if raw_frame is not None:
                surf = pygame.surfarray.make_surface(np.transpose(raw_frame, (1, 0, 2)))
                self.screen.blit(surf, (0, 0))
                pygame.display.flip()
            pygame.event.pump()

        return ob, custom_reward, terminated, truncated, info

# --- THE STABLE RETRO BOILERPLATE ---

class StochasticFrameSkip(gym.Wrapper):
    # (Keeping their original class code here...)
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated, truncated, totrew = False, False, 0
        for i in range(self.n):
            if self.curac is None: self.curac = ac
            elif i == 0 and self.rng.rand() > self.stickprob: self.curac = ac
            elif i == 1: self.curac = ac
            
            ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated: break
        return ob, totrew, terminated, truncated, info

def make_retro(*, game, state=None, **kwargs):
    if state is None: state = retro.State.DEFAULT
    # Force rgb_array to allow our Pygame hijack
    env = retro.make(game, state, render_mode="rgb_array", **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SuperMarioWorld-Snes-v0")
    args = parser.parse_args()

    def make_env(rank):
        def _init():
            env = make_retro(game=args.game)
            # Only the first worker (rank 0) gets to render to Pygame
            # Otherwise, 12 windows would pop up and crash your script!
            env = SMWVisualAndRewardWrapper(env, render_logic=(True))
            env = WarpFrame(env) # This is the Grayscale/Resize 84x84
            return env
        return _init

    # We use 12 parallel environments to train 12x faster
    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env(i) for i in range(12)]), n_stack=4))

    model = PPO(
        policy="CnnPolicy",
        env=venv,
        device="cuda",
        learning_rate=lambda f: f * 2.5e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        verbose=1,
    )

    model.learn(total_timesteps=100_000_000)

if __name__ == "__main__":
    main()