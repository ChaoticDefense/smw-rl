import stable_retro as retro
import pygame
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class MarioDiscretizer(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # B, Y, SEL, STA, UP, DOWN, LEFT, RIGHT, A, X, L, R
        self._actions = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 0: Walk Right
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 1: Jump Right
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 2: Run Right
        ]
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        return np.array(self._actions[action]).astype(np.int8)

class SMWWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SMWWrapper, self).__init__(env)
        pygame.init()
        self.screen = pygame.display.set_mode((256, 224))
        self.prev_x = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Custom Reward Logic: Reward for moving right
        # 'x' is a variable defined in the game's data.json
        current_x = info.get('x', 0)
        reward = current_x - self.prev_x
        self.prev_x = current_x

        # 2. Death Penalty / Reset logic
        # If lives decrease, we force 'terminated' to True
        # if info.get('lives', 5) < 5: # Assuming starts at 5
        #     terminated = True

        self._render_pygame(obs)
        return obs, reward, terminated, truncated, info

    def _render_pygame(self, obs):
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
        pygame.event.pump() # Keeps window from freezing

def make_env():
    # Ensure you use a state file so it knows where to start
    env = retro.make(
        game="SuperMarioWorld-Snes-v0", 
        render_mode="rgb_array"
    )
    
    env = MarioDiscretizer(env)
    
    env = SMWWrapper(env)
    return env

def main():
    env = DummyVecEnv([make_env])
    
    # We use a lower learning rate for more "consistent" training
    model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.0001)

    print("Training... Mario should now reset on death.")
    model.learn(total_timesteps=100000)

if __name__ == "__main__":
    main()