import stable_retro as retro
import pygame
import numpy as np
import gymnasium as gym
import cv2
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class JoypadSpaceSNES(gym.ActionWrapper):
    """
    Simplifies SNES controller to a list of allowed button combinations.
    Precomputes binary vectors for efficiency.
    """
    
    BUTTON_MAPPING = {
        'B': 0,
        'Y': 1, 
        'SELECT': 2, 
        'START': 3,
        'UP': 4, 
        'DOWN': 5, 
        'LEFT': 6, 
        'RIGHT': 7,
        'A': 8, 
        'L': 9, 
        'R': 10
    }

    def __init__(self, env, combos):
        super().__init__(env)
        self.n_buttons = env.action_space.n  # MultiBinary(n)
        self.action_space = gym.spaces.Discrete(len(combos))
        
        # Precompute the full binary vectors
        self._actions = []
        for combo in combos:
            vector = np.zeros(self.n_buttons, dtype=np.int8)
            for button in combo:
                vector[self.BUTTON_MAPPING[button]] = 1
            self._actions.append(vector)

    def action(self, action):
        # Direct lookup, no loop at runtime
        return self._actions[action]

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
        
    def step(self, action):
        total_reward = 0.0
        terminated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return obs, reward, terminated, truncated, info
    
    
class ResizeEnv(gym.ObservationWrapper):
        def __init__(self, env, size):
            super().__init__(env)
            
            _, _, oldChannels = env.observation_space.shape
            newShape = (size, size, oldChannels)
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=newShape, dtype=np.uint8)
            
        def observation(self, frame):
            height, width, _ = self.observation_space.shape
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if frame.ndim == 2:
                frame = frame[:,:,None]
            return frame
        
class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env=None):
        super().__init__(env)


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
        
        ram = self.env.unwrapped.get_ram()
        
        if info.get('lives') < 4:
            terminated = True
        
        # 1. Custom Reward Logic: Reward for moving right
        # 'x' is a variable defined in the game's data.json
        current_x = info.get('x', 0)
        reward = current_x - self.prev_x
        self.prev_x = current_x

        self._render_pygame(obs)
        # print(terminated)
        
        
        return obs, reward, terminated, truncated, info

    def _render_pygame(self, obs):
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
        pygame.event.pump() # Keeps window from freezing

def make_env():
    env = retro.make(
        game="SuperMarioWorld-Snes-v0", 
        render_mode="rgb_array"
    )
    
    SIMPLE_MOVEMENT = [
    [], # NOOP
    ['RIGHT'], # Just right
    ['RIGHT', 'B'], # Right and jump
    ['B'], # Jump
    ['RIGHT', 'Y'], # Dash and right
    ['RIGHT', 'Y', 'B'], # Dash and jump right
    ['LEFT'] # Left
]
    
    env = JoypadSpaceSNES(env, SIMPLE_MOVEMENT)
    
    env = SMWWrapper(env)
    
    return env

def main():
#     # Overwrite data.json with custom_data.json
# +    retro.data.path('SuperMarioWorld-Snes-v0')
    
    
    
    env = DummyVecEnv([make_env])
    
    # We use a lower learning rate for more "consistent" training
    model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.0001)

    print("Training... Mario should now reset on death.")
    model.learn(total_timesteps=100000)

if __name__ == "__main__":
    main()