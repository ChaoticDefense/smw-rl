import cv2
import gymnasium as gym
import numpy as np
import pygame
import stable_retro as retro
from gymnasium.wrappers import GrayscaleObservation

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
        return self._actions[action]

class RenderEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        pygame.init()
        self.screen = pygame.display.set_mode((256, 224))
        
    def observation(self, obs):
        self._render_pygame(obs)
        return obs
        
    def _render_pygame(self, obs):
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
        pygame.event.pump() # Keeps window from freezing

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        
        self._skip = skip
        
    def step(self, action):
        total_reward = 0.0
        terminated = False
        
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return obs, total_reward, terminated, truncated, info
    
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

class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env=None):
        super().__init__(env)
        
        self.start_x = None
        self.max_x = None
        self.prev_num_coins = 0
        self.prev_num_yoshi_coins = 0
        self.already_reached_checkpoint = False
        self.prev_powerup_state = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        self.start_x = None
        self.max_x = None
        self.prev_num_coins = 0
        self.prev_num_yoshi_coins = 0
        self.already_reached_checkpoint = False
        self.prev_powerup_state = 0
        
        return obs
    
    def step(self, action):
        obs, score_earned, terminated, truncated, info = self.env.step(action)
        reward = 0
        
        reward += score_earned * 0.1

        # Initialize starting x if not already
        if self.start_x is None:
            self.start_x = info.get('x')
            self.max_x = self.start_x

        # Reward progress in level (relative to max_x)
        current_x = info.get('x')
        delta_x = current_x - self.max_x
        if delta_x > 0:  # only reward forward progress
            reward += delta_x
            self.max_x = current_x
        
        # Reward new coins collected
        current_num_coins = info.get('coins')
        reward += max(0, current_num_coins - self.prev_num_coins) * 5
        self.prev_num_coins = current_num_coins
        
        # Reward new yoshi coins collected 
        current_num_yoshi_coins = info.get('yoshiCoins')
        reward += max(0, current_num_yoshi_coins - self.prev_num_yoshi_coins) * 20
        self.prev_num_yoshi_coins = current_num_yoshi_coins
        
        # Reward reaching the checkpoint
        if not self.already_reached_checkpoint and info.get('checkpoint'):
            reward += 100
            self.already_reached_checkpoint = True
            
        # Reward/punish based on powerup state
        current_powerup_state = info.get('powerups')
        if current_powerup_state != self.prev_powerup_state:
            # Small -> Big
            if self.prev_powerup_state == 0 and current_powerup_state == 1:
                reward += 50
            # Big -> Small
            elif self.prev_powerup_state == 1 and current_powerup_state == 0:
                reward -= 100
            # Big -> Cape/Fire
            elif self.prev_powerup_state == 1 and current_powerup_state in (2, 3):
                reward += 50
            # Cape -> Big
            elif self.prev_powerup_state == 2 and current_powerup_state == 1:
                reward -= 100
            # Cape -> Fire or Fire -> Cape
            elif (self.prev_powerup_state, current_powerup_state) in [(2, 3), (3, 2)]:
                reward += 50
            # Fire -> Big
            elif self.prev_powerup_state == 3 and current_powerup_state == 1:
                reward -= 100
            
            self.prev_powerup_state = current_powerup_state

        
        # Extremely reward reaching the goal
        if info.get('endOfLevel'):
            reward += 5000
            terminated = True
            print("GOAL")
        
        # Punish dying
        if info.get('dead') == 0:
            reward -= 500
            terminated = True
            
       
        return obs, reward, terminated, truncated, info
    
def make_env():
    
    SIMPLE_MOVEMENT = [
    [], # NOOP
    ['RIGHT'], # Just right
    ['RIGHT', 'B'], # Right and jump
    ['B'], # Jump
    ['RIGHT', 'Y'], # Dash and right
    ['RIGHT', 'Y', 'B'], # Dash and jump right
    ['LEFT'] # Left
    ]
    
    # env = retro.make(game=GAME_FILENAME, render_mode="rgb_array", state='YoshiIsland1')
    env = retro.make(game='SuperMarioWorld-Snes-v0', render_mode="rgb_array", state='DonutPlains1')

    env = JoypadSpaceSNES(env, SIMPLE_MOVEMENT)
    env = CustomRewardEnv(env)
    env = SkipFrame(env, skip=4)
    env = RenderEnv(env)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)

    return env