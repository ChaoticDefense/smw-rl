import stable_retro as retro
import pygame
import numpy as np
import gymnasium as gym
import cv2
import json
import matplotlib.pyplot as plt
import datetime
import os
from pathlib import Path
from pytz import timezone
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    DummyVecEnv
)

import torch as th
from torch import nn

# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

SCRIPT_DIR = Path(__file__).parent
GAME_FILENAME = 'SuperMarioWorld-Snes-v0'



# Model Param
CHECK_FREQ_NUMB = 10000
TOTAL_TIMESTEP_NUMB = 5000000
LEARNING_RATE = 0.0001
GAE = 1.0
ENT_COEF = 0.01
N_STEPS = 512
GAMMA = 0.9
BATCH_SIZE = 64
N_EPOCHS = 10

# Test Param
EPISODE_NUMBERS = 20
MAX_TIMESTEP_TEST = 1000
     


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

class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env=None):
        super().__init__(env)
        
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
    
    def reset(self, **kwargs):
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        reward += max(0, info['x'] - self.max_x)
        
        if (info['x'] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0
        
        if info["endOfLevel"]:
            reward += 500
            terminated = True
            print("GOAL")
        
        if info["dead"] == 0:
            reward -= 500
            terminated = True
        
        self.current_score = info["score"]
        self.max_x = max(self.max_x, self.current_x)
        self.current_x = info["x"]
        
        return obs, reward, terminated, truncated, info



class MarioNet(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioNet, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, reward_log_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.reward_log_path = reward_log_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = (self.save_path / 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

            total_reward = [0] * EPISODE_NUMBERS
            total_time = [0] * EPISODE_NUMBERS
            best_reward = 0

            for i in range(EPISODE_NUMBERS):
                state = self.env.reset()  # reset for each new trial
                done = False
                total_reward[i] = 0
                total_time[i] = 0
                while not done and total_time[i] < MAX_TIMESTEP_TEST:
                    action, _ = self.model.predict(state)
                    state, reward, done, info = self.env.step(action)
                    total_reward[i] += reward[0]
                    total_time[i] += 1

                if total_reward[i] > best_reward:
                    best_reward = total_reward[i]
                    best_epoch = self.n_calls

                state = self.env.reset()  # reset for each new trial

            print('time steps:', self.n_calls, '/', TOTAL_TIMESTEP_NUMB)
            print('average reward:', (sum(total_reward) / EPISODE_NUMBERS),
                  'average time:', (sum(total_time) / EPISODE_NUMBERS),
                  'best_reward:', best_reward)

            with open(self.reward_log_path, 'a') as f:
                print(self.n_calls, ',', sum(total_reward) / EPISODE_NUMBERS, ',', best_reward, file=f)

        return True

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
    
    env = retro.make(game=GAME_FILENAME, render_mode="rgb_array")

    env = JoypadSpaceSNES(env, SIMPLE_MOVEMENT)
    env = CustomRewardEnv(env)
    env = SkipFrame(env, skip=4)
    env = RenderEnv(env)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)

        
    return env


def inject_custom_data():
    custom_data_file = SCRIPT_DIR / 'custom_data.json'
    
    with open(custom_data_file, 'r') as f:
        custom_data = json.load(f)
    
    stable_retro_data_file = Path(retro.data.path()).joinpath('stable', GAME_FILENAME, 'data.json')
    
    with open(stable_retro_data_file, 'w') as f:
        json.dump(custom_data, f, indent=2)
    
    return



def display_all_frame(state):
    plt.figure(figsize=(16,16))
    for idx in range(state.shape[3]):
        plt.subplot(1,4,idx+1)
        plt.imshow(state[0][:,:,idx])
    plt.show()
     


def main():
    # Overwrite data.json with custom_data.json
    # Enables tracking more info
    inject_custom_data()
    
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, 4, channels_order='last')
    
    # env.reset()
    # state, reward, done, info = env.step([0])
    # print('state:', state.shape) #Color scale, height, width, num of stacks
     
    # display_all_frame(state)
    
    policy_kwargs = dict(
        features_extractor_class=MarioNet,
        features_extractor_kwargs=dict(features_dim=512),
    )
     

    save_dir = Path('./model')
    save_dir.mkdir(parents=True, exist_ok=True)
    reward_log_path = (save_dir / 'reward_log.csv')
        

    with open(reward_log_path, 'a') as f:
        print('timesteps,reward,best_reward', file=f)
     

    callback = TrainAndLoggingCallback(check_freq=CHECK_FREQ_NUMB, save_path=save_dir, reward_log_path=reward_log_path)
     

    model = PPO('CnnPolicy', env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=save_dir, learning_rate=LEARNING_RATE, n_steps=N_STEPS,
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=GAE, ent_coef=ENT_COEF)
        

    model.learn(total_timesteps=TOTAL_TIMESTEP_NUMB, callback=callback)
        

    
    
    # # We use a lower learning rate for more "consistent" training
    # model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.0001)

    # model.learn(total_timesteps=100000)

if __name__ == "__main__":
    main()