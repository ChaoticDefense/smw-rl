import os
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    VecTransposeImage,
    DummyVecEnv
)

import torch
from torch import nn

from wrappers import make_env

class MarioNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    # def forward(self, observations: torch.Tensor) -> torch.Tensor:
    #     return self.linear(self.cnn(observations))
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Convert NumPy to float tensor if needed
        if isinstance(observations, np.ndarray):
            x = torch.as_tensor(observations, device=self.linear[0].weight.device).float()
        else:
            x = observations.to(self.linear[0].weight.device).float()
            
        # Normalize pixel values
        x /= 255.0
        
        # Make sure channels-first
        if x.shape[-1] == 4:  # your frame stack
            x = x.permute(0, 3, 1, 2)
            
        return self.linear(self.cnn(x))
    
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, reward_log_path, num_episodes, total_timesteps, verbose=1):
        super().__init__(verbose)
        
        self.check_freq = check_freq
        self.save_path = save_path
        self.reward_log_path = reward_log_path
        self.num_episodes = num_episodes
        self.total_timesteps = total_timesteps
        
        self.eval_env = DummyVecEnv([make_env(0, True)])
        self.eval_env = VecFrameStack(self.eval_env, 4, channels_order='last')
        self.eval_env = VecTransposeImage(self.eval_env)

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = self.save_path / f'best_model_{self.n_calls}'
            self.model.save(model_path)

            total_reward = [0] * self.num_episodes
            total_time = [0] * self.num_episodes
            best_reward = 0

            for i in range(self.num_episodes):
                state = self.eval_env.reset()  # reset for each new trial
                done = False
                total_reward[i] = 0
                total_time[i] = 0
                
                while not done:
                    action, _ = self.model.predict(state, deterministic=True)
                    state, reward, done, info = self.eval_env.step(action)
                    total_reward[i] += reward[0]
                    total_time[i] += 1

                if total_reward[i] > best_reward:
                    best_reward = total_reward[i]

                state = self.eval_env.reset()  # reset for each new trial

            print('time steps:', self.n_calls, '/', self.total_timesteps)
            print('average reward:', (sum(total_reward) / self.num_episodes),
                  'average time:', (sum(total_time) / self.num_episodes),
                  'best_reward:', best_reward)

            with open(self.reward_log_path, 'a') as f:
                print(self.n_calls, ',', sum(total_reward) / self.num_episodes, ',', best_reward, file=f)

        return True
    
    