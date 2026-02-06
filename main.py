import stable_retro as retro

import json
import matplotlib.pyplot as plt
import datetime
import pytz
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from training import MarioNet, TrainAndLoggingCallback
from wrappers import make_env

SCRIPT_DIR = Path(__file__).parent

def inject_custom_data():
    custom_data_file = SCRIPT_DIR / 'custom_data.json'
    
    with open(custom_data_file, 'r') as f:
        custom_data = json.load(f)
    
    stable_retro_data_file = Path(retro.data.path()).joinpath('stable', 'SuperMarioWorld-Snes-v0', 'data.json')
    
    with open(stable_retro_data_file, 'w') as f:
        json.dump(custom_data, f, indent=2)
    
    return


def display_all_frame(state):
    plt.figure(figsize = (16, 16))
    
    for idx in range(state.shape[3]):
        plt.subplot(1, 4, idx + 1)
        plt.imshow(state[0][:,:,idx])
    
    plt.show()
     

def main():
    # Model Params
    CHECK_FREQ_NUM = 10000
    TIMESTEP_PER_ENV = 5_000_000
    LEARNING_RATE = 0.0001
    GAE = 1.0
    ENT_COEF = 0.05
    N_STEPS = 512
    GAMMA = 0.95
    BATCH_SIZE = 64
    N_EPOCHS = 10

    # Test Params
    
    # Overwrite data.json with custom_data.json
    # Enables tracking more info
    inject_custom_data()
    
    # env = DummyVecEnv([make_env])
    num_envs = 8
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    env = VecFrameStack(env, 4, channels_order='last')
    
    # env.reset()
    # state, reward, done, info = env.step([0])
    # print('state:', state.shape) #Color scale, height, width, num of stacks
     
    # display_all_frame(state)
     
    save_dir = Path('./model')
    save_dir.mkdir(parents = True, exist_ok = True)
    reward_log_path = save_dir / 'reward_log.csv'

    with open(reward_log_path, 'a') as f:
        print('timesteps,reward,best_reward', file=f)

    callback = TrainAndLoggingCallback(check_freq = CHECK_FREQ_NUM, save_path = save_dir, reward_log_path = reward_log_path, total_timesteps = TIMESTEP_PER_ENV, num_episodes = 1)
    
    policy_kwargs = dict(
    features_extractor_class = MarioNet,
    features_extractor_kwargs = dict(features_dim = 1024),
    )
    
    model = PPO('CnnPolicy',
                env, 
                verbose = 0, 
                policy_kwargs = policy_kwargs, 
                tensorboard_log = save_dir, 
                learning_rate = LEARNING_RATE, 
                n_steps = N_STEPS,
                batch_size = BATCH_SIZE, 
                n_epochs = N_EPOCHS, 
                gamma = GAMMA, 
                gae_lambda = GAE, 
                ent_coef = ENT_COEF, 
                device = 'cuda'
            )

    model.learn(total_timesteps = TIMESTEP_PER_ENV * num_envs, callback = callback, progress_bar = True)

if __name__ == "__main__":
    main()