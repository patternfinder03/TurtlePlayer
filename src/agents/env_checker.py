# An environment checker to check if the environment is compatible with stable-baselines3
# Also a good robustness check



import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import gymnasium as gym
import gym_turtle_env
from stable_baselines3.common.env_checker import check_env


env = gym.make('TurtleTradingEnv', render_mode=None, override_params={})

check_env(env, warn=True, skip_render_check=True)

env = gym.make('TurtleTradingEnv', render_mode=None, override_params={'state_type': 'BasicWithPositions', 'reward_type': 'PerTrade'})

check_env(env)