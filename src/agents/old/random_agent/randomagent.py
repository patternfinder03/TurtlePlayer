# Random agent for TurtleTradingEnv

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import time
import random
import numpy as np
import gym_turtle_env
import gymnasium as gym
from config import override_params, render_wait_time, print_state




class RandomAgent:
    def __init__(self, env):
        self.env = env
        
        self.name = "RandomAgent"

    def get_action(self):
        valid_actions = self.env.unwrapped.controller.get_valid_actions()
        action = np.random.choice(valid_actions)
        return action

    def run_episode(self):
        state = self.env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not done and not truncated:
            action = self.get_action()
            next_state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            state = next_state
            
            if print_state:
                print(next_state)

        return total_reward, self.env.unwrapped.get_metrics()
    
    def run_episode_w_render(self):
        state, info = self.env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not done and not truncated:
            action = self.get_action()
            next_state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            state = next_state
            if print_state:
                print(next_state)
            
            time.sleep(render_wait_time)

        return total_reward, self.env.unwrapped.get_metrics()
