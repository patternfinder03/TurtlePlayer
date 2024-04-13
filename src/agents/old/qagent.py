# Q-Learning Agent

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import time
import random
import argparse
import numpy as np
import gym_turtle_env
import gymnasium as gym
from agents.run_multiple_stocks import run_all_stocks
from agents.run_single_stock import train_and_evaluate_agent_single_stock
from config import override_params, render_wait_time, print_state, print_rewards, train_episodes, discount_factor, exploration_decay, learning_rate


class QLearningAgent:
    def __init__(self, env, learning_rate=learning_rate, discount_factor=discount_factor, exploration_rate=1, exploration_decay=exploration_decay):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}
        
        self.name = 'Q-Learning Agent'

    # def get_action(self, state):
    #     state_key = self.state_to_key(state)
    #     valid_actions = self.env.unwrapped.controller.get_valid_actions()
    #     self.initialize_state_in_q_table(state_key)

    #     if random.uniform(0, 1) < self.exploration_rate:
    #         return random.choice(valid_actions)  # Explore
    #     else:
    #         # Exploit by selecting the best valid action based on Q-values
    #         q_values = np.array([self.q_table[state_key][a] for a in range(self.env.action_space.n)])
    #         # Mask q_values of invalid actions with a very low number
    #         mask = np.ones(len(q_values)) * -np.inf
    #         mask[valid_actions] = 0
    #         q_values_masked = q_values + mask
    #         return np.argmax(q_values_masked)  # Exploit
    def get_action(self, state):
        state_key = self.state_to_key(state)
        valid_actions = self.env.unwrapped.controller.get_valid_actions()
        self.initialize_state_in_q_table(state_key)

        # Check if the state_key exists in the Q-table
        if state_key in self.q_table and random.uniform(0, 1) >= self.exploration_rate:
            # Exploit by selecting the best valid action based on Q-values
            q_values = np.array([self.q_table[state_key][a] for a in range(self.env.action_space.n)])
            # Mask q_values of invalid actions with a very low number
            mask = np.ones(len(q_values)) * -np.inf
            mask[valid_actions] = 0
            q_values_masked = q_values + mask
            # print(f"Exploited")
            return np.argmax(q_values_masked)  # Exploit
        else:
            # print("Explored")
            return random.choice(valid_actions)  # Explore

        
    def update_q_table(self, state, action, reward, next_state):
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)

        # Ensure both the current and next states are in the Q-table
        self.initialize_state_in_q_table(state_key)
        self.initialize_state_in_q_table(next_state_key)

        # Find the best next action from Q-table
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.discount_factor * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_error

    # Inverse discount factor
    # def update_q_table(self, state, action, reward, next_state):
    #     state_key = self.state_to_key(state)
    #     next_state_key = self.state_to_key(next_state)

    #     self.initialize_state_in_q_table(state_key)
    #     self.initialize_state_in_q_table(next_state_key)

    #     # Find the best next action from Q-table
    #     best_next_action = np.argmax(self.q_table[next_state_key])

    #     # Implement reverse discounting: higher discount for earlier steps
    #     reverse_discount_factor = 1 / (1 + self.discount_factor)

    #     td_target = reward + reverse_discount_factor * self.q_table[next_state_key][best_next_action]
    #     td_error = td_target - self.q_table[state_key][action]
    #     self.q_table[state_key][action] += self.learning_rate * td_error


    def initialize_state_in_q_table(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = [0 for _ in range(self.env.action_space.n)]


    def state_to_key(self, state):
        return tuple(state)
    
    
    def run_episode(self):
        state, info = self.env.reset()
        done, truncated = False, False
        total_reward = 0

        while not done and not truncated:
            action = self.get_action(state)
            next_state, reward, done, truncated, info = self.env.step(action)
            if print_state:
                print(f"State: {next_state}")
                
            if print_rewards:
                print(f"Reward: {reward}, info: {info}")
                
                
            self.update_q_table(state, action, reward, next_state)

            total_reward += reward
            state = next_state
            if one_run:
                self.exploration_rate *= self.exploration_decay
                if self.exploration_rate < 0.1:
                    self.exploration_rate = 0
                    print(self.env.unwrapped.controller.trader.dates[-1])

        else:
            self.exploration_rate *= self.exploration_decay
        metric_details = self.env.unwrapped.get_metrics()
        return total_reward, metric_details
    
    
    def run_episode_w_render(self):
        state, info = self.env.reset()
        done, truncated = False, False
        total_reward = 0

        while not done and not truncated:
            action = self.get_action(state)
            next_state, reward, done, truncated, info = self.env.step(action)
            
            if print_state:
                print(f"State: {next_state}")
                
            if print_rewards:
                print(f"Reward: {reward}")
                
            self.update_q_table(state, action, reward, next_state)

            total_reward += reward
            state = next_state
            
            time.sleep(render_wait_time)

            if one_run:
                self.exploration_rate *= self.exploration_decay
                print(f"Exploration rate: {self.exploration_rate}")
            
        if not one_run:
            self.exploration_rate *= self.exploration_decay
            
        metric_details = self.env.unwrapped.get_metrics()
        return total_reward, metric_details

if train_episodes == 1:
    one_run = True
else:
    one_run = False


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Base Turtle Trading Strategy with specified parameters.')
    parser.add_argument('--all_stocks', dest='all_stocks', action='store_true', help='Process all stocks if this flag is set.')
    parser.add_argument('--render', dest='render', action='store_true', help='Render the environment if this flag is set.')
    parser.set_defaults(render=False, all_stocks=False)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    render_mode = 'human' if args.render else None
        
    agent = QLearningAgent(None)
    
    if args.all_stocks:
        run_all_stocks(agent, override_params)
    else:
        env = gym.make('TurtleTradingEnv', render_mode=render_mode, override_params=override_params)
        agent = QLearningAgent(env)
        train_and_evaluate_agent_single_stock(agent, env, train_episodes)
        
# if override_params['render'] == True:
#     render_mode = 'human'
# else:
#     render_mode = None
    
# env = gym.make('TurtleTradingEnv', render_mode=render_mode, override_params=override_params)
# agent = QLearningAgent(env)

# train_and_evaluate_agent(agent, env, train_episodes, render_final_episode=True)

