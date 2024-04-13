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
from copy import deepcopy
from base_agent.baseagent import BaseAgent
from helpers import apply_gradient_reward
from agents.run_multiple_stocks import run_all_stocks
from agents.run_single_stock import train_and_evaluate_agent_single_stock
from config import override_params, render_wait_time, print_state, print_rewards, train_episodes, discount_factor, exploration_decay, learning_rate


class QLearningAgentN:
    def __init__(self, env, learning_rate=learning_rate, discount_factor=discount_factor, exploration_rate=1, exploration_decay=exploration_decay, one_run=True):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}
        self.base_agent_env = deepcopy(self.env)
        self.base_agent = BaseAgent(self.base_agent_env)
        
        self.rewards = []
        
        self.one_run = one_run
        
        self.name = 'Q-Learning Agent N'

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
            return np.argmax(q_values_masked)
        else:
            return random.choice(valid_actions)

        
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

    def initialize_state_in_q_table(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = [0 for _ in range(self.env.action_space.n)]


    def state_to_key(self, state):
        return tuple(state)
    
    
    def run_episode(self):
        self.rewards.append(0)
        state, info = self.env.reset()
        base_state, base_info = self.base_agent.env.reset()
        done, truncated = False, False
        total_reward = 0
        state_action_rewards = []

        last_sell_step = 0  # Track the last sell step

        while not done and not truncated:
            action = self.get_action(state)
            next_state, reward, done, truncated, info = self.env.step(action)
            
            base_action = self.base_agent.get_action()
            base_next_state, base_reward, base_done, base_truncated, base_info = self.base_agent.env.step(base_action)

            state_action_rewards.append((state, action, reward))

            if 'length_of_trade' in info and info['length_of_trade'] > 0:
                # Determine the time chunk and calculate rewards
                current_step = len(state_action_rewards)
                try:
                    peak_step = min(current_step - info['length_of_trade'], current_step - base_info['length_of_trade'])
                except:
                    peak_step = current_step - info['length_of_trade']
                max_reward = reward - base_reward  # Difference in rewards
                slope_intensity = 10

                # Apply gradient reward adjustment
                gradient_rewards = apply_gradient_reward(peak_step - last_sell_step, current_step - last_sell_step, max_reward, 1, slope_intensity)
                
                # Adjust rewards based on the gradient
                for i in range(last_sell_step, current_step):
                    adjusted_reward = gradient_rewards[i - last_sell_step]
                    past_state, past_action, _ = state_action_rewards[i]
                    next_state = state_action_rewards[min(i + 1, current_step - 1)][0] if i + 1 < current_step else next_state
                    self.update_q_table(past_state, past_action, adjusted_reward, next_state)
                    self.rewards.append(adjusted_reward)
                    # if adjusted_reward > 0 or adjusted_reward < 0:
                    #     print(f"Adjusted reward: {adjusted_reward}, past state: {past_state}, past action: {past_action}, next state: {next_state}")

                last_sell_step = current_step  # Update last sell step

            total_reward += reward
            state = next_state
            base_state = base_next_state

            if print_state:
                print(f"State: {state}")

            if print_rewards:
                print(f"Reward: {reward}, info: {info}")

            if self.one_run:
                self.exploration_rate *= self.exploration_decay
                if self.exploration_rate < 0.1:
                    self.exploration_rate = 0

        if not self.one_run:
            self.exploration_rate *= self.exploration_decay
            
        metric_details = self.env.unwrapped.get_metrics()
        return total_reward, metric_details


    
    
    # def run_episode(self):
    #         state, info = self.env.reset()
    #         base_state, base_info = self.base_agent.env.reset()
    #         done, truncated = False, False
    #         total_reward = 0

    #         # Store the state, action, reward tuples
    #         state_action_rewards = []

    #         while not done and not truncated:
    #             action = self.get_action(state)
    #             next_state, reward, done, truncated, info = self.env.step(action)
                
    #             base_action = self.base_agent.get_action()
    #             base_next_state, base_reward, base_done, base_truncated, base_info = self.base_agent.env.step(base_action)

    #             # Append the current state, action, and reward to the list
    #             state_action_rewards.append((state, action, reward))

    #             # Update Q-values for states leading up to the trade entry
    #             if 'length_of_trade' in info and info['length_of_trade'] > 0:
    #                 length_of_trade = info['length_of_trade']
    #                 # Loop backwards through state_action_rewards to assign rewards
    #                 # print(f"Length of trade: {length_of_trade}")
    #                 for i in range(1, min(length_of_trade + 1, len(state_action_rewards))):  
    #                     past_state, past_action, past_reward = state_action_rewards[-i]
    #                     # Update the reward for the state that led to the trade entry
    #                     # You might need to adjust this logic based on how you want to distribute the reward
    #                     adjusted_reward = reward if i == 1 else past_reward
    #                     # print(f"Adjusted reward: {adjusted_reward}")
    #                     self.update_q_table(past_state, past_action, adjusted_reward, state)

    #             total_reward += reward
    #             state = next_state

    #             if print_state:
    #                 print(f"State: {next_state}")

    #             if print_rewards:
    #                 print(f"Reward: {reward}, info: {info}")

    #             # Decay exploration rate
    #             if one_run:
    #                 self.exploration_rate *= self.exploration_decay
    #                 if self.exploration_rate < 0.1:
    #                     self.exploration_rate = 0
    #                     print(self.env.unwrapped.controller.trader.dates[-1])

    #         metric_details = env.unwrapped.get_metrics()
    #         return total_reward, metric_details
    
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

            if self.one_run:
                self.exploration_rate *= self.exploration_decay
                print(f"Exploration rate: {self.exploration_rate}")
            
        if not self.one_run:
            self.exploration_rate *= self.exploration_decay
            
        metric_details = env.unwrapped.get_metrics()
        return total_reward, metric_details




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
        
