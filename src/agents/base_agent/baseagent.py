# Base agent doesn't change the conditions of the turtle trader. 
# It just uses the default strategy of enter on 20 day highs and exit on 10 day lows.


import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import time
import argparse
import numpy as np
import gym_turtle_env
import gymnasium as gym
from agents.run_multiple_stocks import run_all_stocks
from agents.run_single_stock import train_and_evaluate_agent_single_stock
from config import override_params, render_wait_time, print_state, train_episodes, render_episode, print_rewards




class BaseAgent:
    def __init__(self, env):
        self.env = env
        self.name = 'BaseAgent'
        
        self.reward_intervals = []
        self.exploration_rates = [0]
        
    def reset(self, env):
        self.env = env

    def get_action(self):
        # Base turtle strategy doesn't change conditions
        return 2

    def reset_params(self):
        pass
    
    def run_episode(self):
        state = self.env.reset()

        done, truncated = False, False
        total_reward = 0
        
        days = []
        state_memory = []
        next_state_memory = []
        pre_prices = []
        previous_segment_prices = []
        segment_prices = []
        agent_windows = []
        agent_actions_converted = []
        agent_actions = []
        first_exit = True
        starting_index = 0
        
        
        reward_iter = 0

        while not done and not truncated:
            if render_episode:
                time.sleep(render_wait_time)
            
            self.exploration_rates.append(0)
            days.append(starting_index)
            state_memory.append(state)
            action = self.get_action()
            
            high_price = self.env.unwrapped.controller.trader.high_price_list[-1]
            low_price = self.env.unwrapped.controller.trader.low_price_list[-1]
            close_price = self.env.unwrapped.controller.trader.close_price_list[-1]
            segment_prices.append({"High": high_price, "Low": low_price, "Close": close_price})
            
            next_state, reward, done, truncated, info = self.env.step(action)
            agent_windows.append(20)
            agent_actions.append(2)
            agent_actions_converted.append("Nothing")
            next_state_memory.append(next_state)
            
            if info['is_close_10']:
                if not first_exit:
                    pre_prices = previous_segment_prices[-self.env.unwrapped.controller.trader.absolute_max:]
                    
                rewards, solver_data = self.env.unwrapped.controller.get_interval_reward_10(pre_prices=pre_prices,
                                stock_prices=segment_prices, trader_actions=agent_actions_converted, trader_windows=agent_windows,
                                first_exit=first_exit)
                
                self.reward_intervals.extend(rewards)

                for day_t, reward_t, state_t, next_state_t, action_t, agent_window_t, optimal_t in zip(days, rewards, state_memory, next_state_memory, agent_actions, agent_windows, solver_data):
                    if print_state:
                        print(f"Day: {day_t}, State: {state_t}")
                    if print_rewards:
                        print(f"Day:{day_t:4},Rew:{reward_t:5.2f},Act:{action_t:1},Per:{agent_window_t:2},ExpRate:{0},RIt:{reward_iter:3},IL:{int(optimal_t['smoothed_ideal'])},MIN:{optimal_t['min']},MAX:{optimal_t['max']},OA:{optimal_t['optimal_action']:13},EP:{optimal_t['exit_period']}")
                    total_reward += reward_t
                
                
                reward_iter += 1
                previous_segment_prices.extend(segment_prices)
                
                days = []
                segment_prices = []
                agent_windows = []
                agent_actions = []
                agent_actions_converted = []
                state_memory = []
                next_state_memory = []
                first_exit = False        

                
            state = next_state

            starting_index += 1

        metric_details = self.env.unwrapped.get_metrics()

        
        for _ in range(0, len(segment_prices)):
            self.reward_intervals.append(0)
        
        
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
        
    agent = BaseAgent(None)
    
    if args.all_stocks:
        run_all_stocks(agent, override_params)
    else:
        env = gym.make('TurtleTradingEnv', render_mode=render_mode, override_params=override_params)
        agent = BaseAgent(env)
        train_and_evaluate_agent_single_stock(agent, env, train_episodes)
        