# Runs multiple stocks to help understand behavior better

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import json
import pandas as pd
import numpy as np
import gym_turtle_env
import gymnasium as gym
from tabulate import tabulate
import matplotlib.pyplot as plt
from config import train_episodes


def train_and_evaluate_agent_single_stock(agent, env, train_episodes):
    """
    Train and evaluate a Q-Learning agent in a given environment.

    Args:
        agent: The agent to train and evaluate.
        env: The environment where the agent operates.
        train_episodes (int): Number of episodes to train the agent.
    """
    total_rewards = []
    total_pnls = []
    
    for i in range(train_episodes):
        total_reward, metrics = agent.run_episode()
        total_rewards.append(total_reward)
        total_pnls.append(metrics['total_pnl'])
        print(f'Ep {i + 1}: PnL: {metrics['total_pnl']}')
        
    return total_reward, total_pnls



def run_all_stocks(agent, override_params, use_valid_stocks=True, directory_path="../price_movement/actual/data"):
    """
    Run multiple stocks, but just keep track of final PNL and reward
    """
    all_pnls = {}
    all_rewards = {}
    
    with open('../price_movement/actual/valid_stocks.json') as f:
        valid_stocks = json.load(f)['valid_files']
    
    valid_stocks = [stock.replace('.csv', '') for stock in valid_stocks]

    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} not found.")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            stock_name = filename.replace('.csv', '')
            stock_name_pass = f'Actual {stock_name}'
            
            if use_valid_stocks and stock_name not in valid_stocks:
                continue
            
            print(f"Running agent for {stock_name}...")
            override_params['price_movement_type'] = stock_name_pass

            env = gym.make('TurtleTradingEnv', render_mode=None, override_params=override_params)
            agent.reset(env)
            
            rewards, pnls = train_and_evaluate_agent_single_stock(agent, env, train_episodes)
            all_rewards[stock_name] = [rewards]
            all_pnls[stock_name] = pnls
    
    plot_and_summarize_results(all_rewards, all_pnls)



def plot_and_summarize_results(all_rewards, all_pnls):
    """
    Plot histograms and summarize the results for all stocks.
    """
    
    all_pnls_combined = [pnl for pnls in all_pnls.values() for pnl in pnls]
    all_rewards_combined = [reward for rewards in all_rewards.values() for reward in rewards]
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_pnls_combined, bins=50, color='blue', alpha=0.7)
    plt.title('Combined Distribution of PnLs Across All Stocks')
    plt.xlabel('PnL')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(all_rewards_combined, bins=50, color='green', alpha=0.7)
    plt.title('Combined Distribution of Rewards Across All Stocks')
    plt.xlabel('Rewards')
    plt.ylabel('Frequency')
    plt.show()

    print("Combined Statistics for All Stocks:\n")
    pnl_df = pd.DataFrame(all_pnls_combined, columns=['PnL'])
    reward_df = pd.DataFrame(all_rewards_combined, columns=['Reward'])
    
    pnl_stats = pnl_df.describe().reset_index()
    reward_stats = reward_df.describe().reset_index()

    merged_stats = pd.merge(pnl_stats, reward_stats, on='index', suffixes=('_PnL', '_Reward'))
    print(tabulate(merged_stats, headers='keys', tablefmt='psql', showindex=False))
    print("\n")
