# Pass in the env and agent and will compute metrics, save to log, and display graphs

import os
import numpy as np
import matplotlib.pyplot as plt
from config import save_log, plot_training_results
from gym_turtle_env.log_helpers import get_session_log_directory, save_single_stock_training_log, save_single_stock_trading_log, save_single_stock_state_history, log_config_file

def train_and_evaluate_agent_single_stock(agent, env, train_episodes):
    """
    Train and evaluate a Q-Learning agent in a given environment.

    Args:
        agent: The agent to train and evaluate.
        env: The environment where the agent operates.
        train_episodes (int): Number of episodes to train the agent.
        render_final_episode (bool): Whether to render the final episode.
    """
    rewards = []
    final_periods = []
    mean_periods = []
    median_periods = []
    total_pnl_percents = []
    total_pnls = []
    exploration_rates = []
    total_units_traded = []
    
    if save_log:
        session_log_dir = get_session_log_directory(False)
        trading_log_dir = os.path.join(session_log_dir, "trading_logs")
        state_log_dir = os.path.join(session_log_dir, "state_history_logs")
        os.makedirs(trading_log_dir, exist_ok=True)
        os.makedirs(state_log_dir, exist_ok=True)

    

    for i in range(train_episodes):
        agent.reset_params()
        total_reward, metrics = agent.run_episode()
        rewards.append(total_reward)
        final_periods.append(metrics['current_period'])
        mean_periods.append(metrics['mean_period'])
        median_periods.append(metrics['median_period'])
        total_pnl_percents.append(metrics['total_pnl_percent'])
        total_pnls.append(metrics['total_pnl'])
        try:
            exploration_rates.append(agent.exploration_rate)
        except: 
            exploration_rates.append(0)
        total_units_traded.append(metrics['total_units_traded'])
        
        # Pass the episode-specific log directory to the log-saving functions
        if save_log:
            trading_log_filename = f"trading_log_ep_{i + 1}.csv"
            state_log_filename = f"state_history_log_ep_{i + 1}.csv"
            save_single_stock_trading_log(env.unwrapped.controller.trader.completed_trades, trading_log_dir, trading_log_filename)
            save_single_stock_state_history(env.unwrapped.controller.trader.symbol, env.unwrapped.controller.trader.dates, env.unwrapped.controller.trader.close_price_list, env.unwrapped.controller.trader.account_value_list, env.unwrapped.controller.trader.account_equity_list, env.unwrapped.controller.trader.account_dollars_list,env.unwrapped.controller.trader.units_traded_list, agent.reward_intervals, env.unwrapped.controller.trader.period_list, env.unwrapped.controller.trader.exit_period_list, agent.exploration_rates, state_log_dir, state_log_filename)
        print(f'Ep {i + 1}: Rew: {total_reward:.2f}, PnL: {metrics['total_pnl']}, PnL%: {metrics['total_pnl_percent']:.2f}, FP: {metrics['current_period']}, MP: {metrics['mean_period']:.2f}, MedP: {metrics['median_period']}, ExpRate: {exploration_rates[-1]}, total_units_traded: {metrics['total_units_traded']}')
    
    if save_log:
        session_log_dir = get_session_log_directory(True)
        save_single_stock_training_log(agent.name, rewards, final_periods, mean_periods, median_periods, total_pnl_percents, total_pnls, exploration_rates, total_units_traded, session_log_dir)
        log_config_file(session_log_dir)
        
    if plot_training_results:
        plot_single_stock_training_results(rewards, final_periods, mean_periods, median_periods, total_pnl_percents, exploration_rates, total_units_traded)
        
    if train_episodes > 0:
        print("Rewards: ", rewards, "Average Reward: ", np.mean(rewards))
        print("PnL$: ", total_pnls, "Average PnL$: ", np.mean(total_pnls))
        
        
        

def plot_single_stock_training_results(rewards, final_periods, mean_periods, median_periods, total_pnl_percents, exploration_rates, total_units_traded):
    """Plot the training results of the agent with the addition of total units traded."""
    fig, axs = plt.subplots(7, 1, figsize=(10, 18))  # Adjusting figure size to fit more comfortably on screen
    
    # Define episodes for X-axis
    episodes = np.arange(len(rewards))

    # Adding total_units_traded to the data to be plotted
    data_list = [rewards, final_periods, mean_periods, median_periods, total_pnl_percents, exploration_rates, total_units_traded]
    titles = ['Rewards', 'Final Periods', 'Mean Periods', 'Median Periods', 'PnL%', 'Exploration Rate', 'Total Units Traded']
    colors = ['black', 'red', 'green', 'orange', 'blue', 'purple', 'brown']

    for i, (data, title, color) in enumerate(zip(data_list, titles, colors)):
        axs[i].plot(episodes, data, label=title, color=color)
        axs[i].set_title(f'{title} Over Episodes')
        axs[i].set_xlabel('Episode')
        axs[i].set_ylabel(title)
        
        # Calculate and plot linear trend line
        z = np.polyfit(episodes, data, 1)
        p = np.poly1d(z)
        axs[i].plot(episodes, p(episodes), "r--", label='Trend')
        
        axs[i].legend()

    plt.tight_layout()
    plt.show()
