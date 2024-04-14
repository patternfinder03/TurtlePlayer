# Helper functions for logging training results and plotting them from a log file

import os
import json
import numpy as np
import importlib.util
import matplotlib
import pandas as pd
from tabulate import tabulate
from .trade import Trade
matplotlib.use('TkAgg') # Hack; MATPLOTLIB not showing without thanks GPT!
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

current_working_dir = Path.cwd()

# Construct paths relative to the current working directory
base_log_dir = current_working_dir / 'logs'
config_dir = current_working_dir / 'config.py'



def log_config_file(session_log_dir):
    """
    Logs the contents of the config.py file to the session's log directory.
    """
    if os.path.exists(config_dir):
        config_file_path = config_dir
    else:
        print(f"Config file not found in {config_dir}")
        return

    # Define the path for the new log file within the session's log directory
    config_log_file_path = os.path.join(session_log_dir, 'config_log.txt')

    # Read the content of config.py and write it to the new log file
    try:
        with open(config_file_path, 'r') as config_file, open(config_log_file_path, 'w') as config_log_file:
            config_content = config_file.read()
            config_log_file.write(config_content)
    except IOError as e:
        print(f"Error logging config file: {e}")



def get_session_log_directory(last_episode):
    """
    Creates a unique log directory for a new session within the turtlePlayer/logs directory, 
    using an incremental integer as the directory name, and returns its path.
    """
    # Update the directory path to go two levels up from the current file's directory
    log_dir_base = base_log_dir
    index_file_path = os.path.join(log_dir_base, 'index.txt')

    # Check if the index file exists and read the current index
    if os.path.exists(index_file_path):
        with open(index_file_path, 'r') as index_file:
            index = int(index_file.read().strip()) + 1
    else:
        index = 1  # Start with index 1 if the index file does not exist

    # Update the index file with the new index
    if last_episode:
        with open(index_file_path, 'w') as index_file:
            index_file.write(str(index))

    # Create the new session log directory using the index
    session_log_dir = os.path.join(log_dir_base, str(index))
    os.makedirs(session_log_dir, exist_ok=True)

    return session_log_dir



def get_log_file_path(session_number, file_name):
    """
    Determines the correct log file path, checking two potential directories.

    :param session_number: The session number for which the log file is needed.
    :param file_name: The name of the log file to look for.
    :return: The path to the log file if found, else None.
    """
    global base_log_dir
    primary_path = os.path.join(base_log_dir, str(session_number), file_name)

    if os.path.exists(primary_path):
        return primary_path
    else:
        print(f"Log file {file_name} not found in either directory for session {session_number}.")
        return None



def save_single_stock_state_history(symbol, dates, prices, total_value_list, account_total_equity_list, account_total_dollar_list, num_units_traded_list, rewards, periods, exit_periods, exploration_rates, log_dir, file_name):
    """
    Saves the state history log as a CSV file.

    :param symbol: The stock symbol.
    :param dates: List of dates for each recorded state.
    :param prices: List of prices for each state.
    :param total_value_list: List of total values for each state.
    :param account_total_equity_list: List of total equity values for each state.
    :param account_total_dollar_list: List of total dollar values for each state.
    :param num_units_traded_list: List of number of units traded for each state.
    :param log_dir: The directory where the log file will be saved.
    """
    # Create a DataFrame from the provided lists
    
    rewards.append(0)  # Add a zero to the end of the list to match the length of the other lists
    # This doesn't cause misalighment as in the step function the next price is calculated before returning
    
    # print(len(dates), len(prices), len(total_value_list), len(account_total_equity_list), len(account_total_dollar_list), len(num_units_traded_list), len(rewards), len(periods), len(exploration_rates))
    # print(f"Length of dates and rewards: {len(dates)} and {len(rewards)}")
    state_history_data = pd.DataFrame({
        'Symbol': [symbol] * len(dates),
        'Date': dates,
        'Price': prices,
        'Total Value': total_value_list,
        'Account Total Equity': account_total_equity_list,
        'Account Total Dollar': account_total_dollar_list,
        'Num Units Traded': num_units_traded_list,
        'Reward': rewards,
        'Period': periods,
        'Exit Period': exit_periods,
        'Exploration Rate': exploration_rates
    })

    state_history_data = state_history_data.round(3)

    csv_file_path = os.path.join(log_dir, file_name)
    state_history_data.to_csv(csv_file_path, index=False)



def save_single_stock_trading_log(trade_list, log_dir, file_name):
    """
    Converts trade_list elements to Python native types, rounds float values, and saves the trading log as a CSV file.
    """
    trades_data = pd.DataFrame([{
        k: round(v, 3) if isinstance(v, float) else int(v) if type(v) is np.int64 else v 
        for k, v in trade.__dict__.items()
    } for trade in trade_list])
    
    csv_file_path = os.path.join(log_dir, file_name)
    trades_data.to_csv(csv_file_path, index=False)



def save_single_stock_training_log(agent_name, rewards, final_periods, mean_periods, median_periods, total_pnl_percents, total_pnls, exploration_rates, units_traded, session_log_dir):
    """
    Saves the training results in the current session's log directory.
    """
    log_file_path = os.path.join(session_log_dir, 'training_log.json')

    log_data = {
        'agent_type': agent_name,
        'training_results': {
            'rewards': rewards,
            'final_periods': final_periods,
            'mean_periods': mean_periods,
            'median_periods': median_periods,
            'total_pnl_percents': total_pnl_percents,
            'total_pnls': total_pnls,
            'exploration_rates': exploration_rates,
            'units_traded': units_traded 
        }
    }

    with open(log_file_path, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)


def plot_training_results_from_log(session_number):
    """
    Plots important metrics from the specified training log.
    """
    file_path = get_log_file_path(session_number, 'training_log.json')
    if not file_path:
        return

    with open(file_path, 'r') as log_file:
        log_data = json.load(log_file)

    training_results = log_data['training_results']
    rewards = training_results['rewards']
    final_periods = training_results['final_periods']
    mean_periods = training_results['mean_periods']
    median_periods = training_results['median_periods']
    total_pnls = training_results['total_pnls']
    exploration_rates = training_results['exploration_rates']

    fig, axs = plt.subplots(6, 1, figsize=(10, 20))

    metrics = [
        (rewards, 'Rewards', 'blue'),
        (final_periods, 'Final Periods', 'red'),
        (mean_periods, 'Mean Periods', 'green'),
        (median_periods, 'Median Periods', 'orange'),
        (total_pnls, 'PnL$', 'purple'),
        (exploration_rates, 'Exploration Rate', 'brown')
    ]

    for i, (metric, title, color) in enumerate(metrics):
        episodes = np.arange(len(metric))
        axs[i].plot(episodes, metric, label=title, color=color)
        z = np.polyfit(episodes, metric, 1)
        p = np.poly1d(z)
        axs[i].plot(episodes, p(episodes), "r--", label='Trend')
        axs[i].legend()
        axs[i].set_xlabel('Episode')
        axs[i].set_ylabel('Value')
        axs[i].set_title(title)
        
        
    statistics = []
    metrics = [
        (rewards, 'Rewards'),
        (final_periods, 'Final Periods'),
        (mean_periods, 'Mean Periods'),
        (median_periods, 'Median Periods'),
        (total_pnls, 'PnL$')
    ]

    for data, name in metrics:
        statistics.append([
            name,
            np.mean(data),
            np.std(data),
            np.min(data),
            np.max(data)
        ])

    # Print the statistics table
    headers = ['Metric', 'Mean', 'Standard Deviation', 'Minimum', 'Maximum']
    print(tabulate(statistics, headers=headers, tablefmt='grid'))

    plt.tight_layout()
    plt.show()
    
    
def tabulate_state_history_stats(session_number, start_date=None, end_date=None):
    directory_path = get_log_file_path(session_number, 'state_history_logs')
    if not os.path.exists(directory_path):
        print("Directory path not found.")
        return

    all_dfs = []

    # Step 1: Extract the episode numbers and sort the files based on them
    file_names = [f for f in os.listdir(directory_path) if f.startswith("state_history_log_ep_") and f.endswith(".csv")]
    file_names_sorted = sorted(file_names, key=lambda x: int(x.split('_')[4].split('.')[0]))

    # Step 2: Iterate through the sorted file names
    for file_name in file_names_sorted:
        file_path = os.path.join(directory_path, file_name)
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

        if 'Reward' in df.columns:
            df['Cumulative Reward'] = df['Reward'].cumsum()

        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date, format='%Y%m%d')]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date, format='%Y%m%d')]

        all_dfs.append(df)
            

    total_initial_value = total_final_value = total_cumulative_reward = pnl_change_sum = 0
    total_units_traded = 0

    pnl_statistics = []
    for idx, df in enumerate(all_dfs):
        if df.empty:
            continue

        episode_num = idx + 1
        # print(df)
        initial_value = df['Total Value'].iloc[0] if 'Total Value' in df.columns else np.nan
        final_value = df['Total Value'].iloc[-1] if 'Total Value' in df.columns else np.nan
        cumulative_reward = df['Cumulative Reward'].iloc[-1] if 'Cumulative Reward' in df.columns else np.nan
        pnl_change = ((final_value - initial_value) / initial_value) * 100 if initial_value != 0 else np.nan
        
        # Calculate the total units traded for the episode
        if 'Num Units Traded' in df.columns:
            units_traded = df['Num Units Traded'].iloc[-1] - df['Num Units Traded'].iloc[0]
        else:
            units_traded = 0

        total_initial_value += initial_value
        total_final_value += final_value
        total_cumulative_reward += cumulative_reward
        pnl_change_sum += pnl_change
        total_units_traded += units_traded

        pnl_statistics.append([
            f"Episode {episode_num}",
            f"{initial_value:,.2f}",
            f"{final_value:,.2f}",
            f"{cumulative_reward:,.2f}",
            f"{pnl_change:.2f}%",
            f"{units_traded:,.2f}"
        ])

    num_episodes = len(all_dfs)
    avg_initial_value = total_initial_value / num_episodes
    avg_final_value = total_final_value / num_episodes
    avg_cumulative_reward = total_cumulative_reward / num_episodes
    avg_pnl_change = pnl_change_sum / num_episodes
    avg_units_traded = total_units_traded / num_episodes

    pnl_statistics.append([
        "Episode Average",
        f"{avg_initial_value:,.2f}",
        f"{avg_final_value:,.2f}",
        f"{avg_cumulative_reward:,.2f}",
        f"{avg_pnl_change:.2f}%",
        f"{avg_units_traded:,.2f}"
    ])

    headers = ['Episode', 'Initial Total Value', 'Final Total Value', 'Cumulative Reward', 'PnL% Change', 'Total Units Traded']
    print(tabulate(pnl_statistics, headers=headers, tablefmt='grid'))



def plot_state_history(session_number, episode_nums, start_date=None, end_date=None):
    """
    Plot account states over time for specified episode numbers (or all episodes if 'All' is passed),
    filtered by the specified date range on the same plot for comparison.

    Parameters:
    - session_number: The session number to plot data for.
    - episode_nums: A list of episode numbers to plot or 'All' for all episodes.
    - start_date: The start date to filter the data.
    - end_date: The end date to filter the data.
    """
    
    if episode_nums != 'All':
        directory_path = get_log_file_path(session_number, 'state_history_logs')
        if not os.path.exists(directory_path):
            print("Directory path not found.")
            return

        # Define metrics to be plotted
        metrics = ['Price', 'Account Total Equity', 'Num Units Traded', 'Cumulative Reward']
        period_metrics = ['Period', 'Exit Period']

        # Initialize subplots outside the loop
        fig, axs = plt.subplots(len(metrics) + 2, 1, figsize=(12, 3 * (len(metrics) + 2)))

        pnl_statistics = []

        for episode_num in episode_nums:
            file_path = os.path.join(directory_path, f'state_history_log_ep_{episode_num}.csv')
            if not os.path.exists(file_path):
                print(f"File path not found for episode {episode_num}.")
                continue

            state_log_df = pd.read_csv(file_path)
            state_log_df['Date'] = pd.to_datetime(state_log_df['Date'], format='%Y%m%d')

            # Apply date filtering
            if start_date:
                state_log_df = state_log_df[state_log_df['Date'] >= pd.to_datetime(start_date, format='%Y%m%d')]
            if end_date:
                state_log_df = state_log_df[state_log_df['Date'] <= pd.to_datetime(end_date, format='%Y%m%d')]

            # Calculating the cumulative sum of 'Reward'
            if 'Reward' in state_log_df.columns:
                state_log_df['Cumulative Reward'] = state_log_df['Reward'].cumsum()

            # Plot individual metrics for each episode on the same axes
            for i, metric in enumerate(metrics):
                axs[i].plot(state_log_df['Date'], state_log_df[metric], label=f"{metric} (Episode {episode_num})")
                axs[i].set_title(f"{metric} over Time")
                axs[i].set_xlabel('Date')
                axs[i].set_ylabel(metric)
                axs[i].legend()

            # Plotting 'Total Value' and 'Exploration Rate' on the same subplot with different y-axes
            ax_tv = axs[-2]
            ax_tv.plot(state_log_df['Date'], state_log_df['Total Value'], label=f"Total Value (Episode {episode_num})")
            ax_tv.set_title('Total Value and Exploration Rate over Time')
            ax_tv.set_xlabel('Date')
            ax_tv.set_ylabel('Total Value')
            ax_tv.legend(loc='upper left')

            ax_er = ax_tv.twinx()  # Make sure to only call twinx once, outside the loop
            if 'Exploration Rate' in state_log_df.columns:
                ax_er.plot(state_log_df['Date'], state_log_df['Exploration Rate'], label=f'Exploration Rate (Episode {episode_num})', linestyle='--')
                ax_er.set_ylabel('Exploration Rate')
                ax_er.legend(loc='upper right')

            # Plot Period and Exit Period on the same subplot
            ax_periods = axs[-1]
            for metric in period_metrics:
                ax_periods.plot(state_log_df['Date'], state_log_df[metric], label=f"{metric} (Episode {episode_num})")
            ax_periods.set_title('Period and Exit Period over Time')
            ax_periods.set_xlabel('Date')
            ax_periods.set_ylabel('Value')
            ax_periods.legend()

            # Collect PnL statistics for tabulation
            if 'Total Value' in state_log_df.columns and not state_log_df.empty:
                initial_value = state_log_df['Total Value'].iloc[0]
                final_value = state_log_df['Total Value'].iloc[-1]
                pnl_change = ((final_value - initial_value) / initial_value) * 100 if initial_value != 0 else 0
                pnl_statistics.append([
                    f"Episode {episode_num}", 
                    f"{initial_value:,.2f}",
                    f"{final_value:,.2f}",
                    f"{pnl_change:.2f}%"
                ])

        plt.tight_layout()
        plt.show()

        # Display PnL% change using tabulate
        if pnl_statistics:
            headers = ['Episode', 'Initial Total Value', 'Final Total Value', 'PnL% Change']
            print(tabulate(pnl_statistics, headers=headers, tablefmt='grid'))
    else:
        tabulate_state_history_stats(session_number, start_date, end_date)


def plot_trading_log(session_number, episode_nums, start_date=None, end_date=None):
    """
    For specific episodes, plots the distribution of PnL percentages on multiple subplots within a single figure
    with consistent bins and x-axis ranges. Displays statistics in a table format.
    For 'All', displays statistics in a table format for each trading log without plotting.
    An 'Episode Average' row is included in the statistics table.

    Parameters:
    - session_number: The session number to plot data for.
    - episode_nums: A list of episode numbers or 'All' for all episodes.
    - start_date: The start date to filter the data.
    - end_date: The end date to filter the data.
    """
    directory_path = get_log_file_path(session_number, 'trading_logs')
    if not os.path.exists(directory_path):
        print("Directory path not found.")
        return

    all_stats = []
    global_min, global_max = float('inf'), float('-inf')
    to_plot = True

    if episode_nums == 'All':
        to_plot = False
        episode_files = [file for file in os.listdir(directory_path) if file.startswith("trading_log_") and file.endswith(".csv")]
        episode_nums = [file_name.split('_')[-1].split('.')[0] for file_name in episode_files]
    else:
        episode_files = [f"trading_log_ep_{ep}.csv" for ep in episode_nums]

    if episode_nums != 'All':
        # Determine global min and max PnL_percent for consistent binning
        for episode_file in episode_files:
            file_path = os.path.join(directory_path, episode_file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                local_min = df['PnL_percent'].min()
                local_max = df['PnL_percent'].max()
                global_min = min(global_min, local_min)
                global_max = max(global_max, local_max)

        # Set up the subplot figure
        fig, axs = plt.subplots(len(episode_files), figsize=(10, 6 * len(episode_files)))
        # Define consistent bins based on global min and max
        bins = np.linspace(global_min, global_max, 50)

    for idx, episode_file in enumerate(episode_files):
        file_path = os.path.join(directory_path, episode_file)
        if not os.path.exists(file_path):
            print(f"File path not found for {episode_file}.")
            continue

        episode_num = episode_file.split('_')[-1].split('.')[0]

        df = pd.read_csv(file_path)
        df['start_date'] = pd.to_datetime(df['start_date'], format='%Y%m%d')
        df['end_date'] = pd.to_datetime(df['end_date'], format='%Y%m%d')

        if start_date:
            df = df[df['start_date'] >= pd.to_datetime(start_date, format='%Y%m%d')]
        if end_date:
            df = df[df['end_date'] <= pd.to_datetime(end_date, format='%Y%m%d')]

        # Plotting for specific episodes
        if episode_nums != 'All':
            ax = axs[idx] if len(episode_files) > 1 else axs
            df['PnL_percent'].hist(bins=bins, ax=ax)
            ax.set_title(f'Distribution of PnL Percentages for Episode {episode_num}')
            ax.set_xlabel('PnL Percent')
            ax.set_ylabel('Frequency')

        # Compute and collect stats
        pnl_stats = df['PnL'].describe().reset_index()
        pnl_percent_stats = df['PnL_percent'].describe().reset_index()
        merged_stats = pd.merge(pnl_stats, pnl_percent_stats, on='index', suffixes=('_PnL', '_PnL_percent'))
        all_stats.append((episode_num, merged_stats))

    print(episode_nums)
    if episode_nums != 'All' and to_plot:
        plt.tight_layout()
        plt.show()

    # Calculating averages across all episodes
    avg_stats = pd.concat([stats for _, stats in all_stats]).groupby('index').mean().reset_index()
    all_stats.append(("Episode Average", avg_stats))

    # Display the statistics in a tabulated format for each episode and the average
    for episode_num, stats in all_stats:
        print(f"\nPnL Statistics for {episode_num}:")
        print(tabulate(stats, headers='keys', tablefmt='psql', showindex=False))
        
        
def plot_state_history_comparison(session_number1, session_number2, episode_nums1, episode_nums2, start_date=None, end_date=None):
    """
    Plot account states over time for specified episode numbers from two sessions,
    filtered by the specified date range on the same plot for comparison.

    Parameters:
    - session_number1: The first session number to plot data for.
    - session_number2: The second session number to plot data for.
    - episode_nums1: The list of episode numbers from the first session to plot.
    - episode_nums2: The list of episode numbers from the second session to plot.
    - start_date: The start date to filter the data.
    - end_date: The end date to filter the data.
    """
    metrics = ['Price', 'Account Total Equity']
    colors = ['blue', 'red']

    # Adjust the figure size and the subplot layout
    fig, axs = plt.subplots(len(metrics) + 3, 1, figsize=(12, 10))  # Total height increased
    
    
    min_exploration_rate = float('inf')
    max_exploration_rate = float('-inf')
    
    min_cumulative_reward = float('inf')
    max_cumulative_reward = float('-inf')

    # First, determine the global min and max Exploration Rate across all data
    for session_number, episode_nums, color in [(session_number1, episode_nums1, colors[0]), (session_number2, episode_nums2, colors[1])]:
        directory_path = get_log_file_path(session_number, 'state_history_logs')
        
        for episode_num in episode_nums:
            file_path = os.path.join(directory_path, f'state_history_log_ep_{episode_num}.csv')
            
            if not os.path.exists(file_path):
                continue

            state_log_df = pd.read_csv(file_path)
            state_log_df['Date'] = pd.to_datetime(state_log_df['Date'], format='%Y%m%d')
            state_log_df['Cumulative Reward'] = state_log_df['Reward'].cumsum()

            # Update the min/max values for exploration rate across all episodes and sessions
            min_exploration_rate = min(min_exploration_rate, state_log_df['Exploration Rate'].min())
            max_exploration_rate = max(max_exploration_rate, state_log_df['Exploration Rate'].max())
            
            min_cumulative_reward = 0
            max_cumulative_reward = max(max_cumulative_reward, state_log_df['Cumulative Reward'].max())
            

    # Load and filter data for each session and episode
    for session_number, episode_nums, color in [(session_number1, episode_nums1, colors[0]), (session_number2, episode_nums2, colors[1])]:
        directory_path = get_log_file_path(session_number, 'state_history_logs')
        
        for episode_num in episode_nums:
            file_path = os.path.join(directory_path, f'state_history_log_ep_{episode_num}.csv')
            
            if not os.path.exists(file_path):
                print(f"File path not found for session {session_number}, episode {episode_num}.")
                continue

            state_log_df = pd.read_csv(file_path)
            state_log_df['Date'] = pd.to_datetime(state_log_df['Date'], format='%Y%m%d')

            # Apply date filtering
            if start_date:
                state_log_df = state_log_df[state_log_df['Date'] >= pd.to_datetime(start_date, format='%Y%m%d')]
            if end_date:
                state_log_df = state_log_df[state_log_df['Date'] <= pd.to_datetime(end_date, format='%Y%m%d')]

            # Calculating the cumulative sum of 'Reward'
            state_log_df['Cumulative Reward'] = state_log_df['Reward'].cumsum()

            # Plot individual metrics
            for i, metric in enumerate(metrics):
                axs[i].plot(state_log_df['Date'], state_log_df[metric], label=f"{metric} Session {session_number} Episode {episode_num}", color=color)
                axs[i].set_title(f"{metric} over Time", fontsize=10)
                axs[i].set_xlabel('Date', fontsize=9)
                axs[i].set_ylabel(metric, fontsize=9)
                axs[i].legend(framealpha=0.5, loc='upper left')

            # Plot Total Value and Exploration Rate
            ax_tv = axs[-3]
            ax_tv.plot(state_log_df['Date'], state_log_df['Total Value'], label=f"Total Value (S{session_number} E{episode_num})", color=color)
            ax_tv.set_title('Total Value and Exploration Rate', fontsize=10)
            ax_tv.set_xlabel('Date', fontsize=9)
            ax_tv.set_ylabel('Total Value', fontsize=9)
            ax_tv.legend(loc='upper left', framealpha=0.5)

            ax_er = ax_tv.twinx()
            ax_er.plot(state_log_df['Date'], state_log_df['Exploration Rate'], label=f'Exploration Rate (S{session_number} E{episode_num})', color=color, linestyle='--', linewidth=2.0)
            ax_er.set_ylabel('Exploration Rate', fontsize=9)
            ax_er.set_ylim([min_exploration_rate, max_exploration_rate])
            # Adjusting both legends to be in the upper right with different vertical offsets
            if color == 'blue':
                ax_er.legend(loc='lower right', bbox_to_anchor=(1, 0.2), framealpha=0.5)  # Slightly higher within the upper right
            else:
                ax_er.legend(loc='lower right', bbox_to_anchor=(1, 0.0), framealpha=0.5)  # Slightly lower within the upper right


            # Combined plot for Num Units Traded and Cumulative Reward
            ax_combined = axs[-2]
            ax_combined.plot(state_log_df['Date'], state_log_df['Num Units Traded'], label=f"Units Traded (S{session_number} E{episode_num})", color=color)
            ax_combined.set_ylabel('Units Traded', fontsize=9)
            ax_combined.set_title('Num Units Traded and Cumulative Reward', fontsize=10)
            ax_combined.legend(loc='upper left', framealpha=0.5)

            ax_cr = ax_combined.twinx()
            ax_cr.plot(state_log_df['Date'], state_log_df['Cumulative Reward'], label=f'Cumulative Reward (S{session_number} E{episode_num})', color=color, linestyle='--')
            ax_cr.set_ylabel('Cumulative Reward', fontsize=9)
            ax_cr.set_ylim([min_cumulative_reward, max_cumulative_reward])
            # Adjusting both legends to be in the lower right with different vertical offsets
            if color == 'blue':
                ax_cr.legend(loc='lower right', bbox_to_anchor=(1, 0.2), framealpha=0.5)  # Positioning slightly higher
            else:
                ax_cr.legend(loc='lower right', bbox_to_anchor=(1, 0.0), framealpha=0.5)  # Positioning even lower



            # Plot Periods
            ax_periods = axs[-1]
            ax_periods.plot(state_log_df['Date'], state_log_df['Period'], label=f"Period (S{session_number} E{episode_num})", color=color)
            ax_periods.plot(state_log_df['Date'], state_log_df['Exit Period'], label=f"Exit Period (S{session_number} E{episode_num})", linestyle='--', color=color)
            ax_periods.set_title('Period and Exit Period over Time', fontsize=10)
            ax_periods.set_xlabel('Date', fontsize=9)
            ax_periods.set_ylabel('Period Values', fontsize=9)
            ax_periods.legend(framealpha=0.5)

    # Adjust layout to avoid overlaps and make the plot look neat
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
    
    
def find_zero_exploration_intervals_and_data(session_number, print_all=True):
    directory_path = get_log_file_path(session_number, 'state_history_logs')
    if not os.path.exists(directory_path):
        print("Directory path not found.")
        return

    file_names = [f for f in os.listdir(directory_path) if f.startswith("state_history_log_ep_") and f.endswith(".csv")]
    file_names_sorted = sorted(file_names, key=lambda x: int(x.split('_')[4].split('.')[0]))

    pnl_statistics = []
    detailed_data = []
    date_pnl_map = {}

    for file_name in file_names_sorted:
        file_path = os.path.join(directory_path, file_name)
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

        if 'Reward' in df.columns:
            df['Cumulative Reward'] = df['Reward'].cumsum()

        df['zero_exploration'] = df['Exploration Rate'] == 0
        df['block'] = (df['zero_exploration'].shift(1) != df['zero_exploration']).astype(int).cumsum()

        zero_blocks = df[df['zero_exploration']].groupby('block')

        for block_num, block_data in zero_blocks:
            if block_data.empty:
                continue

            initial_value = block_data['Total Value'].iloc[0]
            final_value = block_data['Total Value'].iloc[-1]
            pnl_change = ((final_value - initial_value) / initial_value) * 100 if initial_value != 0 else 0

            start_date = block_data['Date'].iloc[0].strftime('%Y-%m-%d')
            end_date = block_data['Date'].iloc[-1].strftime('%Y-%m-%d')
            date_key = (start_date, end_date)

            if date_key not in date_pnl_map:
                date_pnl_map[date_key] = []

            date_pnl_map[date_key].append(pnl_change)
            
            pnl_statistics.append([
                file_name.replace('.csv', ''),
                start_date,
                end_date,
                f"{initial_value:.2f}",
                f"{final_value:.2f}",
                f"{pnl_change:.2f}%"
            ])

            # Collecting detailed data for each episode
            detailed_data.append([
                file_name.replace('.csv', ''),
                start_date,
                end_date,
                block_data.to_dict(orient='list')
            ])

    # Average PnL for each date pair
    average_pnl_statistics = []
    for dates, pnl_changes in date_pnl_map.items():
        avg_pnl = sum(pnl_changes) / len(pnl_changes)
        average_pnl_statistics.append([
            dates[0], dates[1], f"{avg_pnl:.2f}%"
        ])

    headers = ['Episode', 'Start Date', 'End Date', 'Initial Total Value', 'Final Total Value', 'PnL% Change']
    average_headers = ['Start Date', 'End Date', 'Average PnL% Change']
    if print_all:
        print(tabulate(pnl_statistics, headers=headers, tablefmt='grid'))
    print("\nAverage PnL% Change by Date:")
    print(tabulate(average_pnl_statistics, headers=average_headers, tablefmt='grid'))


def find_zero_exploration_intervals_comparison(base_session_number, session_number):
    find_zero_exploration_intervals_and_data(session_number, print_all=False)
    directory_path = get_log_file_path(session_number, 'state_history_logs')
    base_directory_path = get_log_file_path(base_session_number, 'state_history_logs')

    if not os.path.exists(directory_path) or not os.path.exists(base_directory_path):
        print("Directory path not found for one or both sessions.")
        return

    file_names = sorted(
        [f for f in os.listdir(directory_path) if f.startswith("state_history_log_ep_") and f.endswith(".csv")],
        key=lambda x: int(x.split('_')[4].split('.')[0])
    )

    unique_dates = set()
    date_pnl_map = {}

    # Collect all unique zero exploration intervals from the main session
    for file_name in file_names:
        file_path = os.path.join(directory_path, file_name)
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

        # Mark the rows where exploration rate is zero
        df['is_zero_exploration'] = (df['Exploration Rate'] == 0)
        df['block'] = df['is_zero_exploration'].ne(df['is_zero_exploration'].shift()).cumsum()

        blocks = df[df['is_zero_exploration']].groupby('block')
        for _, block in blocks:
            if block.empty:
                continue
            start_date = block['Date'].min()
            end_date = block['Date'].max()
            initial_value = block['Total Value'].iloc[0]
            final_value = block['Total Value'].iloc[-1]
            pnl_change = ((final_value - initial_value) / initial_value * 100) if initial_value != 0 else 0
            date_key = (start_date, end_date)

            unique_dates.add(date_key)
            if date_key not in date_pnl_map:
                date_pnl_map[date_key] = []
            date_pnl_map[date_key].append(pnl_change)

    # Sort the unique dates
    sorted_unique_dates = sorted(unique_dates, key=lambda x: x[0])

    # Process the base session data based on the intervals found above
    pnl_statistics_base_session = []
    for start_date, end_date in sorted_unique_dates:
        base_file_names = sorted(
            [f for f in os.listdir(base_directory_path) if f.startswith("state_history_log_ep_") and f.endswith(".csv")],
            key=lambda x: int(x.split('_')[4].split('.')[0])
        )
        
        if len(base_file_names) != 1:
            raise ValueError("For the first session please use a BaseAgent with only one episode.")

        for base_file_name in base_file_names:
            base_file_path = os.path.join(base_directory_path, base_file_name)
            base_df = pd.read_csv(base_file_path)
            base_df['Date'] = pd.to_datetime(base_df['Date'], format='%Y%m%d')

            mask = (base_df['Date'] >= start_date) & (base_df['Date'] <= end_date)
            filtered_df = base_df[mask]

            if not filtered_df.empty:
                initial_value = filtered_df['Total Value'].iloc[0]
                final_value = filtered_df['Total Value'].iloc[-1]
                pnl_change = ((final_value - initial_value) / initial_value * 100) if initial_value != 0 else 0

                pnl_statistics_base_session.append([
                    base_file_name.replace('.csv', ''),
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    f"{initial_value:.2f}",
                    f"{final_value:.2f}",
                    f"{pnl_change:.2f}%"
                ])

    headers_base = ['Episode', 'Start Date', 'End Date', 'Initial Total Value', 'Final Total Value', 'PnL% Change']
    print("\nDetailed PnL% Change by Date from Base Session Number:")
    print(tabulate(pnl_statistics_base_session, headers=headers_base, tablefmt='grid'))