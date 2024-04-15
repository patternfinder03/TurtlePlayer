# Config for params for turtle env and agent
# To run the specific agent file cd into agents and run the corresponding agent

from pathlib import Path
project_dir = Path.cwd()


## BE SURE TO LOOK THROUGH THE WHOLE CONFIG FILE BEFORE RUNNING ##

"""
Log Notes:

# Example logs
1. Microsoft Base
2. Microsoft DQN
3. Nvidia Base
4. Nvidia DQN
5. Ford Base
6. Ford DQN
7. Target Base
8. Target DQN
9. Macy's Base
10. Macy's DQN
11. Netflix Base
12. Netflix DQN
13. Coke Base
14. Coke DQN
15. Chevron Base
16. Chevron DQN
17. Google Base
18. Google DQN
19. Amazon Base
20. Amazon DQN
#

21. Your first log
"""

# CRITICAL CRITCAL CRITICAL CRITICAL CRITICAL CRITICAL CRITICAL CRITICAL CRITICAL
render_episode = False # Needs to be set to True for render_wait_time to activate
render_wait_time = 0.0001 # Lower time makes render go faster
save_log = False # Save training log
plot_training_results = False # Plot training results

# DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
print_period_results = False # Prints period results to console
print_trade_results = False # Prints trade results to console
print_state = False # Prints state to console
print_rewards = False # Prints rewards to console


# AGENT AGENT AGENT AGENT AGENT AGENT AGENT AGENT AGENT AGENT AGENT AGENT AGENT
discount_factor = .5 # Discount factor
train_episodes = 1 # Number of episodes to train | Base agent must have 1 episode | Note experience does not carry over between episodes as its the stock market and keeping any information is cheating
exploration_decay = .997 # Exploration decay rate
learning_rate = .001 # Learning rate
batch_size = 64 # Batch size
memory_size = 512 # Memory size

# ENVIRONMENT ENVIRONMENT ENVIRONMENT ENVIRONMENT ENVIRONMENT ENVIRONMENT ENVIRONMENT
override_params = {
    'state_type': 'BasicWithPositions', # Others: 'BasicWithPositions'
    'price_movement_type': 'Actual MSFT', # See price_movement/actual/data for tickers available. For more data contact ay13@illinois.edu
    'num_prev_obvs': 50, # Number of lags to use in state
    'offset_scaling': True, # Applies offset scaling to state
    'scale': True, # Applies a StandardScaler to state
    'base_period': 20, # Initial Period
    'expand_by': 1, # Increase period by
    'shrink_by': 1, # Decrease period by
    'absolute_min': 15, # Min period
    'absolute_max': 35, # Max period
    'account_value': 10000000, # Default is 1000000
}