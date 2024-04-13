# Various helper functions to assist with hyperparameter tuning and model evaluation.
# Not actually used anywhere in the code, but helps me visualize the process of hyperparameter tuning and model evaluation.

import numpy as np
import matplotlib.pyplot as plt

def calculate_discount_factor(steps, threshold):
    """
    Calculate the discount factor needed to reach a certain threshold value
    at a given number of steps.

    Args:
    - steps (int): The number of steps over which the threshold should be reached.
    - threshold (float): The threshold value that the discount factor should reach
                         over the given number of steps.

    Returns:
    - float: The calculated discount factor.
    """
    discount_factor = threshold ** (1 / steps)
    return discount_factor


def calculate_episodes_until_exploration_zero(initial_exploration_rate, exploration_decay, minimum_exploration_rate=0.01):
    """
    Calculate the number of episodes until the exploration rate reaches zero or a minimum threshold.
    
    Args:
    - initial_exploration_rate (float): The initial exploration rate.
    - exploration_decay (float): The decay rate applied to the exploration rate after each episode.
    - minimum_exploration_rate (float): The minimum threshold for the exploration rate.
    
    Returns:
    - int: The number of episodes until the exploration rate reaches the minimum threshold.
    """
    episode_count = 0
    current_exploration_rate = initial_exploration_rate
    while current_exploration_rate > minimum_exploration_rate:
        current_exploration_rate *= exploration_decay
        episode_count += 1
    
    return episode_count



def apply_gradient_reward(peak_step, total_steps, max_reward, action_length, slope_intensity):
    if action_length > 5:
        raise ValueError('Action length must be 5 or less.')
    rewards = np.zeros(total_steps)
    # Define the peak width based on the action length
    peak_width = max(1, min(total_steps, (5 -action_length)))
    
    for step in range(total_steps):
        # Calculate the distance from the peak within the current segment
        distance_from_peak = abs(step - peak_step)
        
        # Determine the reward based on the distance from the peak
        if distance_from_peak <= peak_width:
            rewards[step] = (1 - (distance_from_peak / peak_width)) * max_reward
        else:
            rewards[step] = 0

    return rewards





# Example usage
if __name__ == "__main__":
    # initial_exploration_rate = 1.0  # Starting at 100% exploration
    # exploration_decay = 0.9985  # Decrease exploration rate by 1% each episode
    # minimum_exploration_rate = 0.01  # Stop at 1% exploration

    # print(calculate_discount_factor(100, 0.1))  # 0.9772372209558108
    # print(calculate_episodes_until_exploration_zero(initial_exploration_rate, exploration_decay, minimum_exploration_rate))
    # Parameters
    # peak_step = 25
    # total_steps = 100
    # max_reward = 100  # Example max reward
    # action_lengths = [1, 3, 5]

    # # Plot
    # plt.figure(figsize=(10, 6))
    # for action_length in action_lengths:
    #     rewards = apply_gradient_reward(peak_step, total_steps, max_reward, action_length)
    #     plt.plot(range(1, total_steps + 1), rewards, label=f'Action Length {action_length}')

    # plt.title('Reward Distribution Across Steps for Different Action Lengths')
    # plt.xlabel('Step')
    # plt.ylabel('Reward')
    # plt.legend()
    # plt.grid()
    # plt.show()
    import numpy as np
    import matplotlib.pyplot as plt

    # Sequence length and dimensionality
    sequence_length = 10
    d_model = 6

    # Initialize the positional encoding matrix
    positional_encoding = np.zeros((sequence_length, d_model))

    # Populate it
    for pos in range(sequence_length):
        for i in range(d_model // 2):
            positional_encoding[pos, 2 * i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            positional_encoding[pos, 2 * i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))

    print("Positional Encoding Matrix:\n", positional_encoding)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(positional_encoding)
    plt.title('Positional Encoding')
    plt.xlabel('Position')
    plt.ylabel('Encoding value')
    plt.legend(['dim %d'%i for i in range(d_model)])
    plt.grid(True)
    plt.show()

    
    
