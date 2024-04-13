# O(N^3) solution. Not my best work, but compute is not a problem here

# As there is more than 1 "optimal solution" to the dynamic turtle problem,
# I decided to try to make a solution this is optimal for helping compute the reward for
# the agent. I store the max, min, ideal look back window as well as optimal-possible-action
# at each time step and plan to use a combination of the three in the reward function.

# The ideal is a weighted average of the min, max with more weight on the max.

import matplotlib.pyplot as plt
import math
from typing import List, Tuple, Union



# Used for non-dynamic exit reward func

# This function is meant to be called when an exit occurs, and it's behavior is untested and will
# throw an error. This function should only be called when a theortical exit point is found(Even if the long count is 0).
def turtleSolver(starting_index: int, pre_prices: List[float], stock_prices: List[float], min_lookback: int, max_lookback: int, max_change: int = 1, weight_max_window:float =.66, first_exit:bool = False, initial_balance:float = 10000000, verbose: bool = True):
    if first_exit:
        pre_prices = [stock_prices[0]] * max_lookback
    elif len(pre_prices) < max_lookback:
        pre_prices = [stock_prices[0]] * (max_lookback - len(pre_prices)) + pre_prices
        
    
    assert len(pre_prices) >= min_lookback, "Pre-prices must be at least min_lookback days long. They have lengths of: " + str(len(pre_prices)) + " and " + str(min_lookback) + " respectively."""
    pre_prices_len = len(pre_prices)
    combined_prices = pre_prices + stock_prices
        
    if verbose:
        for p in pre_prices:
            print(p)
        
    def find_exit_points(stock_prices):
        exit_points = []
        for i in range(10, len(stock_prices)):
            if stock_prices[i]['Close'] < min(stock_prices[j]['Low'] for j in range(i - 10, i)):
                exit_points.append(i)
        return exit_points

    def calculate_lookback_windows(stock_prices, min_lookback, max_lookback):
        exit_points = find_exit_points(stock_prices)
        
        if len(exit_points) == 1:
            # Set exit points of 0 to index 0
            exit_points = [0] + exit_points
        else:
            print("DEBUG PRINT")
            for i, price in enumerate(stock_prices):
                print(i, price)
            raise ValueError("There should only be one exit point, there are " + str(len(exit_points)) + " exit points. at " + str(exit_points))

        
        lookback_windows = [{'min': -1, 'max': -1} for _ in range(len(stock_prices))]
        

        for i in range(len(exit_points) - 1):
            segment_start, segment_end = exit_points[i], exit_points[i + 1]
            close_price = stock_prices[segment_end]['Close']
            for day in range(segment_start, segment_end):
                min_window, max_window, optimal_action = -1, -1, 'Nothing'
                
                for lookback in range(min_lookback, max_lookback + 1):
                    if day + pre_prices_len - lookback < 0:
                        break
                    high_price = max(combined_prices[j + pre_prices_len]['High'] for j in range(day - lookback, day))
                    if close_price > stock_prices[day]['Close'] > high_price:
                        optimal_action = 'BuyRange'
                        if min_window == -1:
                            min_window = lookback
                        max_window = lookback
                    elif high_price >= stock_prices[day]['Close'] >= close_price:
                        optimal_action = 'AvoidBuyRange'
                        if min_window == -1:
                            min_window = lookback
                        max_window = lookback
                        
                if optimal_action == 'Nothing' and stock_prices[day]['Close'] >= close_price and day + starting_index >= min_lookback:
                    optimal_action = 'BAD_ForcedBuy'
                elif optimal_action == 'Nothing' and stock_prices[day]['Close'] < close_price and day + starting_index >= min_lookback:
                    optimal_action = 'BAD_CantBuy'
                    
                lookback_windows[day] = {'day': day + starting_index, 'close': stock_prices[day]['Close'],'min': min_window, 'max': max_window, 'optimal_action': optimal_action, 'exit_period': 10}
                
            lookback_windows[segment_end] = {'day': segment_end + starting_index, 'close': stock_prices[segment_end]['Close'], 'min': -1, 'max': -1, 'optimal_action': 'ClosePosition', 'exit_period': 10}


        for i, window in enumerate(lookback_windows):
            if window['min'] == -1 or window['max'] == -1:
                future_index = i + 1
                while future_index < len(lookback_windows):
                    if lookback_windows[future_index]['min'] != -1 and lookback_windows[future_index]['max'] != -1:
                        lookback_windows[i]['min'] = lookback_windows[future_index]['min']
                        lookback_windows[i]['max'] = lookback_windows[future_index]['max']
                        break
                    future_index += 1
                    
        return lookback_windows

    def non_linear_transition(lookback_windows, min_lookback, max_lookback):
        i = 0
        while i < len(lookback_windows) - 1:
            if lookback_windows[i]['optimal_action'] in ['BAD_CantBuy', 'BAD_ForcedBuy']:
                start = i
                while i < len(lookback_windows) and lookback_windows[i]['optimal_action'] in ['BAD_CantBuy', 'BAD_ForcedBuy']:
                    i += 1
                end = i - 1

                # Use the lookback window from the start of the next phase for the whole BAD phase
                if i < len(lookback_windows) and lookback_windows[i]['optimal_action'] in ['BuyRange', 'AvoidBuyRange']:
                    for j in range(start, end + 1):
                        lookback_windows[j]['adjusted_min'] = max(lookback_windows[i]['min'], min_lookback)
                        lookback_windows[j]['adjusted_max'] = min(lookback_windows[i]['max'], max_lookback)
            else:
                i += 1

        return lookback_windows

    def calculate_weighted_lookback_windows(lookback_windows):
        for window in lookback_windows:
            # Use adjusted values if available; otherwise, fall back to the original min and max
            min_window = window.get('adjusted_min', window['min'])
            max_window = window.get('adjusted_max', window['max'])

            # If both min and max windows are unset, set the ideal to -1
            if min_window == -1 and max_window == -1:
                window['ideal'] = -1
            else:
                # Use the original min/max if adjusted values are not available
                min_window = min_window if min_window != -1 else max_window
                max_window = max_window if max_window != -1 else min_window

                # Calculate the weighted average for the ideal window using weight_max_window
                window['ideal'] = int((min_window * (1 - weight_max_window)) + (max_window * weight_max_window))

        return lookback_windows


    def constrained_smoothing(lookback_windows, max_change=1):
        # Initialize the list for smoothed_ideal with the first ideal window value
        # If the first value is -1, temporarily set smoothed_ideal to 0
        smoothed_ideal = [lookback_windows[0]['ideal'] if lookback_windows[0]['ideal'] != -1 else 0]

        # Flag to check if we are still in the initial sequence of -1s
        initial_sequence = True if lookback_windows[0]['ideal'] == -1 else False
        first_non_negative_value = None

        # Apply constrained smoothing
        for i in range(1, len(lookback_windows)):
            current_ideal = lookback_windows[i]['ideal']

            if initial_sequence:
                if current_ideal == -1:
                    smoothed_ideal.append(0)  # Continue with 0 until a non-negative value is found
                else:
                    # First non-negative value found, update all previous values
                    first_non_negative_value = current_ideal
                    smoothed_ideal = [first_non_negative_value for _ in range(i)] + [first_non_negative_value]
                    initial_sequence = False
            else:
                # Once the initial sequence is done, apply the regular smoothing logic
                previous_ideal = smoothed_ideal[-1]
                if current_ideal == -1:
                    smoothed_ideal.append(previous_ideal)  # Carry over the previous value if current is -1
                else:
                    # Calculate the allowed change based on max_change
                    adjusted_ideal = max(min(current_ideal, previous_ideal + max_change), previous_ideal - max_change)
                    smoothed_ideal.append(adjusted_ideal)

        # Update the lookback windows with the constrained smoothed ideal values
        for i, window in enumerate(lookback_windows):
            window['smoothed_ideal'] = smoothed_ideal[i]

        return lookback_windows

    
    
    def calculate_max_profit(lookback_windows, stock_prices, initial_balance=10000000):
        total_invested = 0  # Total money invested
        investment = initial_balance * 0.0175  # 1.75% of account balance
        shares_held = 0
        pnl_dollars = 0

        for window in lookback_windows:
            action = window['optimal_action']
            day = window['day']
            close_price = stock_prices[day - starting_index]['Close']  # Adjust index based on starting_index
            
            if action == 'BuyRange' or action == 'BAD_ForcedBuy':  # Buy condition
                shares_bought = investment / close_price
                shares_held += shares_bought
                total_invested += investment  # Add investment to total invested

            if action == 'ClosePosition' and shares_held > 0:  # Sell condition
                sell_proceeds = shares_held * close_price
                pnl_dollars += sell_proceeds - total_invested  # Calculate PnL in dollars
                shares_held = 0  # Reset shares held after closing the position
                total_invested = 0  # Reset total invested

        # Calculate PnL in percentage
        pnl_percent = (pnl_dollars / initial_balance) * 100

        return pnl_dollars, pnl_percent



    # Execute the functions in sequence
    lookback_windows = calculate_lookback_windows(stock_prices, min_lookback, max_lookback)
    lookback_windows = non_linear_transition(lookback_windows, min_lookback, max_lookback)
    lookback_windows = calculate_weighted_lookback_windows(lookback_windows)
    lookback_windows = constrained_smoothing(lookback_windows, max_change)
    pnl_dollars, pnl_percent = calculate_max_profit(lookback_windows, stock_prices, initial_balance)
    return lookback_windows, pnl_dollars, pnl_percent



def calculate_reward(solver_data, agent_actions, agent_windows):
    assert len(agent_windows) == len(agent_actions) == len(solver_data), "Input lists must have the same length. They have lengths of: " + str(len(agent_windows)) + ", " + str(len(agent_actions)) + ", and " + str(len(solver_data)) + " respectively."""
    rewards = []  # List to store rewards for each time step
    future_window_size = 5  # Define the future window size for checking closeness to transition points
    optimal_windows = []

    # Iterate over each time step
    for i, (agent_window, agent_action) in enumerate(zip(agent_windows, agent_actions)):
        reward = 0  # Initialize reward for the current time step
        solver_window = solver_data[i]
        if solver_window['optimal_action'] == 'ClosePosition':
            rewards.append(reward)
            continue

        # Calculate distance from the ideal window
        distance_to_ideal = abs(agent_window - solver_window['smoothed_ideal'])

        # Check for approaching transition to BuyRange or AvoidBuyRange within the future window
        transition_approaching = False
        for j in range(1, min(future_window_size + 1, len(solver_data) - i)):
            future_window = solver_data[i + j]
            if future_window['optimal_action'] in ['BuyRange', 'AvoidBuyRange'] and solver_window['optimal_action'] not in ['BuyRange', 'AvoidBuyRange']:
                transition_approaching = True
                break

        # Smooth reward/penalty logic based on the distance to the ideal window
        if solver_window['optimal_action'] in ['BuyRange', 'AvoidBuyRange']:
            if solver_window['min'] <= agent_window <= solver_window['max']:
                # Reward is inversely proportional to the distance to the ideal, within the range
                base_reward = 0.75 * (1 - (distance_to_ideal / max((solver_window['max'] - solver_window['min']), 1)))
                if solver_window['optimal_action'] in ['BAD_ForcedBuy', 'BAD_CantBuy']:
                    if transition_approaching:
                        base_reward *= .55
                    else:
                        base_reward *= .3
                        
                if (agent_window > solver_window['smoothed_ideal'] and agent_action == 'Decrease') or \
                (agent_window < solver_window['smoothed_ideal'] and agent_action == 'Increase'):
                    reward += 0.15  # Bonus for moving towards the ideal
                elif agent_action == 'Nothing':
                    reward += 0.075
                else:
                    reward -= .15
                
                reward += base_reward
            else:
                # Try to penalize less when the min max window is super small
                base_penalty = -.5 * (1 - (1 / max(1/math.log(max(solver_window['max'] - solver_window['min'], 2)), 1)))
                if transition_approaching:
                    base_penalty *= 1.15
                elif solver_window['optimal_action'] in ['BuyRange', 'AvoidBuyRange']:
                    base_penalty *= 1.75
                reward += base_penalty
                
                if (agent_window < solver_window['min'] and agent_action == 'Increase') or \
                (agent_window > solver_window['max'] and agent_action == 'Decrease'):
                    reward += 0.2  # Bonus for aligning with moving towards the range
                else:
                    reward -= 0.2  # Penalty for moving away from the range
                            
        rewards.append(round(reward, 3))  # Append the reward for the current time step to the list
        optimal_windows.append(solver_window['smoothed_ideal'])

    return rewards, solver_data



# TEST CASES
if __name__ == '__main__':


    # Test case 1
    pre_prices = [
        {"High": 1, "Low": 1, "Close": 1},
    ]
        
    test_case_one = [{'High': 5, 'Low': 5, 'Close': 5}, {'High': 5, 'Low': 5, 'Close': 5}, {'High': 5, 'Low': 5, 'Close': 5}, {'High': 5, 'Low': 5, 'Close': 5}, {'High': 5, 'Low': 5, 'Close': 5},
              {'High': 5, 'Low': 5, 'Close': 5}, {'High': 5, 'Low': 5, 'Close': 5}, {'High': 10, 'Low': 10, 'Close': 10},
              {'High': 10, 'Low': 10, 'Close': 10}, {'High': 1, 'Low': 1, 'Close': 1}, {'High': 0, 'Low': 0, 'Close': 0}]
    lookback_windows, pnl, pnl_pct = turtleSolver(0, pre_prices, test_case_one, 1, 10, 1, .8, False)
    print("Test Case 1 ##############################################################")
    for window in lookback_windows:
        day = str(window['day']).zfill(5)
        min_window = str(window.get('adjusted_min', window['min'])).zfill(2)  # Use adjusted_min if available
        max_window = str(window.get('adjusted_max', window['max'])).zfill(2)  # Use adjusted_max if available
        ideal_window = str(window['ideal']).zfill(2) if window['ideal'] != -1 else '--'
        smoothed_ideal = str(window['smoothed_ideal']).zfill(2) if 'smoothed_ideal' in window else ideal_window
        action = window['optimal_action']
        print(f"Day {day}, Close: {window['close']:8.2f}, Min Window: {min_window}, Max Window: {max_window}, Ideal Window: {ideal_window}, Smoothed Ideal Window: {smoothed_ideal}, Optimal Action: {action}")

    print("PnL: ", pnl)

    # Test case 2
    
    pre_prices = [
        {"High": 1, "Low": 1, "Close": 1},
    ]
    
    test_case_two = [
        {"High": 5, "Low": 5, "Close": 5},
        {"High": 5, "Low": 5, "Close": 5},
        {"High": 5, "Low": 5, "Close": 5},
        {"High": 5, "Low": 5, "Close": 5},
        {"High": 6, "Low": 6, "Close": 6},
        {"High": 7, "Low": 7, "Close": 7},
        {"High": 8, "Low": 8, "Close": 8},
        {"High": 9, "Low": 9, "Close": 9},
        {"High": 10, "Low": 10, "Close": 10},
        {"High": 11, "Low": 11, "Close": 11},
        {"High": 12, "Low": 12, "Close": 12},
        {"High": 13, "Low": 13, "Close": 13},
        {"High": 14, "Low": 14, "Close": 14},
        {"High": 15, "Low": 15, "Close": 15},
        {"High": 16, "Low": 16, "Close": 16},
        {"High": 17, "Low": 17, "Close": 17},
        {"High": 17, "Low": 17, "Close": 17},
        {"High": 17, "Low": 17, "Close": 17},
        {"High": 17, "Low": 17, "Close": 17},
        {"High": 17, "Low": 17, "Close": 17},
        {"High": 16, "Low": 16, "Close": 16},
        {"High": 17, "Low": 17, "Close": 17},
        {"High": 17, "Low": 17, "Close": 17},
        {"High": 17, "Low": 17, "Close": 17},
        {"High": 15, "Low": 15, "Close": 15},
    ]
    lookback_windows, pnl, pnl_pct = turtleSolver(0, pre_prices, test_case_two, 1, 10, 4, .8, False)
    
    print("Test Case 2 ##############################################################")
    for window in lookback_windows:
        day = str(window['day']).zfill(5)
        min_window = str(window.get('adjusted_min', window['min'])).zfill(2)  # Use adjusted_min if available
        max_window = str(window.get('adjusted_max', window['max'])).zfill(2)  # Use adjusted_max if available
        ideal_window = str(window['ideal']).zfill(2) if window['ideal'] != -1 else '--'
        smoothed_ideal = str(window['smoothed_ideal']).zfill(2) if 'smoothed_ideal' in window else ideal_window
        action = window['optimal_action']
        print(f"Day {day}, Close: {window['close']}, Min Window: {min_window}, Max Window: {max_window}, Ideal Window: {ideal_window}, Smoothed Ideal Window: {smoothed_ideal}, Optimal Action: {action}")
    
    print("PnL: ", pnl)
        
    pre_prices = [
        {"High": 1, "Low": 1, "Close": 1},
        {"High": 1, "Low": 1, "Close": 1},
        {"High": 1, "Low": 1, "Close": 1},
        {"High": 1, "Low": 1, "Close": 1},
        {"High": 1, "Low": 1, "Close": 1},
    ]

    test_case_three = [
        {"High": 1.1, "Low": 1.1, "Close": 1.1},
        {"High": 1, "Low": 1, "Close": 1},
        {"High": 2, "Low": 2, "Close": 2},
        {"High": 1, "Low": 1, "Close": 1},
        {"High": 3, "Low": 3, "Close": 3},
        {"High": 2, "Low": 2, "Close": 2},
        {"High": 1, "Low": 1, "Close": 1},
        {"High": 3, "Low": 3, "Close": 3},
        {"High": 2, "Low": 2, "Close": 2},
        {"High": 3, "Low": 3, "Close": 3},
        {"High": 4, "Low": 4, "Close": 4},
        {"High": 4, "Low": 4, "Close": 4},
        {"High": 3, "Low": 3, "Close": 3},
        {"High": 2, "Low": 2, "Close": 2},
        {"High": 3, "Low": 3, "Close": 3},
        {"High": 4, "Low": 4, "Close": 4},
        {"High": 4, "Low": 4, "Close": 4},
        {"High": 3, "Low": 3, "Close": 3},
        {"High": 2, "Low": 2, "Close": 2},
        {"High": 5, "Low": 5, "Close": 5},
        {"High": 4, "Low": 4, "Close": 4},
        {"High": 5, "Low": 5, "Close": 5},
        {"High": 5, "Low": 5, "Close": 5},
        {"High": 6, "Low": 6, "Close": 6},
        {"High": 7, "Low": 7, "Close": 7},
        {"High": 5, "Low": 5, "Close": 5},
        {"High": 6, "Low": 6, "Close": 6},
        {"High": 7, "Low": 7, "Close": 7},
        {"High": 7, "Low": 7, "Close": 7},
        {"High": 7, "Low": 7, "Close": 7},
        {"High": 8, "Low": 8, "Close": 8},
        {"High": 7, "Low": 7, "Close": 7},
        {"High": 7, "Low": 7, "Close": 7},
        {"High": 6, "Low": 6, "Close": 6},
        {"High": 7, "Low": 7, "Close": 7},
        {"High": 7, "Low": 7, "Close": 7},
        {"High": 6, "Low": 6, "Close": 6},
        {"High": 7, "Low": 7, "Close": 7},
        {"High": 8, "Low": 8, "Close": 8},
        {"High": 8, "Low": 8, "Close": 8},
        {"High": 9, "Low": 9, "Close": 9},
        {"High": 9, "Low": 9, "Close": 9},
        {"High": 10, "Low": 10, "Close": 10},
        {"High": 11, "Low": 11, "Close": 11},
        {"High": 12, "Low": 12, "Close": 12},
        {"High": 13, "Low": 13, "Close": 13},
        {"High": 13, "Low": 13, "Close": 13},
        {"High": 13, "Low": 13, "Close": 13},
        {"High": 14, "Low": 14, "Close": 14},
        {"High": 13, "Low": 13, "Close": 13},
        {"High": 14, "Low": 14, "Close": 14},
        {"High": 8, "Low": 8, "Close": 8},
    ]


    lookback_windows, pnl, pnl_pct = turtleSolver(0, pre_prices, test_case_three, min_lookback=5, max_lookback=15, max_change=5, weight_max_window=.8, first_exit=False)
    
    agent_windows = [13] * 52
    agent_actions = ['NA'] * 52

    # Calculate the reward using the test case
    rewards, _ = calculate_reward(lookback_windows, agent_actions, agent_windows)

    print("Test Case 3 ##############################################################")
    for i, window in enumerate(lookback_windows):
        day = str(window['day']).zfill(5)
        min_window = str(window.get('adjusted_min', window['min'])).zfill(2)  # Use adjusted_min if available
        max_window = str(window.get('adjusted_max', window['max'])).zfill(2)  # Use adjusted_max if available
        ideal_window = str(window['ideal']).zfill(2) if window['ideal'] != -1 else '--'
        smoothed_ideal = str(window['smoothed_ideal']).zfill(2) if 'smoothed_ideal' in window else ideal_window
        action = window['optimal_action']
        reward = rewards[i]
        agent_action = agent_actions[i]
        agent_window = agent_windows[i]
        print(f"Day {day}, Close: {window['close']}, Min Window: {min_window}, Max Window: {max_window}, Ideal Window: {ideal_window}, Smoothed Ideal Window: {smoothed_ideal}, Reward: {reward}, O Action: {action}, Action: {agent_action}, Window: {agent_window}")
        
        
    print("PnL: ", pnl)
    
    
    days = [window['day'] for window in lookback_windows]
    ideal_windows = [window['smoothed_ideal'] for window in lookback_windows]
    agent_windows_values = [20] * len(lookback_windows)  # Assuming agent windows are all set to 20 for this example

    # Plotting the ideal windows
    plt.figure(figsize=(10, 5))
    plt.plot(days, ideal_windows, label='Ideal Windows', color='blue', linestyle='-')

    # Plotting the agent windows
    plt.plot(days, agent_windows_values, label='Agent Windows', color='green', linestyle='--')

    # Plotting the rewards
    plt.plot(days, rewards, label='Rewards', color='red', linestyle='-.')

    plt.title('Ideal Windows, Agent Windows, and Rewards Over Time')
    plt.xlabel('Day')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    # Test case 4
    pre_prices = [
        {"High": 1, "Low": 1, "Close": 1},
    ]

    test_case_two = []
    # Gradual increase
    for i in range(1, 21):
        test_case_two.append({'High': i, 'Low': i - 0.5, 'Close': i - 0.25})

    # Fluctuating plateau
    for i in range(21, 41):
        test_case_two.append({'High': 20 + (i % 3), 'Low': 18 + (i % 2), 'Close': 19 + (i % 2)})

    # Sharp drop
    for i in range(41, 51):
        test_case_two.append({'High': 50 - i, 'Low': 50 - i - 1, 'Close': 50 - i - 0.5})
        
    lookback_windows, pnl, pnl_pct = turtleSolver(0, pre_prices, test_case_three, min_lookback=1, max_lookback=15, max_change=5, weight_max_window=.8, first_exit=False)
    

    # print("Test Case 4 ##############################################################")
    # for window in lookback_windows:
    #     day = str(window['day']).zfill(5)
    #     min_window = str(window.get('adjusted_min', window['min'])).zfill(2)  # Use adjusted_min if available
    #     max_window = str(window.get('adjusted_max', window['max'])).zfill(2)  # Use adjusted_max if available
    #     ideal_window = str(window['ideal']).zfill(2) if window['ideal'] != -1 else '--'
    #     smoothed_ideal = str(window['smoothed_ideal']).zfill(2) if 'smoothed_ideal' in window else ideal_window
    #     action = window['optimal_action']
    #     print(f"Day {day}, Close: {window['close']}, Min Window: {min_window}, Max Window: {max_window}, Ideal Window: {ideal_window}, Smoothed Ideal Window: {smoothed_ideal}, Optimal Action: {action}")
        
    # print("PnL: ", pnl)
    
    
