import pandas as pd
from turtle_solver import turtleSolver, calculate_reward

df = pd.read_csv('../price_movement/actual/data/msft.csv')
def process_data(df, min_lookback=15, max_lookback=25, max_change=3):
    pre_prices = []
    previous_segment_prices = []
    segment_prices = []
    agent_windows = []
    agent_actions = []
    first_exit = True
    starting_index = 0
    
    
    balance = 10000000
    total_pnl_dollars = 0
    total_reward = 0
    

    for index, row in df.iterrows():
        segment_prices.append({"High": row['High'], "Low": row['Low'], "Close": row['Close']})
        agent_windows.append(20)
        agent_actions.append("Nothing")
        if len(segment_prices) >= 11:  # Ensure there are enough points to check for the 10-day low
            ten_day_low = min(segment_prices[-11:-1], key=lambda x: x['Low'])['Low']
            if row['Close'] < ten_day_low:
                # Call turtleSolver for the segment
                if not first_exit:
                    # Exclude the first data point for subsequent segments if it's not the first exit
                    pre_prices = previous_segment_prices[-max_lookback:]
                    
                lookback_windows, pnl_dollars, pnl_pct = turtleSolver(starting_index, pre_prices, segment_prices, min_lookback, max_lookback, max_change, .8, first_exit, balance)
                rewards, _ = calculate_reward(lookback_windows, agent_actions, agent_windows)
                
                total_reward += sum(rewards)
                
                total_pnl_dollars += pnl_dollars
                balance += pnl_dollars
                
                starting_index = index + 1

                
                print("#" * 50)
                for i, window in enumerate(lookback_windows):
                    day = str(window['day']).zfill(2)
                    min_window = str(window.get('adjusted_min', window['min'])).zfill(2)  # Use adjusted_min if available
                    max_window = str(window.get('adjusted_max', window['max'])).zfill(2)  # Use adjusted_max if available
                    ideal_window = str(window['ideal']).zfill(2) if window['ideal'] != -1 else '--'
                    smoothed_ideal = str(window['smoothed_ideal']).zfill(2) if 'smoothed_ideal' in window else ideal_window
                    action = window['optimal_action']
                    print(f"Day {day}, Close: {window['close']:.2f}, Min Window: {min_window}, Max Window: {max_window}, Ideal Window: {ideal_window}, Smoothed Ideal Window: {smoothed_ideal}, Reward: {rewards[i]:5.2f}, O Action: {action:13}, Action: {agent_actions[i]}, Window: {agent_windows[i]}")
                print("#" * 50, pnl_dollars, "$", pnl_pct, "%")

                # return
                # Process the lookback_windows if needed, then reset for the next segment
                previous_segment_prices.extend(segment_prices)
                segment_prices = []
                agent_windows = []
                agent_actions = []
                first_exit = False
                

    print("Total PnL:", total_pnl_dollars, "$")
    print("Total Reward:", total_reward)

# Execute the function
process_data(df)