# Config for params for turtle env and agent
# To run the specific agent file cd into agents and run the corresponding agent


# Project dir. Change on your machine
project_dir = 'C:\\Users\\theal\\PycharmProjects\\waipons\\src\\algorithms\\gamePlayers\\turtlePlayer'

"""
Notes:

4. Increased memory, increased discount_factor, increased learning_rate
5. Increased exploration decay
6. Decreased MIN, Expand_by
7. Dynamic Exit, BaseAgent
8. Dyanmic Exit, DQN INterval Agent
9. Same as 8, but with 10 train episodes
10. Added exploration over time and fixed reward over time
11. Testing with DQN Interval
12. Running 30 times with DQN Interval
13. Trying to get a big PnL$ to observe, 110000$
14. Sames as 13, but 310000$
15. Same as 13, barely wins but destroys base turtle in 2017
16. New reward func for dynamic exit
17. No exploration
18. Using 10days for reward calc
19. New Log system! Using 10 days exit reward func
20. Using X days exit reward func
21. Making sure deleting stuff didn't break anything
22. 50 periods with cleaned code
23. Base Agent
24. Increased prev obvs
25. Basic with positions
26. Optimizer saved as self.optimizer
27. Running again but with 100
28. Trying not as self
29. Memory size decrease
30. Memory size increase
31. Back to normal memory size, decrease discount factor -- Very good
32. Increase learning rate
33. Back to normal learning rate, decrease expand and shrink by
34. Trying on John Deere
35. Base agent DE
36. Base agent APPL
37. APPL DQN
38. NVDA DQN -- USE
39. NVDA Base
40. MSFT DQN -- USE
41. MSFT Base
42. Ford Base
43. Ford DQN -- 2009 to 2010 THE BIGGEST COOK, but exploration still going down
44. Target Base
45. Target DQN -- Somewhat interesting
46. FST Base
47. FST DQN -- Not enough data
48. Macys base
49. Macys DQN - First non-explore win, good example for crab markets
50. NFLX base
51. NFLX DQN
52. NFLX Trying decrreased learn rate - KEEP, 51 BAD example, 22 Good Example: Big win here too
53. Same learn rate as above and increased batch size - NO KEEP
54. Increase expand and shink by from 5 to 1 - NO KEEP
55. Try max period 40 - NO KEEP
56. Offset scaling - Negligent - NO KEEP
57. Coke BASE
58. Coke DQN -- First non-explore win, rest are good comparisons, Ep40 BAD example, EP91 GOOD example
59. ABBV Base
60. ABBV DQN -- Not enough time steps
61. CVX Base
62. CVX DQN, 96 Bad example slightly underperforms  98 Good example: Basically even
63. TESLA base
64. TESLA DQN, 87 Good Example not that good, 34 BAd somewhat interesting
65. GOOG base
66. Goog DQN - 92 Good, basically the same, 81 Bad same
67. AMZN base
68. AMZN DQN - Looses badly, but not in comparison phases EP 40 bad, 91 Good but loses
"""

# CRITICAL CRITCAL CRITICAL CRITICAL CRITICAL CRITICAL CRITICAL CRITICAL CRITICAL
render_episode = False # Needs to be set to True for render_wait_time to activate
render_wait_time = 0.0001 # Lower time makes render go faster
save_log = True # Save training log
plot_training_results = False # Plot training results

# DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
print_period_results = False # Prints period results to console
print_trade_results = False # Prints trade results to console
print_state = False # Prints state to console
print_rewards = False # Prints rewards to console


# AGENT AGENT AGENT AGENT AGENT AGENT AGENT AGENT AGENT AGENT AGENT AGENT AGENT
discount_factor = .5 # Discount factor
train_episodes = 100 # Number of episodes to train
exploration_decay = .997 # Exploration decay rate
learning_rate = .001 # Learning rate
batch_size = 64 # Batch size
memory_size = 512


# ENVIRONMENT ENVIRONMENT ENVIRONMENT ENVIRONMENT ENVIRONMENT ENVIRONMENT ENVIRONMENT
override_params = {
    'state_type': 'BasicWithPositions', # Others: 'BasicWithPositions'
    'price_movement_type': 'Actual AMZN', # Default is 'Actual MSFT'. Others: 'Actual AAPL', 'Actual GOOGL', 'Actual AMZN', 'Actual FB', 'Actual DE'
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