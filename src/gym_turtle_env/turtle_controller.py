import math
import pygame
import numpy as np
from gymnasium import spaces
# from turtle_trader_simple import TurtleTrader # Remove in future versions
from .turtle_trader import TurtleTrader
from .turtle_graph import TurtleStockGraph
from .turtle_solver import turtleSolver, calculate_reward
from .turtle_solver_v2 import turtleSolverV2, calculate_reward_v2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Tuple, Union
from price_movement.actual.actual_price_getter import TurtleActualPriceGetter


class TurtleController:
    """
    The Controller class manages the interaction between the turtle trading environment and an agent.
    It handles the generation of price movements, tracks the state of the trading environment,
    executes actions, and computes rewards. This returns truncated as well(due to Gymnasium use Vs. Gym)
    """
    
    def __init__(self, state_type, reward_type, price_movement_type, num_prev_obvs, offset_scaling, scale, num_steps, base_period=20, expand_by=5, shrink_by=5, absolute_min=1, absolute_max=252, account_value=1000000, dollars_per_point=1, dynamic_exit=False,render=False, **kwargs):
        """
        Initializes the Controller object.

        Args:
            state_type (str): Type of state representation.
            reward_type (str): Type of reward calculation method.
            price_movement_type (str): Type of price movement model.
            num_prev_obvs (int): Number of previous observations to consider in the state.
            offset_scaling (bool): Whether to apply offset scaling to the state.
            scale (bool): Whether to scale the state.
            num_steps (int): Number of steps in an episode.
            base_period (int): Base period for the rolling high calculation.
            expand_by (int): Number of days to expand the look-back period.
            shrink_by (int): Number of days to shrink the look-back period.
            absoulute_min (int): Absolute minimum look-back period.
            absolute_max (int): Absolute maximum look-back period.
            account_value (int): Account equity for the TurtleTrader object.
            dollars_per_point (int): Dollars per point for the TurtleTrader object.
            dynamic_exit (bool): Whether to dynamically adjust the exit period based on actions.
            **kwargs: Additional keyword arguments.
        """

        # TRADER -----------------------------------------------------------------------------------------------------
        stock_name = price_movement_type.split(" ")[1]
        self.trader = TurtleTrader(stock_name, base_period, expand_by, shrink_by, absolute_min, absolute_max, account_value, dollars_per_point, dynamic_exit)
                      #  Initialize the TurtleTrader object. Will need to change if using TurtleSimple Turtle Trader

        
        # GRAPH ------------------------------------------------------------------------------------------------------        
        self.render_graph = render # Initialize the TurtleStockGraph object if rendering is enabled
        if self.render_graph:
            self.graph = TurtleStockGraph(800, 600, (0, 0, 0), self.trader.dynamic_exit, 50)
            
        # PRICE GENERATOR --------------------------------------------------------------------------------------------
        if "Actual" in price_movement_type: # Only Actual in CRSP dataset supported for now
            stock_name = price_movement_type.split(" ")[1]
            try:
                self.price_generator = TurtleActualPriceGetter(stock_name)
            except:
                raise ValueError("This price generation is not supported yet")
        else:
            raise ValueError("This price generation is not supported yet")
        
        assert(len(self.trader.close_price_list) == len(self.trader.action_list)), "Price and action list must be the same length"
        
        
        # ENVIRONMENT VARS ------------------------------------------------------------------------------------------------
        self.current_price = None
        self.step_count = -1 # Becomes 0 when _get_next_price() is called below
        self.state_type = state_type
        self.num_prev_obvs = num_prev_obvs 
        self.reward_type = reward_type
        
        if num_steps == "Dynamic": # Dynamic because not all stocks exist for same timespan
            self.num_steps = self.price_generator.get_num_steps()
        else:
            self.num_steps = num_steps        

        self.scale = scale
        self.offset_scaling = offset_scaling
        self.kwargs = kwargs
        self._get_next_price() # Increments step_count to 0

    def step(self, action):
        """
        Executes a trading action, updates the environment state, and calculates the reward.

        Args:
            action (int): The trading action to execute.

        Returns:
            tuple: Tuple containing the new state, reward, completion status, truncated, and additional info.

        Raises:
            ValueError: If an invalid action is attempted.
        """

        
        if not self.trader.is_valid_action(action): # If invalid action return truncated=True
            return self.get_state(), -100, False, True, {}

        self.trader.action(action) # Execute the action using the Trader subclass
                            
        if self.step_count + 1 < self.num_steps: # Check if this step should be the last
            self._get_next_price()                              # Returns False for done
            return self.get_state(), self.get_reward(is_complete=False), False, False, self.get_info()
        else:
            self.trader.close_all_positions()
            self._get_next_price()                             # Returns True for done
            return self.get_state(), self.get_reward(is_complete=True), True, False, self.get_info()

    def _get_next_price(self):
        """
        Generates the next price using the price movement model and updates the environment state.
        """

        new_date, new_close_price, new_high_price, new_low_price, new_next_open_price = self.price_generator.get_next_price()
        self.trader.step(new_date, new_close_price, new_high_price, new_low_price, new_next_open_price)
        self.current_price = new_close_price
        self.step_count += 1

    def get_state(self):
        """
        Retrieves the current state of the environment.

        Returns:
            np.ndarray: The current state of the environment.

        Raises:
            ValueError: If an unsupported state type is requested.
        """

        if self.state_type == 'Basic':
            return self._get_basic_state()
        elif self.state_type == 'BasicWithPositions':
            return self._get_basic_state_w_positions()
        else:
            raise ValueError(f"State type ({self.state_type}) not yet implemented")

    def get_reward(self, is_complete=False):
        """
        Calculates the reward based on the current state of the environment.

        Args:
            is_complete (bool): Flag indicating if the episode is complete.

        Returns:
            float: The calculated reward.
        """

        if self.reward_type == "Interval":
            return 0 # Rewards should be gotten from get_interval_reward when price < 10 days low
        else:
            raise ValueError(f"Reward type ({self.reward_type}) not yet implemented")
        
        
    def get_info(self):
        """
        Retrieves additional information about the environment state.

        Returns:
            dict: A dictionary containing additional information about the environment state.
        """

        if self.reward_type == 'Interval':
            return {
                'is_close_X': self.trader.is_price_below_X_days_low(),
                'is_close_10': self.trader.is_price_below_10_days_low(),
            }
        else:
            return {}

    def render(self):
        """
        Renders the environment state using Pygame and the TurtleStockGraph object.
        """
        if not self.graph.initialized:
            self.graph._initialize_window()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        if len(self.trader.action_list) > 1:
            self.graph.screen.fill(self.graph.background_color)
            if len(self.trader.close_price_list) > len(self.trader.action_list):
                self.graph.update_graph(self.trader.close_price_list[:-1], self.trader.trade_action_list, self.trader.rolling_high_list[:-1], self.trader.rolling_low_list[:-1],self.trader.action_list)
            else:
                raise ValueError("Given implementation shouldn't be here. Close Price list and Action list should be the same length")
            
            pygame.display.flip()

    def get_valid_actions(self):
        """
        Retrieves a list of valid actions based on the current state.

        Returns:
            list: A list of valid actions.
        """

        return [0, 1, 2] # Look in turtle_trader.py for the meaning of these actions

    def get_observation_space(self):
        """
        Get the observation space of the environment based on the state configuration.

        Returns:
            gym.spaces.Box: The observation space.
        """

        if self.state_type == 'Basic':
            if self.scale:
                return spaces.Box(low=-5, high=5, shape=(self.num_prev_obvs,), dtype=np.float32)
            else:
                low = 1
                high = self.num_prev_obvs
                return spaces.Box(low=low, high=high, shape=(self.num_prev_obvs,), dtype=int)
        elif self.state_type == 'BasicWithPositions':
            if self.scale:
                return spaces.Box(low=-5, high=5000, shape=(self.num_prev_obvs + 1,), dtype=np.float32)
            else:
                low = 1
                high = self.num_prev_obvs
                return spaces.Box(low=low, high=5000, shape=(self.num_prev_obvs + 1,), dtype=int)
        else:
            raise ValueError(f"State type ({self.state_type}) not yet implemented")


    # State representation methods --------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    
    def _get_basic_state(self):
        """
        Computes the basic state representation with left padding for insufficient data.
        """
        num_available_prices = min(self.step_count + 1, self.num_prev_obvs)
        prev_prices = self.trader.close_price_list[-num_available_prices:]
        ranks = np.argsort(np.argsort(prev_prices)) + 1

        # Padding because I need to keep observation space constant
        # but don't really want any agent to start trading right away
        # so I make the padded values very high and on the left
        if self.scale:
            ranks_reshaped = np.array(ranks).reshape(-1, 1)
            scaler = StandardScaler()
            scaled_ranks = scaler.fit_transform(ranks_reshaped).flatten()
            if self.offset_scaling:
                min_offset = self.kwargs.get('min_offset', 0.01)
                scaled_ranks += (min_offset * (1 - scaled_ranks))
            padded_value = 1
            ranks = scaled_ranks
        else:
            padded_value = self.num_prev_obvs

        pad_size = self.num_prev_obvs - len(ranks) # Calculate the padding size
        padded_state = np.pad(ranks, (pad_size, 0), 'constant', constant_values=padded_value) # Apply left padding
        return padded_state.astype(np.float32 if self.scale else int)

        
        
    def _get_basic_state_w_positions(self):
        """
        Computes the basic state representation with number of units held and left padding for insufficient data.
        """
        num_available_prices = min(self.step_count + 1, self.num_prev_obvs)
        prev_prices = self.trader.close_price_list[-num_available_prices:]
        ranks = np.argsort(np.argsort(prev_prices)) + 1

        # Only diference between this one and one above HERE
        num_long_positions = len(self.trader.open_positions['long'])
        position_feature = [num_long_positions]

        # Padding because I need to keep observation space constant
        # but don't really want any agent to start trading right away
        # so I make the padded values very high and on the left
        if self.scale:
            ranks_reshaped = np.array(ranks).reshape(-1, 1)
            scaler = StandardScaler()
            scaled_ranks = scaler.fit_transform(ranks_reshaped).flatten()
            if self.offset_scaling:
                min_offset = self.kwargs.get('min_offset', 0.01)
                scaled_ranks += (min_offset * (1 - scaled_ranks))
            padded_value = 1
            ranks = scaled_ranks
        else:
            padded_value = self.num_prev_obvs

    
        pad_size = self.num_prev_obvs - len(ranks) # Calculate the padding size
        padded_state = np.pad(ranks, (pad_size, 0), 'constant', constant_values=padded_value) # Apply left padding
        state_representation = np.concatenate((padded_state, position_feature)).astype(np.float32 if self.scale else int)

        return state_representation
    
    

        
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------    
    


    # Reward calculation methods ----------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
   
    def get_interval_reward_10(self, pre_prices: List[float], stock_prices: List[float], 
                            trader_actions: List[str], trader_windows: List[float], 
                            first_exit: bool) ->List[float]:
        
        change_size = (self.trader.expand_by + self.trader.shrink_by) / 2
        
        lookback_windows, pnl, pnl_pct = turtleSolver(starting_index=self.step_count, 
                pre_prices=pre_prices, stock_prices=stock_prices, 
                min_lookback=self.trader.absolute_min, max_lookback=self.trader.absolute_max, 
                max_change=change_size, weight_max_window=.8, first_exit=first_exit, 
                initial_balance=-1, verbose=False)
        
        rewards = calculate_reward(lookback_windows, trader_actions, trader_windows)
        return rewards     
        
    def get_interval_reward_X(self, pre_prices: List[float], stock_prices: List[float], 
                            trader_actions: List[str], trader_windows: List[float], 
                            first_exit: bool) ->List[float]:
        
        change_size = (self.trader.expand_by + self.trader.shrink_by) / 2
        
        lookback_windows, pnl, pnl_pct = turtleSolverV2(starting_index=self.step_count, 
                exit_period=self.trader.exit_period, pre_prices=pre_prices, stock_prices=stock_prices, 
                min_lookback=self.trader.absolute_min, max_lookback=self.trader.absolute_max, 
                max_change=change_size, weight_max_window=.8, first_exit=first_exit, 
                initial_balance=-1, verbose=False)
        
        rewards = calculate_reward_v2(lookback_windows, trader_actions, trader_windows)
        return rewards
        
        
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------

    def close(self):
        if self.graph and self.graph.initialized:
            pygame.quit()