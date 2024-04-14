# Turtle environment for OpenAI Gymnasium

# State Spaces:
#   Basic w/ scale: Previous Prices Ranked, then scaled betweeen [0, 1]

# Action Spaces:
#   0: Expand the look-back period
#   1: Shrink the look-back period
#   2: Do nothing, keep the current look-back period

# Reward Spaces:
#   Interval: Rewards are calculated every time a trade is exited

# Notes:
# Initial Look-back Period is 20 as per the Turtle Trading Strategy
# Buy/Sell signals are generated based on the look-back period
# This environment only allows longs trades

import time
import warnings
import gymnasium as gym
from typing import Optional
from .turtle_controller import TurtleController

# Env returns obvs sometimes of length less than normal due to it being the beginning of the episode
warnings.filterwarnings("ignore", message=".*is not within the observation space.*")

class TurtleTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    default_params = {
    'state_type': 'Basic',
    'reward_type': 'Interval',
    'price_movement_type': 'Actual MSFT',
    'num_prev_obvs': 10,
    'offset_scaling': True,
    'scale': True,
    'starting_price': 100,
    'num_steps': 'Dynamic',
    'base_period': 20,
    'expand_by': 5,
    'shrink_by': 5,
    'absolute_min': 1,
    'absolute_max': 252,
    'account_value': 1000000,
    'dollars_per_point': 1,
    'render': False,
    'dynamic_exit': True,
    }
    
    def __init__(self, render_mode: Optional[str] = None, **overrides):
        self.params = self.default_params.copy()
        self.params.update(overrides['override_params'])
        
                
        if render_mode is not None:
            self.params['render'] = True
        
        # Initializes the Turtle Controller: handles state/action tracking, price generation, and rendering
        self.controller = TurtleController(**self.params)
        
        # Set the render mode
        self.render_mode = render_mode
        
        # Set the action and observation spaces
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = self.controller.get_observation_space()
        
        # Get starting state
        self.state = self.controller.get_state()
        
        # Store previous reward
        self.prev_reward = 0
        
        
        # Store for metrics
        self.period_history = []
        self.reward_history = []
        
        
        
    
    def step(self, action):
        self.state, reward, done, truncated, info = self.controller.step(action)
        if self.render_mode == "human":
            self.render()
            
        # Store metrics
        self.period_history.append(self.controller.trader.current_period)
        self.reward_history.append(reward)
        
        return self.state, reward, done, truncated, info
            
            
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.period_history = []
        self.reward_history = []
        self.controller = TurtleController(**self.params)
        self.state = self.controller.get_state()
        return self.state, {}
    
    
    def render(self):
        if self.render_mode == "human":
            self.controller.render()
        else:
            gym.logger.warn(
                "TurtleTradingEnv is not configured to render in non-human mode."
                "You can change the render mode by passing 'human' to the constructor."
            )
            
            
    def close(self):
        if self.controller.render_graph:
            self.controller.graph.close_window()
            
            
    def get_metrics(self):
        return {
            'current_period': self.controller.trader.current_period,
            'period_history': self.period_history,
            'reward_history': self.reward_history,
            'mean_period': sum(self.period_history) / len(self.period_history),
            'median_period': sorted(self.period_history)[len(self.period_history) // 2],    
            'total_pnl_percent': self.controller.trader.pnl_pct,
            'total_pnl': self.controller.trader.pnl,
            'total_units_traded': self.controller.trader.total_units_traded,
        }
            
            
        

