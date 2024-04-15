import math
import numpy as np
from .unit import Unit
from .trade import Trade
from config import print_trade_results, print_period_results


class TurtleTrader:
    """
    The Trader class keeps track of trading actions and the corresponding prices.
    It maintains a list of actions (buy, sell, hold) and a list of prices,
    and tracks PnL and open positions.
    """

    EXPAND = 0  # Expand the look-back period
    SHRINK = 1  # Shrink the look-back period
    NOTHING = 2  # Do nothing, keep the current look-back period

    BUY = 0  # Buy action // Not an actual action (Based on Turtle Trading Rules)
    SELL = 1  # Sell action  // Not an actual action (Based on Turtle Trading Rules)
    HOLD = 2  # Hold action  // Not an actual action (Based on Turtle Trading Rules)


    def __init__(self, stock='MSFT', base_period=20, expand_by=5, shrink_by=5, absolute_min=1, absolute_max=252, account_value=1000000, dollars_per_point=1, dynamic_exit=False):
        """
        Initializes the Trader object.

        Args:
            base_period (int): Base period for the rolling high calculation.
            expand_by (int): Number of days to expand the look-back period.
            shrink_by (int): Number of days to shrink the look-back period.
            absolute_min (int): The minimum look-back period allowed.
            absolute_max (int): The maximum look-back period allowed.
            account_equity (float): The initial amount of money in the account.
            dollars_per_point (float): The amount of money gained or lost per point of price movement.
            dynamic_exit (bool): Whether to use a dynamic exit period based on the current look-back period.
        """
        self.symbol = stock
        
        # Price Variables --------------------------------
        self.dates = []
        
        self.account_dollars = account_value
        self.dollars_per_point = dollars_per_point
        
        self.close_price_list = []
        self.high_price_list = []
        self.low_price_list = []
        self.next_open_price_list = []
        self.rolling_high_list = []
        self.rolling_low_list = []
        self.true_range_list = []
        self.N_list = []
        
        # Since the formula requires a previous N value, we must start with 20-day simple average
        self.initialized_n = False
        
        self.action_list = []
        self.trade_action_list = []
        self.open_positions = {'long': [], 'short': []} # Shouldn't short in strategy
        self.account_value_list = []
        self.account_equity_list = []
        self.account_dollars_list = []
        self.units_traded_list = []
        self.period_list = [base_period]
        self.exit_period_list = [math.floor(base_period / 2)] if dynamic_exit else [10]
        self.completed_trades = []
        
        
        # Environment/Agent specific variables ------------
        self.current_period = base_period
        self.base_period = base_period
        self.expand_by = expand_by
        self.shrink_by = shrink_by
        self.absolute_min = absolute_min
        self.absolute_max = absolute_max

        self.dynamic_exit = dynamic_exit
        self.exit_period = math.floor(self.current_period / 2)
        # self.exit_period = 10 # Subject to change in future versions
        
        self.current_step = -1

        # Metrics ----------------------------------------
        self.pnl = 0
        self.pnl_pct = 0
        self.total_units_traded = 0
        self.last_longest_trade_duration = 0
        self.last_average_pnl = None
        self.last_average_pnl_pct = None
        self.last_local_pnl = None
        self.last_local_pnl_pct = None
        self.last_positions_count = 0
        
        
        # For Interval rewards
        self.last_below_X_day_low = -11
        self.last_below_10_day_low = -11
        self.last_exit_period = -1
                        
        
    def step(self, date, price, high_price, low_price, next_open_price):
        """
        Updates the trader state for a new time step with the given price.

        Args:
            price (float): The current price of the stock.
        """
            
        if len(self.high_price_list) < self.current_period and len(self.high_price_list) > 0:
            historical_max = max(self.high_price_list)
        elif len(self.high_price_list) > 0:
            historical_max = max(self.high_price_list[-self.current_period:])
        else:
            historical_max = high_price
        self.rolling_high_list.append(historical_max)

        if len(self.low_price_list) < self.exit_period and len(self.low_price_list) > 0:
            historical_min = min(self.low_price_list)
        elif len(self.low_price_list) > 0:
            historical_min = min(self.low_price_list[-self.exit_period:])
        else:
            historical_min = low_price
        self.rolling_low_list.append(historical_min)

        self.dates.append(date)
        self.close_price_list.append(price)
        self.high_price_list.append(high_price)
        self.low_price_list.append(low_price)
        self.next_open_price_list.append(next_open_price)
        self._update_true_range(high_price, low_price)
        self._update_N()
        
        total_value = self._calculate_account_value(price)
        self.account_value_list.append(total_value)
        self.account_equity_list.append(total_value - self.account_dollars)
        self.account_dollars_list.append(self.account_dollars)
        self.units_traded_list.append(self.total_units_traded)
        

        self.current_step += 1
        
        
    def _calculate_account_value(self, current_price):
        """
        Calculate the account value based on the sum of current equity and the value of all open positions.
        """
        total_value = self.account_dollars
        for unit in self.open_positions['long']:
            unit_value = current_price * unit.num_shares
            total_value += unit_value
            
        return total_value
        
        

    def _update_true_range(self, high_price, low_price):
        """
        Calculate and store the latest true range: calculated as per turtle trading rules.
        """
        previous_close = self.close_price_list[-2] if len(self.close_price_list) > 1 else self.close_price_list[-1]
        true_range = max(high_price - low_price, abs(high_price - previous_close), abs(previous_close - low_price))
        self.true_range_list.append(true_range)
        if len(self.true_range_list) > 20:
            self.true_range_list.pop(0)

    def _update_N(self):
        """
        Calculate and store the latest N value: calculated as per turtle trading rules.
        """
        if len(self.true_range_list) == 20:
            if not self.initialized_n:
                current_N = np.mean(self.true_range_list)
                self.initialized_n = True
            else:
                current_N = (19 * self.N_list[-1] + self.true_range_list[-1]) / 20 if self.N_list[-1] is not None else None
            self.N_list.append(current_N)
        else:
            self.N_list.append(None)
            
    def _calculate_shares_to_trade(self, current_price):
        if not self.N_list or self.N_list[-1] is None:
            raise ValueError("N_list is empty or contains None. Should never get here :(")
        
        current_N = self.N_list[-1]
        
        # In the turtle rules example with small N it usually gets cancelled out with the Dollar's
        # per point. I've looked through a couple other implementations and there's no clear
        # cut way to deal with this issue, so I'm going to make a solution that I think
        # fits with the problem.
        
        # E.X. If N is < 1 then position size would scale up.
        
        # So I'm going to set a hard limit based on a max position size 
        # which will be 1.75% of the account
        units = (0.01 * self.account_dollars) / current_N
        units = min(units, .0175 * self.account_dollars)
        
        num_shares = units / current_price
        
        if num_shares < 1:
            return 1
        
        return math.floor(num_shares)


    def is_valid_action(self, action):
        """
        Determines if the specified action is valid based on the current state and trading rules.

        Args:
            action (int): The trading action to validate.

        Returns:
            bool: True if the action is valid, False otherwise.
        """
        # Check if action is 0, 1, 2
        if action not in [self.EXPAND, self.SHRINK, self.NOTHING]:
            return False
        
        return True
    

    def action(self, action):
        """
        Executes the given trading action, updates the action list, and manages open positions.
        It also calculates and updates the PnL and PnL% based on the action.

        Args:
            action (int): The trading action to execute.
        """
        
        if action == self.EXPAND:
            self.current_period = min(self.current_period + self.expand_by, self.absolute_max)
        elif action == self.SHRINK:
            self.current_period = max(self.current_period - self.shrink_by, self.absolute_min)
            
        self.period_list.append(self.current_period)

        if self.dynamic_exit:
            self.exit_period = math.floor(self.current_period / 2)
            
        self.exit_period_list.append(self.exit_period)

        # I update here, because although computationally ineffective the graphing unit
        # requires for all lists to be the same length. Doesn't look the best, but
        # has to due to Gymnasium rendering functionality
        if action == self.EXPAND or action == self.SHRINK:
            self._update_rolling_both()

        if print_period_results:
            print(f'Current period: {self.current_period}, Exit period: {self.exit_period}')
            
        self.action_list.append(action)

        current_date = self.dates[-1]
        current_price = self.close_price_list[-1]
        next_open_price = self.next_open_price_list[-1]

        trade_action_taken = False
        sell_action_taken = False
        
        # Check for stop loss hits
        # ToDo: Uncomment this in future versions
        # positions_to_close = [unit for unit in self.open_positions['long'] if current_price <= unit.stop_loss_level]
        # if positions_to_close:
        #     self._close_multiple_positions(positions_to_close)
        #     trade_action_taken = True
        #     sell_action_taken = True
        #     self.trade_action_list.append(self.SELL)

        # Check for buy signal if there are no open positions.
        if len(self.close_price_list) >= self.base_period:        # Using base_period as N is calculated with base_period=20
            historical_max = self.rolling_high_list[-1]           # In future versions N might be dynamically updated based 
            if current_price > historical_max:                    # on the current period
                
                current_N = self.N_list[-1]
                num_shares = self._calculate_shares_to_trade(current_price)
                stop_loss_level_for_new_unit = current_price - 2 * current_N
                                
                new_unit = Unit(pos_type='long',
                enter_price=current_price,
                enter_price_open=next_open_price,
                start_step=self.current_step,
                start_date=current_date,
                num_shares=num_shares,
                stop_loss_level=stop_loss_level_for_new_unit,
                current_period=self.current_period)
                                
                self.account_dollars -= (num_shares * current_price)
                self.open_positions['long'].append(new_unit)
                self.trade_action_list.append(self.BUY)
                trade_action_taken = True
                
                if print_trade_results:
                    print(f"Adding new unit: {new_unit} at Date: {self.dates[-1]}")
                    
                
                
                # ToDo: FIGURE OUT WHETHER TO KEEP OR REMOVE THIS ------------------------------
                self._adjust_stop_losses_for_new_unit(current_N, position_type='long')
               
                

        # Check for sell signal if there are open positions
        if self.open_positions.get('long') and len(self.open_positions['long']) > 0 and len(self.close_price_list) >= self.exit_period + 1:
            historical_min = self.rolling_low_list[-1]
            if current_price < historical_min:
                self.close_all_positions()
                
                # So we don't append SELL twice
                if not sell_action_taken:
                    self.trade_action_list.append(self.SELL)
                    sell_action_taken = True
                    
                trade_action_taken = True

        if not trade_action_taken:
            self.trade_action_list.append(self.HOLD)
            
            
    def _update_rolling_both(self):
        """
        Updates the rolling high and low lists based on the current period and exit period.
        Suboptimal, but necessary for graphing purposes. 
        Also must not include the most recent value.
        """
        if len(self.high_price_list) < self.current_period and len(self.high_price_list) > 0:
            historical_max = max(self.high_price_list)
        elif len(self.high_price_list) > 0:
            historical_max = max(self.high_price_list[-self.current_period-1:-1])
        else:
            raise ValueError("High price list is empty. Should never get here :(")
        self.rolling_high_list[-1] = historical_max

        if len(self.low_price_list) < self.exit_period and len(self.low_price_list) > 0:
            historical_min = min(self.low_price_list)
        elif len(self.low_price_list) > 0:
            historical_min = min(self.low_price_list[-self.exit_period-1:-1])
        else:
            raise ValueError("High price list is empty. Should never get here :(")
        self.rolling_low_list[-1] = historical_min
        

    def _adjust_stop_losses_for_new_unit(self, new_unit_N, position_type='long'):
        """
        Adjusts the stop losses for all units in a position when a new unit is added.
        The stop losses are set to 2N from the most recently added unit.
        
        Args:
            new_unit_N (float): The N value for the most recently added unit.
            position_type (str): The type of position ('long' or 'short').
        """
        if position_type not in self.open_positions:
            raise ValueError(f"Position type {position_type} is not recognized.")
        
        if position_type == 'long':
            if self.open_positions[position_type]:
                most_recent_unit = self.open_positions[position_type][-1]
                # print(f"Most recent unit: {most_recent_unit}")
                new_stop_loss_level = most_recent_unit.enter_price_open - (2 * new_unit_N)
                for unit in self.open_positions[position_type]:
                    # print(f"Old stop loss for unit: {unit} is {unit.stop_loss_level}")
                    unit.update_stop_loss_level(new_stop_loss_level)
                    # print(f"Updated stop loss for unit: {unit} to {new_stop_loss_level}")


    def _close_multiple_positions(self, positions_to_close):
        """
        Closes multiple specified trading positions and updates PnL and PnL% accordingly.

        Args:
            positions_to_close (list): A list of Unit objects representing the positions to close.
            closing_price (float): The price at which the positions are closed.
        """
        if not positions_to_close:
            raise ValueError("No positions specified for closing.")

        local_pnl = 0 
        local_pnl_pct = 0
        trades_closed = 0
        longest_trade_duration = 0
        
        current_price = self.close_price_list[-1]
        next_open_price = self.next_open_price_list[-1]
        current_date = self.dates[-1]

        for unit in positions_to_close:
            if unit not in self.open_positions[unit.pos_type]:
                raise ValueError(f"Unit {unit} is not in the list of open positions.")
            
            # print(f"Single Unit PnL: {closing_price - unit.enter_price}")
            # print(f"Single Unit PnL%: {((closing_price / unit.enter_price) - 1) * 100}")
            
            pnl = (next_open_price - unit.enter_price_open) * unit.num_shares
            pnl_pct = ((next_open_price - unit.enter_price_open) / unit.enter_price_open) * 100
            trade_duration = self.current_step - unit.start_step
            self.account_dollars += (unit.num_shares * next_open_price)
            local_pnl += pnl
            local_pnl_pct += pnl_pct
            trades_closed += 1
            longest_trade_duration = max(longest_trade_duration, trade_duration)
            self.open_positions[unit.pos_type].remove(unit)
                                    
            new_trade = Trade(symbol=self.symbol,
                              start_date=unit.start_date,
                              end_date=current_date,
                              enter_current_period=unit.current_period,
                              exit_current_period=self.current_period,
                              enter_price=unit.enter_price,
                              enter_price_open=unit.enter_price_open,
                              exit_price=current_price,
                              exit_price_open=next_open_price,
                              pos_type=unit.pos_type,
                              leverage=1,
                              num_shares=unit.num_shares,
                              PnL=pnl,
                              PnL_percent=pnl_pct)
            
            self.completed_trades.append(new_trade)

            if print_trade_results:
                print(f"SL Closed position: Date={self.dates[-1]},PnL={pnl}, PnL%={pnl_pct}, Trade Duration={trade_duration} steps, Num Shares={unit.num_shares}, Enter Price={unit.enter_price}, Exit Price={current_price}")

        # Update metrics after specified positions have been closed
        if trades_closed > 0:
            self.last_average_pnl = local_pnl / trades_closed
            self.last_average_pnl_pct = local_pnl_pct / trades_closed
            self.last_local_pnl = local_pnl
            self.last_local_pnl_pct = local_pnl_pct
            self.last_positions_count = trades_closed
            self.last_longest_trade_duration = longest_trade_duration
            
            self.total_units_traded += trades_closed
            self.pnl += local_pnl
            self.pnl_pct += local_pnl_pct
            
            if print_trade_results:
                print(f"Last Average PnL: {self.last_average_pnl}, Last Average PnL%: {self.last_average_pnl_pct}, Last Total PnL: {self.last_local_pnl}, Last Total PnL%: {self.last_local_pnl_pct}, Last Positions Count: {self.last_positions_count}, Last Longest Trade Duration: {self.last_longest_trade_duration}, Total Units Traded: {self.total_units_traded}, Total PnL: {self.pnl}, Total PnL%: {self.pnl_pct}")
        else:
            print("No trades were closed.")
 
            
    def close_all_positions(self):
        """
        Closes all open trading positions and updates PnL and PnL% accordingly.
        """
        if len(self.open_positions['long']) == 0:
            return

        current_price = self.close_price_list[-1]
        next_open_price = self.next_open_price_list[-1]
        current_date = self.dates[-1]
        
        local_pnl = 0
        local_pnl_pct = 0
        trades_closed = 0
        longest_trade_duration = 0

        while self.open_positions['long']:
            unit = self.open_positions['long'].pop(0)
            pnl = (next_open_price - unit.enter_price_open) * unit.num_shares
            pnl_pct = ((next_open_price - unit.enter_price_open) / unit.enter_price_open) * 100
            trade_duration = self.current_step - unit.start_step
            self.account_dollars += (unit.num_shares * next_open_price)
            local_pnl += pnl
            local_pnl_pct += pnl_pct
            trades_closed += 1
            longest_trade_duration = max(longest_trade_duration, trade_duration)            
            
            new_trade = Trade(symbol=self.symbol,
                              start_date=unit.start_date,
                              end_date=current_date,
                              enter_current_period=unit.current_period,
                              exit_current_period=self.current_period,
                              enter_price=unit.enter_price,
                              enter_price_open=unit.enter_price_open,
                              exit_price=current_price,
                              exit_price_open=next_open_price,
                              pos_type=unit.pos_type,
                              leverage=1,
                              num_shares=unit.num_shares,
                              PnL=pnl,
                              PnL_percent=pnl_pct)
            
            self.completed_trades.append(new_trade)
            
            
            if print_trade_results:
                print(f"EC Closed position: Date={self.dates[-1]}, PnL={pnl}, PnL%={pnl_pct}, Trade Duration={trade_duration} steps, Num Shares={unit.num_shares}, Enter Price={unit.enter_price}, Exit Price={current_price}")

        # Update metrics after all positions have been closed
        if trades_closed > 0:
            self.last_average_pnl = local_pnl / trades_closed
            self.last_average_pnl_pct = local_pnl_pct / trades_closed
            self.last_local_pnl = local_pnl
            self.last_local_pnl_pct = local_pnl_pct
            self.last_positions_count = trades_closed
            self.last_longest_trade_duration = longest_trade_duration
            
            self.total_units_traded += trades_closed
            self.pnl += local_pnl
            self.pnl_pct += local_pnl_pct
            
            if print_trade_results:
                print(f"Last Average PnL: {self.last_average_pnl}, Last Average PnL%: {self.last_average_pnl_pct}, Last Total PnL: {self.last_local_pnl}, Last Total PnL%: {self.last_local_pnl_pct}, Last Positions Count: {self.last_positions_count}, Last Longest Trade Duration: {self.last_longest_trade_duration}, Total Units Traded: {self.total_units_traded}, Total PnL: {self.pnl}, Total PnL%: {self.pnl_pct}")
        else:
            raise ValueError("Trades closed is 0. Should never get here :(")
        
        
    def is_price_below_X_days_low(self):
        """
        Determines if the previous price is below the X-day low.
        It is called after get next price so have to adjust indexes accordingly
  `      """
        if len(self.low_price_list) < self.exit_period + 1:
            return False

        # I have to index -2 because in the step before returning we are already getting the next price
        current_price = self.close_price_list[-2]
        historical_min = min(self.low_price_list[-2 - self.exit_period:-2]) 

        # Check if at least 10 days have passed since the last occurrence
        if self.current_step - 1 - self.last_below_X_day_low < self.exit_period + 1:
            return False

        # If current price is below the 10-day low, update the last occurrence day
        if current_price < historical_min:
            self.last_below_X_day_low = self.current_step - 1  # Adjust for index in the list
            return True


        return False        
    def is_price_below_10_days_low(self):
        """
        Determines if the previous price is below the 10-day low.
        It is called after get next price so have to adjust indexes accordingly
  `      """
        if len(self.low_price_list) < 11:
            return False

        # I have to index -2 because in the step before returning we are already getting the next price
        current_price = self.close_price_list[-2]
        historical_min = min(self.low_price_list[-12:-2])

        # Check if at least 10 days have passed since the last occurrence
        if self.current_step - 1 - self.last_below_10_day_low < 11:
            return False

        # If current price is below the 10-day low, update the last occurrence day
        if current_price < historical_min:
            self.last_below_10_day_low = self.current_step - 1  # Adjust for index in the list
            return True

        return False
