# Gets actual price data from a csv file
# OHLC calculated from CRSP dataset 5 min prices.
#  https://github.com/lordyabu/CRSP-Lab

#

import sys
import pandas as pd

# Returns Close, High, Low
class TurtleActualPriceGetter:
    def __init__(self, stock_name):
        try:
            self.df = pd.read_csv(f"../price_movement/actual/data/{stock_name}.csv")
        except FileNotFoundError:
            self.df = pd.read_csv(f"./price_movement/actual/data/{stock_name}.csv")
        self.step = 0
        
        
    def get_num_steps(self):
        return len(self.df.index) - 1

    def get_next_price(self):
        if self.step >= len(self.df):
            raise IndexError("No more prices available")

        self.step += 1
        
        if self.step == len(self.df.index):
            return self.df['date'].iloc[self.step - 1],self.df['Close'].iloc[self.step - 1], self.df['High'].iloc[self.step - 1], self.df['Low'].iloc[self.step - 1], self.df['Open'].iloc[self.step - 1]
        else:
            return self.df['date'].iloc[self.step - 1],self.df['Close'].iloc[self.step - 1], self.df['High'].iloc[self.step - 1], self.df['Low'].iloc[self.step - 1], self.df['Open'].iloc[self.step]


if __name__ == '__main__':
    getter = TurtleActualPriceGetter('MSFT')
    print(getter.get_multiple_prices(5))
    print(getter.get_next_price())