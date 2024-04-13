from dataclasses import dataclass


@dataclass
class Trade:
    """
    Represents a Trade. Used to log trades. 
    """
    symbol: str
    start_date: str
    end_date: str
    enter_current_period: int
    exit_current_period: int
    enter_price: float
    enter_price_open: float
    exit_price: float
    exit_price_open: float
    pos_type: str
    leverage: 1
    num_shares: int
    PnL: float
    PnL_percent: float
