from dataclasses import dataclass
    
@dataclass
class Unit:
    """
    Represents a trading unit(used in turtle terminology) with specific attributes related to a trading operation.

    This class is a data structure used to store information about a single trading unit, including its position type,
    entry price, start date and time, and optionally a list of previous prices.

    Attributes:
        pos_type (str): The position type of the trade ('long' or 'short').
        enter_price (float): The price at which the trade was entered.
        start_step (str): The start step of the trade.
        num_shares (int): The number of shares or contracts for this unit.
        stop_loss_level (float): The initial stop loss level for the unit.
    """
    pos_type: str
    enter_price: float
    enter_price_open: float
    start_step: int
    start_date: str
    num_shares: int
    current_period: int
    stop_loss_level: float = None
    
    def update_stop_loss_level(self, new_stop_loss_level):
        """Updates the stop loss level for this trading unit."""
        self.stop_loss_level = new_stop_loss_level
        
        

@dataclass
class UnitSimple:
    """
    Represents a simple trading unit(used in turtle terminology) with specific attributes related to a trading operation.

    This class is a data structure used to store information about a single trading unit, including its position type,
    entry price, start date and time, and optionally a list of previous prices.

    Attributes:
        pos_type (str): The position type of the trade ('long' or 'short').
        enter_price (float): The price at which the trade was entered.
        start_step (str): The start step of the trade.
    """
    pos_type: str
    enter_price: float
    start_step: int