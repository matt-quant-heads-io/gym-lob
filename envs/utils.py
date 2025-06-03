import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class PassiveOrder:
    """Represents a passive limit order"""
    side: str  # 'bid' or 'ask'
    price: float
    size: float
    steps_alive: int
    max_steps: int

class PassiveOrderManager:
    """Manages passive limit orders"""
    
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.bid_order: Optional[PassiveOrder] = None
        self.ask_order: Optional[PassiveOrder] = None
    
    def place_bid_order(self, price: float, size: float) -> bool:
        """Place a passive bid order. Returns True if successful."""
        if self.bid_order is None:
            self.bid_order = PassiveOrder('bid', price, size, 0, self.max_steps)
            return True
        return False
    
    def place_ask_order(self, price: float, size: float) -> bool:
        """Place a passive ask order. Returns True if successful."""
        if self.ask_order is None:
            self.ask_order = PassiveOrder('ask', price, size, 0, self.max_steps)
            return True
        return False
    
    def cancel_all_orders(self):
        """Cancel all passive orders"""
        self.bid_order = None
        self.ask_order = None
    
    def update_and_expire(self) -> Tuple[Optional[PassiveOrder], Optional[PassiveOrder]]:
        """Update order lifetimes and expire old orders. Returns expired orders."""
        expired_bid, expired_ask = None, None
        
        if self.bid_order:
            self.bid_order.steps_alive += 1
            if self.bid_order.steps_alive >= self.bid_order.max_steps:
                expired_bid = self.bid_order
                self.bid_order = None
        
        if self.ask_order:
            self.ask_order.steps_alive += 1
            if self.ask_order.steps_alive >= self.ask_order.max_steps:
                expired_ask = self.ask_order
                self.ask_order = None
        
        return expired_bid, expired_ask
    
    def get_passive_order_vector(self) -> np.ndarray:
        """Return [bid_size, ask_size] vector for observation"""
        bid_size = self.bid_order.size if self.bid_order else 0.0
        ask_size = self.ask_order.size if self.ask_order else 0.0
        return np.array([bid_size, ask_size], dtype=np.float32)

def format_dom(bids: np.ndarray, asks: np.ndarray, levels: int = 20) -> np.ndarray:
    """
    Format bid/ask data into DOM representation
    
    Args:
        bids: Array of [price, size, volume] sorted by descending price
        asks: Array of [price, size, volume] sorted by ascending price  
        levels: Number of price levels to include
    
    Returns:
        DOM array of shape (levels, 6): [bid_vol, bid_size, bid_price, ask_vol, ask_size, ask_price]
    """
    dom = np.zeros((levels, 6), dtype=np.float32)
    
    # Bid side (columns 0-2): volume, size, price (descending price order)
    bid_levels = min(len(bids), levels)
    if bid_levels > 0:
        dom[:bid_levels, 0] = bids[:bid_levels, 2]  # volume
        dom[:bid_levels, 1] = bids[:bid_levels, 1]  # size  
        dom[:bid_levels, 2] = bids[:bid_levels, 0]  # price
    
    # Ask side (columns 3-5): volume, size, price (ascending price order)
    ask_levels = min(len(asks), levels)
    if ask_levels > 0:
        dom[:ask_levels, 3] = asks[:ask_levels, 2]  # volume
        dom[:ask_levels, 4] = asks[:ask_levels, 1]  # size
        dom[:ask_levels, 5] = asks[:ask_levels, 0]  # price
    
    return dom

def generate_synthetic_orderbook(mid_price: float = 100.0, tick_size: float = 0.01, 
                                levels: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic bid/ask data for testing"""
    
    # Generate bid levels (descending from mid_price)
    bid_prices = np.arange(mid_price - tick_size, 
                          mid_price - (levels + 1) * tick_size, 
                          -tick_size)[:levels]
    bid_sizes = np.random.exponential(50, levels) + 10
    bid_volumes = bid_sizes * np.random.uniform(0.8, 1.2, levels)
    
    bids = np.column_stack([bid_prices, bid_sizes, bid_volumes])
    
    # Generate ask levels (ascending from mid_price)  
    ask_prices = np.arange(mid_price + tick_size,
                          mid_price + (levels + 1) * tick_size,
                          tick_size)[:levels]
    ask_sizes = np.random.exponential(50, levels) + 10  
    ask_volumes = ask_sizes * np.random.uniform(0.8, 1.2, levels)
    
    asks = np.column_stack([ask_prices, ask_sizes, ask_volumes])
    
    return bids, asks