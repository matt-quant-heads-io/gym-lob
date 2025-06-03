# env/orderbook_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
from typing import Dict, Any, Tuple, Optional
from .utils import format_dom, PassiveOrderManager, generate_synthetic_orderbook

class OrderbookEnv(gym.Env):
    """
    Limit Order Book Trading Environment
    
    Observation space: (20, 6) DOM + position + passive orders = (122,)
    Action space: 7 discrete actions
    """
    
    # Action constants
    DO_NOTHING = 0
    PASSIVE_LIMIT_BUY_ORDER = 1
    PASSIVE_LIMIT_SELL_ORDER = 2
    CANCEL_PASSIVE_ORDERS = 3
    MARKET_BUY = 4
    MARKET_SELL = 5
    CLOSE_ALL = 6
    
    def __init__(self, config_path: str = "config/default.yaml"):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Environment parameters
        self.episode_length = self.config['env']['episode_length']
        self.tick_size = self.config['env']['tick_size'] 
        self.max_position = self.config['env']['max_position']
        self.transaction_cost = self.config['env']['transaction_cost']
        
        # Order parameters
        self.size_pct_of_level = self.config['order']['size_pct_of_level']
        self.passive_order_ticks_offset = self.config['order']['passive_order_ticks_offset']
        self.passive_order_expiry_steps = self.config['order']['passive_order_expiry_steps']
        
        # Spaces
        self.action_space = spaces.Discrete(7)
        # DOM (20x6) + position (1) + passive orders (2) = 122 features  
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(122,), dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.position = 0.0
        self.cash = 0.0
        self.passive_order_manager = PassiveOrderManager(self.passive_order_expiry_steps)
        
        # Market data
        self.current_bids = None
        self.current_asks = None
        self.mid_price = 100.0
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.position = 0.0
        self.cash = 0.0
        self.passive_order_manager = PassiveOrderManager(self.passive_order_expiry_steps)
        
        # Generate initial market data
        self._update_market_data()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Update market data (simulate market movement)
        self._update_market_data()
        
        # Process passive order expiry
        self._process_passive_order_expiry()
        
        # Execute action
        reward = self._execute_action(action)
        
        # Check for passive order fills
        reward += self._check_passive_order_fills()
        
        self.current_step += 1
        
        # Episode termination
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _update_market_data(self):
        """Update bid/ask data with some random walk"""
        # Simple random walk for mid price
        price_change = np.random.normal(0, 0.05)
        self.mid_price = max(90.0, min(110.0, self.mid_price + price_change))
        
        # Generate new orderbook around current mid price
        self.current_bids, self.current_asks = generate_synthetic_orderbook(
            self.mid_price, self.tick_size, 20
        )
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        # DOM representation (20x6 = 120 features)
        dom = format_dom(self.current_bids, self.current_asks, 20)
        dom_flat = dom.flatten()
        
        # Position (1 feature)
        position_array = np.array([self.position], dtype=np.float32)
        
        # Passive orders (2 features)
        passive_orders = self.passive_order_manager.get_passive_order_vector()
        
        # Concatenate all features
        obs = np.concatenate([dom_flat, position_array, passive_orders])
        
        return obs
    
    def _execute_action(self, action: int) -> float:
        """Execute the given action and return immediate reward"""
        reward = 0.0
        
        if action == self.DO_NOTHING:
            pass
        
        elif action == self.PASSIVE_LIMIT_BUY_ORDER:
            reward += self._place_passive_buy_order()
        
        elif action == self.PASSIVE_LIMIT_SELL_ORDER:
            reward += self._place_passive_sell_order()
        
        elif action == self.CANCEL_PASSIVE_ORDERS:
            self.passive_order_manager.cancel_all_orders()
        
        elif action == self.MARKET_BUY:
            reward += self._execute_market_buy()
        
        elif action == self.MARKET_SELL:
            reward += self._execute_market_sell()
        
        elif action == self.CLOSE_ALL:
            reward += self._close_all_positions()
        
        return reward
    
    def _place_passive_buy_order(self) -> float:
        """Place a passive buy order"""
        if len(self.current_bids) == 0:
            return -0.01  # Small penalty for invalid action
        
        # Use best bid as trigger level
        trigger_price = self.current_bids[0, 0]  # Best bid price
        order_price = trigger_price - (self.passive_order_ticks_offset * self.tick_size)
        
        # Calculate order size
        trigger_volume = self.current_bids[0, 2]  # Volume at best bid
        order_size = self.size_pct_of_level * trigger_volume
        
        success = self.passive_order_manager.place_bid_order(order_price, order_size)
        return 0.0 if success else -0.01
    
    def _place_passive_sell_order(self) -> float:
        """Place a passive sell order"""
        if len(self.current_asks) == 0:
            return -0.01
        
        # Use best ask as trigger level  
        trigger_price = self.current_asks[0, 0]  # Best ask price
        order_price = trigger_price + (self.passive_order_ticks_offset * self.tick_size)
        
        # Calculate order size
        trigger_volume = self.current_asks[0, 2]  # Volume at best ask
        order_size = self.size_pct_of_level * trigger_volume
        
        success = self.passive_order_manager.place_ask_order(order_price, order_size)
        return 0.0 if success else -0.01
    
    def _execute_market_buy(self) -> float:
        """Execute market buy order"""
        if len(self.current_asks) == 0:
            return -0.1
        
        # Use best ask
        ask_price = self.current_asks[0, 0]
        ask_volume = self.current_asks[0, 2]
        
        # Calculate order size
        order_size = self.size_pct_of_level * ask_volume
        
        # Check position limits
        if self.position + order_size > self.max_position:
            return -0.05
        
        # Execute trade
        trade_value = order_size * ask_price
        transaction_cost = trade_value * self.transaction_cost
        
        self.position += order_size
        self.cash -= (trade_value + transaction_cost)
        
        return -transaction_cost  # Immediate cost
    
    def _execute_market_sell(self) -> float:
        """Execute market sell order"""
        if len(self.current_bids) == 0:
            return -0.1
        
        # Use best bid
        bid_price = self.current_bids[0, 0]
        bid_volume = self.current_bids[0, 2]
        
        # Calculate order size
        order_size = self.size_pct_of_level * bid_volume
        
        # Check position limits
        if self.position - order_size < -self.max_position:
            return -0.05
        
        # Execute trade
        trade_value = order_size * bid_price
        transaction_cost = trade_value * self.transaction_cost
        
        self.position -= order_size
        self.cash += (trade_value - transaction_cost)
        
        return -transaction_cost  # Immediate cost
    
    def _close_all_positions(self) -> float:
        """Close all open positions"""
        if abs(self.position) < 1e-6:
            return 0.0
        
        if self.position > 0:
            # Sell to close long position
            if len(self.current_bids) == 0:
                return -0.1
            
            bid_price = self.current_bids[0, 0]
            trade_value = abs(self.position) * bid_price
            transaction_cost = trade_value * self.transaction_cost
            
            self.cash += (trade_value - transaction_cost)
            self.position = 0.0
            
            return -transaction_cost
        
        else:
            # Buy to close short position
            if len(self.current_asks) == 0:
                return -0.1
            
            ask_price = self.current_asks[0, 0]
            trade_value = abs(self.position) * ask_price
            transaction_cost = trade_value * self.transaction_cost
            
            self.cash -= (trade_value + transaction_cost)
            self.position = 0.0
            
            return -transaction_cost
    
    def _check_passive_order_fills(self) -> float:
        """Check if any passive orders should be filled"""
        reward = 0.0
        
        # Check bid order fill
        if self.passive_order_manager.bid_order:
            bid_order = self.passive_order_manager.bid_order
            # Simplified fill logic: if market traded at or below our bid price
            if len(self.current_bids) > 0 and self.current_bids[0, 0] <= bid_order.price:
                # Fill the order
                trade_value = bid_order.size * bid_order.price
                transaction_cost = trade_value * self.transaction_cost * 0.5  # Reduced cost for passive
                
                self.position += bid_order.size
                self.cash -= (trade_value + transaction_cost)
                self.passive_order_manager.bid_order = None
                
                reward += 0.01  # Small reward for successful passive fill
        
        # Check ask order fill
        if self.passive_order_manager.ask_order:
            ask_order = self.passive_order_manager.ask_order
            # Simplified fill logic: if market traded at or above our ask price
            if len(self.current_asks) > 0 and self.current_asks[0, 0] >= ask_order.price:
                # Fill the order
                trade_value = ask_order.size * ask_order.price
                transaction_cost = trade_value * self.transaction_cost * 0.5  # Reduced cost for passive
                
                self.position -= ask_order.size
                self.cash += (trade_value - transaction_cost)
                self.passive_order_manager.ask_order = None
                
                reward += 0.01  # Small reward for successful passive fill
        
        return reward
    
    def _process_passive_order_expiry(self):
        """Process passive order expiry"""
        self.passive_order_manager.update_and_expire()
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary"""
        return {
            'position': self.position,
            'cash': self.cash,
            'mid_price': self.mid_price,
            'step': self.current_step,
            'pnl': self.cash + self.position * self.mid_price,
            'passive_bid': self.passive_order_manager.bid_order is not None,
            'passive_ask': self.passive_order_manager.ask_order is not None
        }