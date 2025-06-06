import yaml
import torch
import numpy as np
import threading
import time
from envs import OrderbookEnv
from model import ActorCritic
from gui.dom_visualizer import DOMVisualizerGUI, MarketData

class InteractiveEvaluator:
    """Interactive evaluation with GUI visualization"""
    
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create environment
        self.env = OrderbookEnv(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = ActorCritic(
            input_size=self.env.observation_space.shape[0],
            lstm_hidden_size=self.config['model']['lstm_hidden_size'],
            transformer_heads=self.config['model']['transformer_heads'],
            transformer_layers=self.config['model']['transformer_layers'],
            action_dim=self.env.action_space.n,
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create GUI
        self.gui = DOMVisualizerGUI()
        
        # Action names for display
        self.action_names = [
            "DO_NOTHING",
            "PASSIVE_LIMIT_BUY",
            "PASSIVE_LIMIT_SELL", 
            "CANCEL_PASSIVE_ORDERS",
            "MARKET_BUY",
            "MARKET_SELL",
            "CLOSE_ALL"
        ]
        
        self.running = False
    
    def run_episode(self, episode_steps: int = 1000, step_delay: float = 0.5):
        """Run a single episode with visualization"""
        obs, _ = self.env.reset()
        hidden = self.model.init_hidden(1, self.device)
        
        for step in range(episode_steps):
            if not self.running:
                break
            
            # Get agent action
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, _, _, _ = self.model.get_action_and_value(obs_tensor, hidden)
                action = action.item()
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Prepare market data for GUI
            market_data = MarketData(
                bids=self.env.current_bids.copy(),
                asks=self.env.current_asks.copy(),
                mid_price=self.env.mid_price,
                position=self.env.position,
                cash=self.env.cash,
                pnl=info['pnl'],
                step=step,
                last_action=self.action_names[action]
            )
            
            # Add passive order information
            if self.env.passive_order_manager.bid_order:
                bid_order = self.env.passive_order_manager.bid_order
                market_data.passive_bid_order = (bid_order.price, bid_order.size)
            
            if self.env.passive_order_manager.ask_order:
                ask_order = self.env.passive_order_manager.ask_order
                market_data.passive_ask_order = (ask_order.price, ask_order.size)
            
            # Update GUI
            self.gui.update_data(market_data)
            
            if done:
                print(f"Episode completed at step {step}")
                print(f"Final P&L: ${info['pnl']:.2f}")
                break
            
            time.sleep(step_delay)
    
    def start_evaluation(self, num_episodes: int = 1, step_delay: float = 0.5):
        """Start evaluation in a separate thread"""
        def evaluation_loop():
            self.running = True
            for episode in range(num_episodes):
                if not self.running:
                    break
                print(f"Starting episode {episode + 1}/{num_episodes}")
                self.run_episode(step_delay=step_delay)
                if episode < num_episodes - 1:
                    time.sleep(2)  # Pause between episodes
        
        self.eval_thread = threading.Thread(target=evaluation_loop, daemon=True)
        self.eval_thread.start()
    
    def run(self):
        """Run the interactive evaluator"""
        print("Starting Interactive DOM Visualizer...")
        print("The agent will begin trading automatically.")
        
        # Start evaluation
        self.start_evaluation(num_episodes=5, step_delay=0.3)
        
        # Run GUI (blocking)
        try:
            self.gui.run()
        finally:
            self.running = False