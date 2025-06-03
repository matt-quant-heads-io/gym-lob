import yaml
import torch
from env import OrderbookEnv
from model import PPOTrainer

def main():
    # Load configuration
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = OrderbookEnv('config/default.yaml')
    
    # Create trainer
    trainer = PPOTrainer(env, config)
    
    print(f"Training on device: {trainer.device}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    print("-" * 50)
    
    # Train the model
    trainer.train(
        total_timesteps=config['ppo']['total_timesteps'],
        log_interval=10
    )
    
    # Save the trained model
    trainer.save_model('trained_model.pth')
    print("Training completed! Model saved to 'trained_model.pth'")


if __name__ == "__main__":
    main()

