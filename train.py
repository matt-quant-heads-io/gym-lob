import yaml
import torch
from envs import OrderbookEnv
from model import PPOTrainer

def main():
    # Load configuration
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("🚀 Starting LOB Trading Agent Training with Frame Stacking")
    print("="*60)
    
    # Create environment
    env = OrderbookEnv('config/default.yaml')
    
    # Print environment info
    print(f"📊 Environment Configuration:")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.n}")
    print(f"   Sequence length: {env.sequence_length}")
    print(f"   Single frame size: {env.single_frame_size}")
    
    # Create trainer
    trainer = PPOTrainer(env, config)
    
    print(f"\n🧠 Model Configuration:")
    print(f"   Device: {trainer.device}")
    print(f"   LSTM hidden size: {config['model']['lstm_hidden_size']}")
    print(f"   Transformer heads: {config['model']['transformer_heads']}")
    print(f"   Transformer layers: {config['model']['transformer_layers']}")
    print(f"   Use positional encoding: {config['model'].get('use_positional_encoding', True)}")
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Total timesteps: {config['ppo']['total_timesteps']:,}")
    print(f"   Rollout steps: {config['ppo']['rollout_steps']}")
    print(f"   Batch size: {config['ppo']['batch_size']}")
    print(f"   Learning rate: {config['ppo']['learning_rate']}")
    print("="*60)
    
    # Train the model
    trainer.train(
        total_timesteps=config['ppo']['total_timesteps'],
        log_interval=10
    )
    
    # Save the trained model
    trainer.save_model('trained_model_stacked.pth')
    print("\n🎉 Training completed! Model saved to 'trained_model_stacked.pth'")

if __name__ == "__main__":
    main()