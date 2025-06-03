# eval.py
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from env import OrderbookEnv
from model import ActorCritic


def evaluate_model(model_path: str, config_path: str, num_episodes: int = 10):
    """Evaluate trained model"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = OrderbookEnv(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ActorCritic(
        input_size=env.observation_space.shape[0],
        lstm_hidden_size=config['model']['lstm_hidden_size'],
        transformer_heads=config['model']['transformer_heads'],
        transformer_layers=config['model']['transformer_layers'],
        action_dim=env.action_space.n,
        dropout=config['model']['dropout']
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluation
    episode_rewards = []
    episode_lengths = []
    final_pnls = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        hidden = model.init_hidden(1, device)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _, _, _ = model.get_action_and_value(obs_tensor, hidden)
                action = action.item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        final_pnls.append(info['pnl'])
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.3f}, PnL={info['pnl']:.3f}, Length={episode_length}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Episodes: {num_episodes}")
    print(f"Mean Episode Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Mean Final PnL: {np.mean(final_pnls):.3f} ± {np.std(final_pnls):.3f}")
    print(f"Best Episode Reward: {np.max(episode_rewards):.3f}")
    print(f"Worst Episode Reward: {np.min(episode_rewards):.3f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].grid(True)
    
    # Final PnLs
    axes[1, 0].plot(final_pnls)
    axes[1, 0].set_title('Final PnL per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('PnL')
    axes[1, 0].grid(True)
    
    # Reward distribution
    axes[1, 1].hist(episode_rewards, bins=10, alpha=0.7)
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    evaluate_model('trained_model.pth', 'config/default.yaml', num_episodes=20)