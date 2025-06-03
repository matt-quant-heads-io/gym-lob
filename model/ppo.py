import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import gymnasium as gym
from .actor_critic import ActorCritic

class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, buffer_size: int, obs_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Buffers
        self.observations = torch.zeros((buffer_size, obs_dim), device=device)
        self.actions = torch.zeros((buffer_size,), dtype=torch.long, device=device)
        self.rewards = torch.zeros((buffer_size,), device=device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=device)
        self.log_probs = torch.zeros((buffer_size,), device=device)
        self.values = torch.zeros((buffer_size,), device=device)
        self.advantages = torch.zeros((buffer_size,), device=device)
        self.returns = torch.zeros((buffer_size,), device=device)
    
    def add(self, obs: np.ndarray, action: int, reward: float, done: bool, 
            log_prob: float, value: float):
        """Add experience to buffer"""
        self.observations[self.ptr] = torch.FloatTensor(obs).to(self.device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute returns and advantages using GAE"""
        with torch.no_grad():
            # Convert to numpy for easier computation
            rewards = self.rewards[:self.size].cpu().numpy()
            values = self.values[:self.size].cpu().numpy()
            dones = self.dones[:self.size].cpu().numpy()
            
            # Add last value for bootstrapping
            values = np.append(values, last_value)
            
            advantages = np.zeros_like(rewards)
            gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = values[t + 1]
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = values[t + 1]
                
                delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
                gae = delta + gamma * gae_lambda * next_non_terminal * gae
                advantages[t] = gae
            
            returns = advantages + values[:-1]
            
            # Convert back to tensors
            self.advantages[:self.size] = torch.FloatTensor(advantages).to(self.device)
            self.returns[:self.size] = torch.FloatTensor(returns).to(self.device)
            
            # Normalize advantages
            self.advantages[:self.size] = (self.advantages[:self.size] - self.advantages[:self.size].mean()) / (self.advantages[:self.size].std() + 1e-8)
    
    def get_batches(self, batch_size: int):
        """Get random batches of data"""
        indices = torch.randperm(self.size, device=self.device)
        
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            batch_indices = indices[start:end]
            
            yield {
                'observations': self.observations[batch_indices],
                'actions': self.actions[batch_indices], 
                'log_probs': self.log_probs[batch_indices],
                'values': self.values[batch_indices],
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices]
            }

class PPOTrainer:
    """PPO Trainer for the Actor-Critic model"""
    
    def __init__(self, env: gym.Env, config: Dict):
        self.env = env
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # PPO hyperparameters
        self.learning_rate = config['ppo']['learning_rate']
        self.clip_param = config['ppo']['clip_param']
        self.entropy_coeff = config['ppo']['entropy_coeff']
        self.value_loss_coeff = config['ppo']['value_loss_coeff']
        self.num_epochs = config['ppo']['num_epochs']
        self.batch_size = config['ppo']['batch_size']
        self.gae_lambda = config['ppo']['gae_lambda']
        self.gamma = config['ppo']['gamma']
        self.rollout_steps = config['ppo']['rollout_steps']
        
        # Initialize model
        self.model = ActorCritic(
            input_size=env.observation_space.shape[0],
            lstm_hidden_size=config['model']['lstm_hidden_size'],
            transformer_heads=config['model']['transformer_heads'],
            transformer_layers=config['model']['transformer_layers'],
            action_dim=env.action_space.n,
            dropout=config['model']['dropout']
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.rollout_steps,
            obs_dim=env.observation_space.shape[0], 
            device=self.device
        )
        
        # Metrics
        self.training_metrics = defaultdict(list)
    
    def collect_rollouts(self) -> Dict[str, float]:
        """Collect rollouts using current policy"""
        self.model.eval()
        
        obs, _ = self.env.reset()
        hidden = self.model.init_hidden(1, self.device)
        
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        for step in range(self.rollout_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, log_prob, value, _ = self.model.get_action_and_value(obs_tensor, hidden)
                action = action.item()
                log_prob = log_prob.item()
                value = value.item()
            
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.buffer.add(obs, action, reward, done, log_prob, value)
            
            # Update tracking
            current_episode_reward += reward
            current_episode_length += 1
            
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0
                
                obs, _ = self.env.reset()
                hidden = self.model.init_hidden(1, self.device)
            else:
                obs = next_obs
        
        # Compute returns and advantages
        with torch.no_grad():
            if not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                _, _, last_value, _ = self.model.get_action_and_value(obs_tensor, hidden)
                last_value = last_value.item()
            else:
                last_value = 0.0
        
        self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)
        
        # Return metrics
        metrics = {
            'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0.0,
            'num_episodes': len(episode_rewards)
        }
        
        return metrics
    
    def update_policy(self) -> Dict[str, float]:
        """Update policy using PPO"""
        self.model.train()
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        num_batches = 0
        
        for epoch in range(self.num_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                # Forward pass
                action_logits, values, _ = self.model(batch['observations'])
                
                # Action distribution
                dist = torch.distributions.Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch['actions'])
                entropy = dist.entropy().mean()
                
                # PPO loss
                ratio = torch.exp(new_log_probs - batch['log_probs'])
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch['returns'])
                
                # Total loss
                loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                num_batches += 1
        
        # Return training metrics
        metrics = {
            'total_loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'entropy': total_entropy_loss / num_batches
        }
        
        return metrics
    
    def train(self, total_timesteps: int, log_interval: int = 10) -> None:
        """Main training loop"""
        timesteps_collected = 0
        update_count = 0
        
        while timesteps_collected < total_timesteps:
            # Collect rollouts
            rollout_metrics = self.collect_rollouts()
            timesteps_collected += self.rollout_steps
            
            # Update policy
            training_metrics = self.update_policy()
            update_count += 1
            
            # Log metrics
            if update_count % log_interval == 0:
                print(f"Update {update_count} | Timesteps: {timesteps_collected}")
                print(f"Mean Episode Reward: {rollout_metrics['mean_episode_reward']:.3f}")
                print(f"Policy Loss: {training_metrics['policy_loss']:.6f}")
                print(f"Value Loss: {training_metrics['value_loss']:.6f}")
                print(f"Entropy: {training_metrics['entropy']:.6f}")
                print("-" * 50)
            
            # Store metrics
            for key, value in rollout_metrics.items():
                self.training_metrics[key].append(value)
            for key, value in training_metrics.items():
                self.training_metrics[key].append(value)
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_metrics': dict(self.training_metrics)
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])