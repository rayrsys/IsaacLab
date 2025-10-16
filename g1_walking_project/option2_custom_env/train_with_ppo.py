#!/usr/bin/env python3
"""
Train G1 walking environment using PPO (Proximal Policy Optimization).
This script implements a simple PPO algorithm for actual learning.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from isaaclab.app import AppLauncher

# CLI args
parser = argparse.ArgumentParser(description="Train G1 walking with PPO")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_iterations", type=int, default=5000, help="Number of training iterations")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
parser.add_argument("--update_epochs", type=int, default=10, help="Update epochs per iteration")
args_cli = parser.parse_args()

print("=" * 60)
print("G1 Walking Environment - PPO Training")
print("=" * 60)

# Launch Isaac Sim first
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching Isaac Sim
import gymnasium as gym
from g1_walking_env import G1WalkingEnv
from g1_walking_env_training_cfg import G1WalkingEnvTrainingCfg

# Register environment
gym.register(
    id="Isaac-G1-Walking-v0",
    entry_point="g1_walking_env:G1WalkingEnv",
    kwargs={"cfg": G1WalkingEnvTrainingCfg()}
)

class PPOPolicy(nn.Module):
    """Simple PPO Policy Network"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared network
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Policy head (actor)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Value head (critic)
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs):
        shared_features = self.shared(obs)
        
        # Policy
        mean = self.policy_mean(shared_features)
        std = torch.exp(self.policy_logstd)
        
        # Value
        value = self.value(shared_features)
        
        return mean, std, value
    
    def get_action(self, obs):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

def train_with_ppo():
    """Train using PPO algorithm"""
    
    try:
        # Create environment
        cfg = G1WalkingEnvTrainingCfg()
        cfg.scene.num_envs = args_cli.num_envs
        
        print(f"Creating environment with {cfg.scene.num_envs} environments...")
        env = G1WalkingEnv(cfg, render_mode="human" if not args_cli.headless else None)
        
        print(f"Environment created successfully!")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Get dimensions
        obs_dim = env.observation_space.shape[1]  # 120
        action_dim = env.action_space.shape[1]    # 37
        
        print(f"Observation dimension: {obs_dim}")
        print(f"Action dimension: {action_dim}")
        
        # Create PPO policy
        policy = PPOPolicy(obs_dim, action_dim).to(env.device)
        optimizer = optim.Adam(policy.parameters(), lr=args_cli.learning_rate)
        
        # Reset environment
        print("Resetting environment...")
        obs, info = env.reset()
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        print(f"Environment reset. Observation shape: {tuple(policy_obs.shape)}")
        
        # Training loop
        print(f"\nStarting PPO training for {args_cli.num_iterations} iterations...")
        print("=" * 60)
        
        # Storage for PPO
        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        value_buffer = []
        log_prob_buffer = []
        
        total_reward = 0.0
        episode_count = 0
        start_time = time.time()
        
        for iteration in range(args_cli.num_iterations):
            # Collect data
            with torch.no_grad():
                action, log_prob, value = policy.get_action(policy_obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_policy_obs = next_obs["policy"] if isinstance(next_obs, dict) else next_obs
                
                # Store data
                obs_buffer.append(policy_obs.clone())
                action_buffer.append(action.clone())
                reward_buffer.append(reward.clone())
                value_buffer.append(value.clone())
                log_prob_buffer.append(log_prob.clone())
                
                # Update for next iteration
                policy_obs = next_policy_obs
                
                # Count episodes
                episode_count += terminated.sum().item()
                total_reward += reward.sum().item()
            
            # PPO Update
            if len(obs_buffer) >= args_cli.batch_size:
                # Convert buffers to tensors
                obs_batch = torch.cat(obs_buffer[:args_cli.batch_size], dim=0)
                action_batch = torch.cat(action_buffer[:args_cli.batch_size], dim=0)
                reward_batch = torch.cat(reward_buffer[:args_cli.batch_size], dim=0)
                value_batch = torch.cat(value_buffer[:args_cli.batch_size], dim=0)
                old_log_prob_batch = torch.cat(log_prob_buffer[:args_cli.batch_size], dim=0)
                
                # Clear buffers
                obs_buffer = obs_buffer[args_cli.batch_size:]
                action_buffer = action_buffer[args_cli.batch_size:]
                reward_buffer = reward_buffer[args_cli.batch_size:]
                value_buffer = value_buffer[args_cli.batch_size:]
                log_prob_buffer = log_prob_buffer[args_cli.batch_size:]
                
                # PPO update
                for _ in range(args_cli.update_epochs):
                    # Get new policy outputs
                    mean, std, new_value = policy(obs_batch)
                    dist = torch.distributions.Normal(mean, std)
                    new_log_prob = dist.log_prob(action_batch).sum(dim=-1)
                    
                    # Compute advantages (simplified)
                    advantages = reward_batch - new_value.squeeze()
                    
                    # Compute policy loss
                    ratio = torch.exp(new_log_prob - old_log_prob_batch)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Compute value loss
                    value_loss = F.mse_loss(new_value.squeeze(), reward_batch)
                    
                    # Total loss
                    total_loss = policy_loss + 0.5 * value_loss
                    
                    # Update
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
            
            # Print progress
            if iteration % 100 == 0:
                avg_reward = total_reward / (iteration + 1) / env.num_envs
                elapsed_time = time.time() - start_time
                print(f"Iteration {iteration:4d}: Avg Reward = {avg_reward:.4f}, Episodes = {episode_count}, Time = {elapsed_time:.1f}s")
        
        # Final statistics
        total_time = time.time() - start_time
        avg_reward = total_reward / args_cli.num_iterations / env.num_envs
        
        print("\n" + "=" * 60)
        print("PPO Training Complete!")
        print(f"Total iterations: {args_cli.num_iterations}")
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Total episodes: {episode_count}")
        print(f"Episodes per second: {episode_count/total_time:.2f}")
        print("=" * 60)
        
        # Save policy
        torch.save(policy.state_dict(), "g1_walking_policy.pth")
        print("Policy saved to g1_walking_policy.pth")
        
        env.close()
        
    except Exception as e:
        print(f"Error during PPO training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        train_with_ppo()
    except KeyboardInterrupt:
        print("\nPPO training interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing simulation...")
        simulation_app.close()
