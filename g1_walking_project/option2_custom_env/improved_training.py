#!/usr/bin/env python3
"""
Improved training script with better action generation and monitoring.
"""

import argparse
import torch
import time
import numpy as np
from isaaclab.app import AppLauncher

# CLI args
parser = argparse.ArgumentParser(description="Improved G1 training")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_iterations", type=int, default=2000, help="Number of training iterations")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
args_cli = parser.parse_args()

print("=" * 60)
print("Improved G1 Training - Better Actions & Monitoring")
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

def improved_training():
    """Improved training with better action generation"""
    
    try:
        # Create environment
        cfg = G1WalkingEnvTrainingCfg()
        cfg.scene.num_envs = args_cli.num_envs
        
        print(f"Creating environment with {cfg.scene.num_envs} environments...")
        env = G1WalkingEnv(cfg, render_mode="human" if not args_cli.headless else None)
        
        print(f"Environment created successfully!")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Reset environment
        print("Resetting environment...")
        obs, info = env.reset()
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        print(f"Environment reset. Observation shape: {tuple(policy_obs.shape)}")
        
        # Training loop
        print(f"\nStarting improved training for {args_cli.num_iterations} iterations...")
        print("Using smaller, more reasonable actions...")
        print("=" * 60)
        
        total_reward = 0.0
        episode_count = 0
        reset_count = 0
        start_time = time.time()
        
        # Action generation parameters
        action_std = 0.1  # Smaller standard deviation
        action_mean = 0.0  # Zero mean
        
        for iteration in range(args_cli.num_iterations):
            # Generate better actions - smaller, more reasonable
            action = torch.randn((env.num_envs, env.action_space.shape[1]), device=env.device, dtype=torch.float32)
            action = action * action_std + action_mean
            
            # Add some sinusoidal patterns to encourage walking-like behavior
            if iteration > 100:  # After initial exploration
                t = torch.tensor(iteration * 0.1, device=env.device)
                for i in range(min(6, env.action_space.shape[1])):  # Apply to first 6 joints (legs)
                    action[:, i] += 0.05 * torch.sin(t + i * 0.5)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Count resets
            reset_count += terminated.sum().item()
            episode_count += terminated.sum().item()
            
            # Accumulate rewards
            total_reward += reward.sum().item()
            
            # Print detailed progress
            if iteration % 100 == 0:
                avg_reward = total_reward / (iteration + 1) / env.num_envs
                elapsed_time = time.time() - start_time
                reset_rate = reset_count / (iteration + 1) / env.num_envs
                
                print(f"Iteration {iteration:4d}:")
                print(f"  Avg Reward: {avg_reward:.4f}")
                print(f"  Episodes: {episode_count}")
                print(f"  Reset Rate: {reset_rate:.3f}")
                print(f"  Time: {elapsed_time:.1f}s")
                
                # Show some action statistics
                action_mean_val = action.mean().item()
                action_std_val = action.std().item()
                print(f"  Action Mean: {action_mean_val:.3f}, Std: {action_std_val:.3f}")
                
                # Show reward breakdown
                if reward.numel() > 0:
                    reward_mean = reward.mean().item()
                    reward_std = reward.std().item()
                    print(f"  Reward Mean: {reward_mean:.3f}, Std: {reward_std:.3f}")
                
                print()
            
            # Print checkpoint every 500 iterations
            if iteration % 500 == 0 and iteration > 0:
                checkpoint_time = time.time() - start_time
                print(f"Checkpoint at iteration {iteration} (Time: {checkpoint_time:.1f}s)")
                print(f"  - Total reward: {total_reward:.2f}")
                print(f"  - Episodes completed: {episode_count}")
                print(f"  - Reset rate: {reset_count/(iteration+1)/env.num_envs:.3f}")
                print("=" * 60)
        
        # Final statistics
        total_time = time.time() - start_time
        avg_reward = total_reward / args_cli.num_iterations / env.num_envs
        final_reset_rate = reset_count / args_cli.num_iterations / env.num_envs
        
        print("\n" + "=" * 60)
        print("Improved Training Complete!")
        print(f"Total iterations: {args_cli.num_iterations}")
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Total episodes: {episode_count}")
        print(f"Final reset rate: {final_reset_rate:.3f}")
        print(f"Episodes per second: {episode_count/total_time:.2f}")
        print("=" * 60)
        
        env.close()
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        improved_training()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing simulation...")
        simulation_app.close()
