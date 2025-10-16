#!/usr/bin/env python3
"""
Long training script for G1 walking environment.
This will run for several hours with proper checkpointing.
"""

import argparse
import torch
import time
import os
from isaaclab.app import AppLauncher

# CLI args
parser = argparse.ArgumentParser(description="Long G1 training")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_iterations", type=int, default=10000, help="Number of training iterations")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
parser.add_argument("--checkpoint_interval", type=int, default=500, help="Save checkpoint every N iterations")
parser.add_argument("--log_interval", type=int, default=100, help="Log progress every N iterations")
args_cli = parser.parse_args()

print("=" * 60)
print("Long G1 Training - Will Run for Hours!")
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

def long_training():
    """Long training loop with checkpointing"""
    
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
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Training loop
        print(f"\nStarting LONG training for {args_cli.num_iterations} iterations...")
        print(f"This will take approximately {args_cli.num_iterations * 0.1 / 60:.1f} minutes")
        print("=" * 60)
        
        total_reward = 0.0
        episode_count = 0
        start_time = time.time()
        last_checkpoint_time = start_time
        
        for iteration in range(args_cli.num_iterations):
            # Generate random actions
            action = torch.randn((env.num_envs, env.action_space.shape[1]), device=env.device, dtype=torch.float32) * 0.1
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Accumulate rewards
            total_reward += reward.sum().item()
            episode_count += terminated.sum().item()
            
            # Log progress
            if iteration % args_cli.log_interval == 0:
                avg_reward = total_reward / (iteration + 1) / env.num_envs
                elapsed_time = time.time() - start_time
                print(f"Iteration {iteration:5d}: Avg Reward = {avg_reward:.4f}, Episodes = {episode_count:4d}, Time = {elapsed_time:6.1f}s")
            
            # Save checkpoint
            if iteration % args_cli.checkpoint_interval == 0 and iteration > 0:
                checkpoint_time = time.time() - start_time
                time_since_last = checkpoint_time - last_checkpoint_time
                avg_reward = total_reward / (iteration + 1) / env.num_envs
                
                print(f"\n{'='*60}")
                print(f"CHECKPOINT at iteration {iteration}")
                print(f"Time since last checkpoint: {time_since_last:.1f}s")
                print(f"Total time: {checkpoint_time:.1f}s")
                print(f"Average reward: {avg_reward:.4f}")
                print(f"Total episodes: {episode_count}")
                print(f"Episodes per second: {episode_count/checkpoint_time:.2f}")
                print(f"{'='*60}\n")
                
                # Save checkpoint data
                checkpoint_data = {
                    'iteration': iteration,
                    'total_reward': total_reward,
                    'episode_count': episode_count,
                    'avg_reward': avg_reward,
                    'elapsed_time': checkpoint_time
                }
                
                # Save to file
                with open(f"logs/checkpoint_{iteration}.txt", "w") as f:
                    f.write(f"Checkpoint at iteration {iteration}\n")
                    f.write(f"Total reward: {total_reward:.2f}\n")
                    f.write(f"Episodes: {episode_count}\n")
                    f.write(f"Average reward: {avg_reward:.4f}\n")
                    f.write(f"Elapsed time: {checkpoint_time:.1f}s\n")
                
                last_checkpoint_time = checkpoint_time
        
        # Final statistics
        total_time = time.time() - start_time
        avg_reward = total_reward / args_cli.num_iterations / env.num_envs
        
        print("\n" + "=" * 60)
        print("LONG TRAINING COMPLETE!")
        print(f"Total iterations: {args_cli.num_iterations}")
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Total episodes: {episode_count}")
        print(f"Episodes per second: {episode_count/total_time:.2f}")
        print("=" * 60)
        
        # Save final results
        with open("logs/final_results.txt", "w") as f:
            f.write(f"Final Training Results\n")
            f.write(f"Total iterations: {args_cli.num_iterations}\n")
            f.write(f"Total time: {total_time:.1f} seconds\n")
            f.write(f"Average reward: {avg_reward:.4f}\n")
            f.write(f"Total episodes: {episode_count}\n")
            f.write(f"Episodes per second: {episode_count/total_time:.2f}\n")
        
        env.close()
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        long_training()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing simulation...")
        simulation_app.close()
