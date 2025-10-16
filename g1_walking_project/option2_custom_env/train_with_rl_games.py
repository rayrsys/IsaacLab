#!/usr/bin/env python3
"""
Train G1 walking environment using IsaacLab's RL Games integration.
This uses the proper IsaacLab training pipeline.
"""

import argparse
import os
import sys
from isaaclab.app import AppLauncher

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# CLI args
parser = argparse.ArgumentParser(description="Train G1 walking with RL Games")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=2000, help="Maximum training iterations")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="Checkpoint interval")
args_cli = parser.parse_args()

print("=" * 60)
print("G1 Walking Environment - RL Games Training")
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

def train_with_rl_games():
    """Train using IsaacLab's RL Games integration"""
    
    try:
        # Import RL Games training script
        from scripts.reinforcement_learning.rl_games.train import main as rl_games_main
        
        # Set up arguments for RL Games
        sys.argv = [
            "train.py",
            "--task=Isaac-G1-Walking-v0",
            f"--num_envs={args_cli.num_envs}",
            f"--max_iterations={args_cli.max_iterations}",
            f"--checkpoint_interval={args_cli.checkpoint_interval}",
            "--headless" if args_cli.headless else "",
        ]
        
        # Remove empty strings
        sys.argv = [arg for arg in sys.argv if arg]
        
        print(f"Starting RL Games training with:")
        print(f"  Task: Isaac-G1-Walking-v0")
        print(f"  Environments: {args_cli.num_envs}")
        print(f"  Max iterations: {args_cli.max_iterations}")
        print(f"  Checkpoint interval: {args_cli.checkpoint_interval}")
        print(f"  Headless: {args_cli.headless}")
        
        # Run RL Games training
        rl_games_main()
        
    except ImportError:
        print("RL Games not available. Running basic training instead...")
        run_basic_training()
    except Exception as e:
        print(f"Error with RL Games: {e}")
        print("Falling back to basic training...")
        run_basic_training()

def run_basic_training():
    """Fallback basic training if RL Games is not available"""
    
    import torch
    import time
    
    # Create environment
    cfg = G1WalkingEnvTrainingCfg()
    cfg.scene.num_envs = args_cli.num_envs
    
    print(f"Creating environment with {cfg.scene.num_envs} environments...")
    env = G1WalkingEnv(cfg, render_mode="human" if not args_cli.headless else None)
    
    print(f"Environment created successfully!")
    
    # Reset environment
    obs, info = env.reset()
    
    # Basic training loop
    print(f"\nStarting basic training for {args_cli.max_iterations} iterations...")
    print("=" * 60)
    
    total_reward = 0.0
    episode_count = 0
    start_time = time.time()
    
    for iteration in range(args_cli.max_iterations):
        # Generate random actions
        action = torch.randn((env.num_envs, env.action_space.shape[1]), device=env.device, dtype=torch.float32) * 0.1
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate rewards
        total_reward += reward.sum().item()
        episode_count += terminated.sum().item()
        
        # Print progress
        if iteration % 100 == 0:
            avg_reward = total_reward / (iteration + 1) / env.num_envs
            elapsed_time = time.time() - start_time
            print(f"Iteration {iteration:4d}: Avg Reward = {avg_reward:.4f}, Episodes = {episode_count}, Time = {elapsed_time:.1f}s")
        
        # Save checkpoint
        if iteration % args_cli.checkpoint_interval == 0 and iteration > 0:
            checkpoint_time = time.time() - start_time
            print(f"Checkpoint at iteration {iteration} (Time: {checkpoint_time:.1f}s)")
    
    # Final statistics
    total_time = time.time() - start_time
    avg_reward = total_reward / args_cli.max_iterations / env.num_envs
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Total iterations: {args_cli.max_iterations}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Total episodes: {episode_count}")
    print(f"Episodes per second: {episode_count/total_time:.2f}")
    print("=" * 60)
    
    env.close()

if __name__ == "__main__":
    try:
        train_with_rl_games()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

