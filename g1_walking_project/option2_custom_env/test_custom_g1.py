#!/usr/bin/env python3
"""
Test script for custom G1 walking environment.
"""

import argparse
from isaaclab.app import AppLauncher

# CLI args
parser = argparse.ArgumentParser(description="Test custom G1 walking environment")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print("=" * 60)
print("Testing Custom G1 Walking Environment")
print("=" * 60)

# Launch Isaac Sim first
print("1. Launching Isaac Sim...")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("   OK: Isaac Sim launched successfully")

# Import IsaacLab / torch AFTER launching the app
print("2. Importing IsaacLab modules...")
import torch
print("   OK: PyTorch imported")

try:
    from g1_walking_env import G1WalkingEnv
    from g1_walking_env_cfg import G1WalkingEnvCfg
    print("   OK: Custom G1 environment imported")
except Exception as e:
    print(f"   ERROR: Failed to import custom G1 environment: {e}")
    exit(1)

print("3. Creating environment configuration...")
try:
    cfg = G1WalkingEnvCfg()
    cfg.scene.num_envs = 4  # Very few environments for testing
    print(f"   OK: Configuration created with {cfg.scene.num_envs} environments")
except Exception as e:
    print(f"   ERROR: Failed to create configuration: {e}")
    exit(1)

print("4. Creating environment...")
try:
    env = G1WalkingEnv(cfg, render_mode="human")
    print("   OK: Environment created successfully!")
except Exception as e:
    print(f"   ERROR: Failed to create environment: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("5. Getting environment info...")
try:
    print(f"   OK: Action space: {env.action_space}")
    print(f"   OK: Observation space: {env.observation_space}")
    print(f"   OK: Number of environments: {env.num_envs}")
    print(f"   OK: Device: {env.device}")
except Exception as e:
    print(f"   ERROR: Failed to get environment info: {e}")
    exit(1)

print("6. Resetting environment...")
try:
    obs, info = env.reset()
    policy_obs = obs["policy"] if isinstance(obs, dict) else obs
    print(f"   OK: Environment reset. Observation shape: {tuple(policy_obs.shape)}")
except Exception as e:
    print(f"   ERROR: Failed to reset environment: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("7. Testing with zero actions...")
try:
    # Create zero actions with correct shape
    action = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device, dtype=torch.float32)
    print(f"   OK: Action tensor shape: {action.shape}")
    
    # Run a few steps
    for i in range(5):
        print(f"   Step {i+1}/5...")
        obs, reward, terminated, truncated, info = env.step(action)
        r_mean = reward.mean().item() if reward.numel() > 0 else float("nan")
        term_count = int(terminated.sum().item()) if terminated is not None else 0
        print(f"     Reward mean: {r_mean:.4f}, Terminated: {term_count}/{env.num_envs}")
    
    print("   OK: Zero action test completed successfully!")
except Exception as e:
    print(f"   ERROR: Failed during zero action test: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("8. Testing with random actions...")
try:
    # Create random actions
    action = torch.randn((env.num_envs, env.action_space.shape[1]), device=env.device, dtype=torch.float32) * 0.1
    print(f"   OK: Random action tensor shape: {action.shape}")
    
    # Run a few steps
    for i in range(5):
        print(f"   Step {i+1}/5...")
        obs, reward, terminated, truncated, info = env.step(action)
        r_mean = reward.mean().item() if reward.numel() > 0 else float("nan")
        term_count = int(terminated.sum().item()) if terminated is not None else 0
        print(f"     Reward mean: {r_mean:.4f}, Terminated: {term_count}/{env.num_envs}")
    
    print("   OK: Random action test completed successfully!")
except Exception as e:
    print(f"   ERROR: Failed during random action test: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("9. Cleaning up...")
try:
    env.close()
    print("   OK: Environment closed successfully")
except Exception as e:
    print(f"   ERROR: Failed to close environment: {e}")

print("=" * 60)
print("Custom G1 Environment Test COMPLETED SUCCESSFULLY!")
print("=" * 60)

# Clean up
simulation_app.close()
