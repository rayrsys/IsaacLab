#!/usr/bin/env python3
"""
Custom G1 walking environment implementation.
"""

import torch
import math
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from g1_walking_env_cfg import G1WalkingEnvCfg
import isaaclab.sim as sim_utils

class G1WalkingEnv(DirectRLEnv):
    """Custom G1 walking environment."""
    
    cfg: G1WalkingEnvCfg
    
    def __init__(self, cfg: G1WalkingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Get joint indices
        self._joint_indices, _ = self.robot.find_joints(".*")
        
        # Initialize data buffers
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.root_pos = self.robot.data.root_pos_w
        self.root_quat = self.robot.data.root_quat_w
        self.root_lin_vel = self.robot.data.root_lin_vel_w
        self.root_ang_vel = self.robot.data.root_ang_vel_w
        
        # Target velocity (forward walking)
        self.target_velocity = torch.tensor([self.cfg.target_velocity, 0.0, 0.0], 
                                          device=self.device).repeat(self.num_envs, 1)
        
        # Previous actions for smoothness
        self.prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
    
    def _setup_scene(self):
        """Set up the simulation scene."""
        print("Setting up G1 walking scene...")
        
        # Create robot
        self.robot = Articulation(self.cfg.robot_cfg)
        print("  OK: Robot created")
        
        # Add terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        print("  OK: Terrain created")
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        print("  OK: Environments cloned")
        
        # Filter collisions for CPU
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
            print("  OK: Collisions filtered for CPU")
        
        # Add robot to scene
        self.scene.articulations["robot"] = self.robot
        print("  OK: Robot added to scene")
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        print("  OK: Lighting added")
        
        print("Scene setup completed!")
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        # Store actions for application
        self.actions = actions.clone()
    
    def _apply_action(self) -> None:
        """Apply actions to the robot."""
        # Apply joint effort targets (like IsaacLab locomotion environments)
        # Scale actions to joint effort targets
        action_scale = 0.5  # Scale factor for actions
        joint_efforts = self.actions * action_scale
        self.robot.set_joint_effort_target(joint_efforts, joint_ids=self._joint_indices)
    
    def _get_observations(self) -> dict:
        """Compute and return observations."""
        # Compute observations
        obs = torch.cat([
            self.joint_pos,                    # Joint positions (37)
            self.joint_vel,                    # Joint velocities (37)
            self.root_lin_vel,                 # Base linear velocity (3)
            self.root_ang_vel,                 # Base angular velocity (3)
            self.target_velocity,              # Target velocity (3)
            self.actions,                      # Previous actions (37)
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute and return rewards."""
        # Compute rewards
        
        # 1. Alive reward (keep robot upright)
        alive_reward = self.cfg.rew_scale_alive * torch.ones(self.num_envs, device=self.device)
        
        # 2. Forward velocity reward (encourage forward movement)
        forward_velocity = self.root_lin_vel[:, 0]  # X component
        velocity_reward = self.cfg.rew_scale_velocity * torch.clamp(forward_velocity, 0, 2.0)
        
        # 3. Orientation reward (keep robot upright) - more sophisticated
        roll, pitch, yaw = self._quat_to_euler(self.root_quat)
        # Reward for staying upright (penalize large roll/pitch)
        orientation_reward = self.cfg.rew_scale_orientation * torch.exp(-(torch.abs(roll) + torch.abs(pitch)))
        
        # 4. Action smoothness reward (penalize large action changes)
        action_diff = torch.norm(self.actions - self.prev_actions, dim=-1)
        smoothness_reward = self.cfg.rew_scale_action_smooth * torch.exp(-action_diff * 10)
        
        # 5. Energy penalty (penalize large actions)
        energy_penalty = self.cfg.rew_scale_energy * torch.sum(self.actions ** 2, dim=-1)
        
        # 6. Joint velocity penalty (penalize excessive joint speeds)
        joint_vel_penalty = 0.01 * torch.sum(torch.abs(self.joint_vel), dim=-1)
        
        total_reward = alive_reward + velocity_reward + orientation_reward + smoothness_reward - energy_penalty - joint_vel_penalty
        
        # Update previous actions
        self.prev_actions = self.actions.clone()
        
        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        # Check timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Check if robot fell (height below threshold) - more sensitive
        fallen = self.root_pos[:, 2] < 0.5  # Increased from 0.3 to 0.5
        
        # Check if robot is moving too fast
        too_fast = torch.norm(self.root_lin_vel, dim=-1) > self.cfg.max_velocity
        
        # Check if robot is too tilted (orientation check)
        roll, pitch, yaw = self._quat_to_euler(self.root_quat)
        too_tilted = (torch.abs(roll) > 0.5) | (torch.abs(pitch) > 0.5)  # ~30 degrees
        
        return fallen | too_fast | too_tilted, time_out
    
    def _reset_idx(self, env_ids):
        """Reset specific environments."""
        # Reset robot to default state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        
        # Add some randomization to joint positions
        joint_pos += torch.randn_like(joint_pos) * 0.05
        
        # Reset root state
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Write to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Reset previous actions
        self.prev_actions[env_ids] = 0.0
    
    def _quat_to_euler(self, quat):
        """Convert quaternion to euler angles (roll, pitch, yaw)."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * math.pi / 2, torch.asin(sinp))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
