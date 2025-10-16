#!/usr/bin/env python3
"""
Configuration for custom G1 walking environment.
"""

from isaaclab_assets import G1_CFG
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

@configclass
class G1WalkingEnvCfg(DirectRLEnvCfg):
    """Configuration for G1 walking environment."""
    
    # Environment settings
    decimation = 4  # 4 physics steps per RL step
    episode_length_s = 20.0  # 20 second episodes
    action_space = 37  # G1 has 37 total joints
    observation_space = 120  # Robot state + task observations (37+37+3+3+3+37)
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4)
    
    # Robot configuration
    robot_cfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,  # Start with fewer environments
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=True
    )
    
    # Terrain (flat for walking)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # Task-specific parameters
    target_velocity = 1.0  # Target walking speed (m/s)
    max_velocity = 2.0     # Maximum allowed velocity
    
    # Reward scales
    rew_scale_alive = 1.0
    rew_scale_velocity = 1.0
    rew_scale_orientation = 0.5
    rew_scale_action_smooth = 0.1
    rew_scale_energy = 0.01
