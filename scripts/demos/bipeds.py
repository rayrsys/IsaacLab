# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate bipedal robots.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/bipeds.py

"""

"""Launch Isaac Sim Simulator first."""

# Import argparse for command line argument parsing
import argparse

# Import the AppLauncher from Isaac Lab to handle Isaac Sim initialization
from isaaclab.app import AppLauncher

# Create argument parser to handle command line arguments
# This allows users to pass configuration options when running the script
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate bipedal robots.")
# Add Isaac Lab's built-in command line arguments (like device selection, headless mode, etc.)
AppLauncher.add_app_launcher_args(parser)
# Parse the command line arguments and store them in args_cli
args_cli = parser.parse_args()

# Create and launch the Isaac Sim application using the parsed arguments
# This initializes the Omniverse Kit and Isaac Sim environment
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Import PyTorch for tensor operations and GPU acceleration
import torch

# Import Isaac Lab simulation utilities and classes
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation  # Class for handling articulated robots
from isaaclab.sim import SimulationContext  # Main simulation context manager

##
# Pre-defined configs
##
# Import robot configuration files for the three bipedal robots
from isaaclab_assets.robots.cassie import CASSIE_CFG  # Cassie bipedal robot configuration
from isaaclab_assets import H1_CFG  # H1 humanoid robot configuration
from isaaclab_assets import G1_CFG  # G1 humanoid robot configuration


def design_scene(sim: sim_utils.SimulationContext) -> tuple[list, torch.Tensor]:
    """Designs the scene with ground plane, lighting, and three bipedal robots."""
    
    # Create a ground plane for the robots to stand on
    # This provides a flat surface with physics properties
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # Add dome lighting to illuminate the scene
    # intensity=2000.0 makes it bright, color=(0.75, 0.75, 0.75) is a neutral white
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Define the starting positions (origins) for each robot in 3D space
    # Each row represents [x, y, z] coordinates for one robot
    # Robot 1: Cassie at (0, -1, 0) - to the left
    # Robot 2: H1 at (0, 0, 0) - in the center  
    # Robot 3: G1 at (0, 1, 0) - to the right
    origins = torch.tensor([
        [0.0, -1.0, 0.0],  # Cassie position
        [0.0, 0.0, 0.0],   # H1 position
        [0.0, 1.0, 0.0],   # G1 position
    ]).to(device=sim.device)  # Move tensor to the same device as simulation (CPU/GPU)

    # Create articulation objects for each robot
    # Articulation handles the robot's joints, physics, and control
    cassie = Articulation(CASSIE_CFG.replace(prim_path="/World/Cassie"))  # Cassie robot
    h1 = Articulation(H1_CFG.replace(prim_path="/World/H1"))              # H1 humanoid
    g1 = Articulation(G1_CFG.replace(prim_path="/World/G1"))              # G1 humanoid
    
    # Store all robots in a list for easy iteration
    robots = [cassie, h1, g1]

    return robots, origins


def run_simulator(sim: sim_utils.SimulationContext, robots: list[Articulation], origins: torch.Tensor):
    """Runs the main simulation loop that controls the robots and advances physics."""
    
    # Get the physics timestep (how much time passes in each simulation step)
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0  # Track total simulation time
    count = 0       # Count simulation steps
    
    # Main simulation loop - runs as long as Isaac Sim is active
    while simulation_app.is_running():
        
        # Reset robots every 200 simulation steps (1 second at 0.005s timestep)
        if count % 200 == 0:
            # Reset simulation counters
            sim_time = 0.0
            count = 0
            
            # Reset each robot to its default state
            for index, robot in enumerate(robots):
                # Reset joint positions and velocities to default values
                joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                
                # Reset robot's root (base) position and orientation
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]  # Add the robot's origin offset
                robot.write_root_pose_to_sim(root_state[:, :7])      # Position + quaternion
                robot.write_root_velocity_to_sim(root_state[:, 7:])  # Linear + angular velocity
                
                # Reset the robot's internal state
                robot.reset()
            
            # Print reset message to console
            print(">>>>>>>> Reset!")
        
        # Apply control actions to each robot
        for robot in robots:
            # Set joint position targets to default standing pose
            # This keeps the robots in their default standing position
            robot.set_joint_position_target(robot.data.default_joint_pos.clone())
            # Write the control commands to the simulation
            robot.write_data_to_sim()
        
        # Advance the physics simulation by one timestep
        sim.step()
        
        # Update simulation time and step counter
        sim_time += sim_dt
        count += 1
        
        # Update robot data buffers (joint states, sensor data, etc.)
        for robot in robots:
            robot.update(sim_dt)


def main():
    """Main function that sets up and runs the bipedal robot simulation."""
    
    # Create simulation configuration
    # dt=0.005 means 200 Hz simulation frequency (0.005 seconds per step)
    # device is set from command line arguments (CPU or GPU)
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set up the camera view for the simulation window
    # eye=[3.0, 0.0, 2.25] positions the camera at (3, 0, 2.25)
    # target=[0.0, 0.0, 1.0] makes the camera look at point (0, 0, 1)
    # This gives a good side view of the robots
    sim.set_camera_view(eye=[3.0, 0.0, 2.25], target=[0.0, 0.0, 1.0])

    # Create the simulation scene with robots and environment
    robots, origins = design_scene(sim)

    # Initialize the simulation (load all assets, set up physics, etc.)
    sim.reset()

    # Print confirmation that setup is complete
    print("[INFO]: Setup complete...")

    # Start the main simulation loop
    run_simulator(sim, robots, origins)


if __name__ == "__main__":
    # This block runs when the script is executed directly (not imported)
    
    # Run the main simulation function
    main()
    
    # Clean up: close the Isaac Sim application when simulation ends
    simulation_app.close()
