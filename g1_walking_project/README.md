# G1 Walking Project

This project demonstrates how to create and train a G1 humanoid robot to walk in IsaacLab using a custom Direct Workflow environment.

## 🚀 Quick Start

### 1. Setup Environment
```bash
conda activate env_isaaclab
cd C:\Users\hansl\Documents\IsaacLab
```

### 2. Test the Environment
```bash
cd g1_walking_project/option2_custom_env
python test_custom_g1.py
```

### 3. Start Training

#### Option A: Random Actions (No Learning)
```bash
# Quick random training (2000 iterations, 16 envs)
python improved_training.py --num_iterations 2000 --num_envs 16

# Long random training (10000 iterations, 1024 envs)
python long_training.py --headless --num_iterations 10000 --num_envs 1024
```

#### Option B: Real RL Training (Actual Learning)
```bash
# PPO algorithm (256 envs, 5000 iterations)
python train_with_ppo.py --num_iterations 5000 --num_envs 256

# IsaacLab RL Games (1024 envs, 2000 iterations)
python train_with_rl_games.py --num_envs 1024 --max_iterations 2000
```

## 📁 Project Structure

### Core Files
- **`g1_walking_env.py`** - Main environment implementation
- **`g1_walking_env_cfg.py`** - Basic configuration (4 environments)
- **`g1_walking_env_training_cfg.py`** - Training configuration (1024 environments)

### Training Scripts
- **`improved_training.py`** - Random actions with patterns (no learning)
- **`long_training.py`** - Extended random training with checkpointing
- **`train_with_ppo.py`** - **REAL RL**: PPO algorithm with neural network
- **`train_with_rl_games.py`** - **REAL RL**: IsaacLab's RL Games integration
- **`test_custom_g1.py`** - Test script to verify environment works

### Documentation
- **`README.md`** - This file
- **`TRAINING_GUIDE.md`** - Detailed training instructions
- **`IMPROVEMENTS.md`** - Technical improvements made
- **`FINAL_SUMMARY.md`** - Project completion summary

## 🎯 Key Features

- **37-joint G1 humanoid robot** with realistic physics
- **Effort-based joint control** for proper movement
- **Multi-component reward function** encouraging walking
- **Proper termination conditions** with automatic resets
- **Scalable training** from 4 to 1024+ environments
- **Checkpointing and logging** for long training runs

## 📊 Training Results

The environment has been tested and shows:
- **Stable training** with increasing rewards
- **Proper robot movement** and joint articulation
- **Automatic resets** when robots fall
- **Scalable performance** for large-scale training

## 🔧 Customization

### Adjust Training Parameters
Edit `g1_walking_env_training_cfg.py`:
- `num_envs` - Number of parallel environments
- `episode_length_s` - Episode duration
- `target_velocity` - Walking speed target
- Reward scales for fine-tuning

### Modify Reward Function
Edit `g1_walking_env.py` in `_get_rewards()` method:
- Forward velocity reward
- Orientation stability reward
- Action smoothness reward
- Energy efficiency penalty

## 🎮 Training Options

### Visual Training (Watch Robots Learn)
```bash
python improved_training.py --num_iterations 2000 --num_envs 16
```

### Headless Training (Faster, No Graphics)
```bash
python long_training.py --headless --num_iterations 10000 --num_envs 1024
```

### Quick Test
```bash
python test_custom_g1.py
```

## 📈 Expected Training Progression

1. **Iterations 0-500**: Robots learn to balance and stay upright
2. **Iterations 500-2000**: Robots develop basic walking patterns
3. **Iterations 2000+**: Robots optimize walking efficiency and speed

## 🏆 Success Metrics

- **Reward > 2.0**: Good walking performance
- **Reset rate < 0.1**: Robots staying upright consistently
- **Forward velocity > 0.5 m/s**: Actual forward movement
- **Episode length > 15s**: Sustained walking behavior

## 📚 Additional Resources

- **`TRAINING_GUIDE.md`** - Complete training documentation
- **`IMPROVEMENTS.md`** - Technical details of improvements made
- **`FINAL_SUMMARY.md`** - Project completion summary

## 🚀 Next Steps

1. **Run training** using the scripts above
2. **Monitor progress** through reward increases and reset rate decreases
3. **Customize rewards** for specific walking behaviors
4. **Scale up** to more environments for faster training
5. **Integrate with RL frameworks** for professional training