# G1 Walking Environment - Training Guide

## 🚀 Ready to Train!

Your G1 walking environment is now ready for training! Here's everything you need to know.

## Quick Start

### 1. Test Environment (4 environments)
```bash
cd g1_walking_project/option2_custom_env
python test_custom_g1.py --headless
```

### 2. Start Training (Multiple Options)

#### Option A: Simple Training (64 environments, 1000 iterations)
```bash
cd g1_walking_project/option2_custom_env
python simple_training.py --headless --num_iterations 1000 --num_envs 64
```

#### Option B: Long Training (1024 environments, 10000 iterations)
```bash
cd g1_walking_project/option2_custom_env
python long_training.py --headless --num_iterations 10000 --num_envs 1024
```

#### Option C: RL Games Training (Professional)
```bash
cd g1_walking_project/option2_custom_env
python train_with_rl_games.py --headless --num_envs 1024 --max_iterations 2000
```

#### Option D: IsaacLab RL Games (Original)
```bash
cd g1_walking_project/option2_custom_env
python quick_train.py --headless
cd ../../
python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-G1-Walking-v0 --num_envs=1024 --headless
```

## Training Configuration

### Environment Settings
- **Environments**: 1024 parallel environments
- **Episode Length**: 20 seconds
- **Physics Timestep**: 0.005s (200 Hz)
- **RL Timestep**: 0.02s (50 Hz)
- **Action Space**: 37 dimensions (all G1 joints)
- **Observation Space**: 48 dimensions

### Reward Function
- **Alive Reward**: +1.0 (stay upright)
- **Velocity Tracking**: +2.0 (walk at target speed)
- **Orientation Penalty**: -1.0 (prevent falling)
- **Action Smoothness**: +0.1 (smooth movements)
- **Energy Penalty**: -0.01 (efficient walking)

### Termination Conditions
- Robot falls (height < 0.3m)
- Robot moves too fast (> 2.0 m/s)
- Episode timeout (20 seconds)

## Training Progress

### What to Expect
1. **Initial Phase** (0-100 iterations): Robot learns to stand up
2. **Walking Phase** (100-500 iterations): Robot learns basic walking
3. **Optimization Phase** (500-1000 iterations): Robot improves walking efficiency

### Monitoring Training
- **Reward**: Should increase from ~0 to ~10+ over time
- **Episode Length**: Should increase as robot learns to stay upright
- **Velocity Tracking**: Should improve as robot learns to walk at target speed

## Training Commands

### Basic Training
```bash
python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-G1-Walking-v0 --num_envs=1024 --headless
```

### Training with Rendering (slower but visual)
```bash
python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-G1-Walking-v0 --num_envs=64
```

### Resume Training from Checkpoint
```bash
python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-G1-Walking-v0 --num_envs=1024 --headless --resume
```

### Training with Custom Parameters
```bash
python scripts/reinforcement_learning/rl_games/train.py \
    --task=Isaac-G1-Walking-v0 \
    --num_envs=1024 \
    --headless \
    --max_iterations=2000 \
    --checkpoint_interval=50
```

## Files Created

### Environment Files
- `g1_walking_env_cfg.py` - Basic configuration (4 envs)
- `g1_walking_env_training_cfg.py` - Training configuration (1024 envs)
- `g1_walking_env.py` - Environment implementation
- `test_custom_g1.py` - Test script
- `quick_train.py` - Quick training setup

### Training Files
- `train_g1_walking.py` - Full training script
- `TRAINING_GUIDE.md` - This guide

## Customization Options

### 1. Adjust Reward Scales
Edit `g1_walking_env_training_cfg.py`:
```python
# Increase velocity tracking reward
rew_scale_velocity = 3.0  # Default: 2.0

# Increase orientation penalty
rew_scale_orientation = 2.0  # Default: 1.0
```

### 2. Change Target Velocity
```python
# Make robot walk faster
target_velocity = 1.5  # Default: 1.0

# Make robot walk slower
target_velocity = 0.5  # Default: 1.0
```

### 3. Adjust Environment Count
```python
# More environments (faster training, more GPU memory)
num_envs = 2048  # Default: 1024

# Fewer environments (less GPU memory)
num_envs = 512   # Default: 1024
```

### 4. Change Episode Length
```python
# Longer episodes
episode_length_s = 30.0  # Default: 20.0

# Shorter episodes
episode_length_s = 10.0  # Default: 20.0
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `num_envs` from 1024 to 512 or 256
2. **Training Too Slow**: Increase `num_envs` or reduce `episode_length_s`
3. **Robot Not Learning**: Adjust reward scales or target velocity
4. **Environment Creation Hanging**: Use `--headless` flag

### Performance Tips

1. **Use Headless Mode**: Always use `--headless` for training
2. **GPU Memory**: Monitor GPU memory usage, reduce environments if needed
3. **Checkpointing**: Save checkpoints frequently to avoid losing progress
4. **Resume Training**: Use `--resume` to continue from last checkpoint

## Next Steps After Training

1. **Evaluate Policy**: Test trained policy with different scenarios
2. **Add Terrain**: Train on rough terrain for robustness
3. **Multi-Task**: Add different walking speeds and directions
4. **Sim-to-Real**: Transfer to real G1 robot

## Success Metrics

- **Reward > 10**: Good walking performance
- **Episode Length > 15s**: Robot stays upright
- **Velocity Error < 0.2 m/s**: Accurate speed tracking
- **No Falls**: Robot doesn't fall during episodes

Happy Training! 🎯
