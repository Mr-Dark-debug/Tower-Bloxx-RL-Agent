# Tower Bloxx RL Agent 🏗️🤖

A reinforcement learning system that learns to play the mobile game **Tower Bloxx** by connecting to an Android device via USB/ADB.

## 🎮 About the Game

**Tower Bloxx** is a physics-based block stacking game where:
- A crane swings left and right with a building block
- You tap to release the block at the right moment
- Goal: Stack blocks as precisely as possible to build the tallest tower
- Perfect placement earns bonus coins; misaligned blocks may cause the tower to collapse

## ✨ Features

- 📱 **Real Device Training**: Connects to Android phone via ADB for authentic gameplay
- 🧠 **PPO Algorithm**: Uses Proximal Policy Optimization from Stable-Baselines3
- 🎯 **CNN Policy**: Convolutional neural network for processing visual observations
- ⚡ **GPU Accelerated**: Optimized for NVIDIA RTX 3050 Ti (4GB VRAM)
- 📊 **TensorBoard Logging**: Real-time training metrics visualization
- 🔄 **Frame Stacking**: 4-frame stack for motion perception
- 💾 **Checkpointing**: Automatic model saving and best model tracking

## 🛠️ Requirements

### Hardware
- **GPU**: NVIDIA RTX 3050 Ti (or compatible CUDA GPU)
- **RAM**: 16GB recommended
- **Android Device**: With USB debugging enabled

### Software
- Python 3.9+
- NVIDIA CUDA 12.6 (or compatible) `https://developer.nvidia.com/cuda-downloads`
- Android Debug Bridge (ADB)
- Tower Bloxx game installed on device

## 📦 Installation

### 1. Clone and Setup Virtual Environment

```bash
cd d:\Opensource\towerblox-rl-agent

# Virtual environment already created, activate it:
.\venv\Scripts\activate
```

### 2. Install PyTorch with CUDA 12.6

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python scripts/setup_environment.py
```

### 5. Test ADB Connection

```bash
# Connect your Android device via USB
adb devices

# Run ADB test
python scripts/test_adb_connection.py
```

## 🚀 Quick Start

### 1. Connect Device
1. Enable USB debugging on your Android device
2. Connect via USB cable
3. Approve USB debugging prompt on device
4. Open Tower Bloxx game

### 2. Test Environment

```bash
python scripts/test_environment.py
```

### 3. Start Training

```bash
# Train with default settings (2M timesteps)
python train.py

# Train for specific number of steps
python train.py --timesteps 500000

# Continue from checkpoint
python train.py --load ./logs/checkpoints/ppo_towerblox_50000_steps.zip
```

### 4. Monitor Training

```bash
# Open TensorBoard
tensorboard --logdir=./logs/tensorboard
```

### 5. Evaluate Trained Model

```bash
# Run evaluation
python evaluate.py --model ./logs/checkpoints/ppo_towerblox_final.zip -n 100

# Watch agent play
python evaluate.py --model ./logs/checkpoints/ppo_towerblox_final.zip --render
```

## 📁 Project Structure

```
towerblox-rl-agent/
├── configs/
│   ├── env_config.yaml      # Environment settings
│   ├── ppo_config.yaml      # PPO hyperparameters
│   └── device_config.yaml   # ADB/device settings
├── src/
│   ├── environment/
│   │   ├── mobile_game_env.py   # Main Gymnasium environment
│   │   ├── screen_capture.py    # ADB screen capture
│   │   ├── action_executor.py   # Touch input execution
│   │   └── reward_shaper.py     # Reward calculation
│   ├── preprocessing/
│   │   ├── frame_processor.py   # Image preprocessing
│   │   └── frame_stacker.py     # Frame stacking wrapper
│   ├── training/
│   │   ├── trainer.py           # Main training loop
│   │   └── callbacks.py         # SB3 callbacks
│   ├── evaluation/
│   │   ├── evaluator.py         # Model evaluation
│   │   └── visualizer.py        # Visualization tools
│   └── utils/
│       ├── adb_manager.py       # ADB connection handler
│       ├── config_loader.py     # YAML config parser
│       ├── logger.py            # Custom logging
│       └── gpu_monitor.py       # GPU monitoring
├── scripts/
│   ├── setup_environment.py     # Installation verification
│   ├── test_adb_connection.py   # ADB connectivity test
│   └── test_environment.py      # Environment test
├── train.py                     # Training entry point
├── evaluate.py                  # Evaluation entry point
├── requirements.txt             # Python dependencies
└── logs/                        # Training outputs
    ├── tensorboard/
    ├── checkpoints/
    └── training_logs/
```

## ⚙️ Configuration

### Environment Config (`configs/env_config.yaml`)
- Observation settings (frame size, stacking)
- Action space definition
- Reward values
- Screen region coordinates

### PPO Config (`configs/ppo_config.yaml`)
- Learning rate: 3e-4
- Batch size: 64 (optimized for 4GB VRAM)
- n_steps: 2048
- n_epochs: 10
- gamma: 0.99

### Device Config (`configs/device_config.yaml`)
- Device serial number
- Screen dimensions
- ADB settings

## 🧮 Reward Structure

| Event | Reward |
|-------|--------|
| Perfect placement | +5.0 |
| Good placement | +2.0 |
| OK placement | +1.0 |
| Wobbly placement | +0.3 |
| Height bonus (per floor) | +0.5 |
| Coin collected | +1.0 |
| Step penalty | -0.01 |
| Game over | -10.0 |

## 🎯 Action Space

| Action | Description |
|--------|-------------|
| 0 | Wait (do nothing) |
| 1 | Tap (release block) |

## 📊 Training Tips

1. **Start Small**: Begin with 100K steps to verify everything works
2. **Monitor GPU**: Watch VRAM usage via `nvidia-smi`
3. **Check TensorBoard**: Look for increasing reward trends
4. **Adjust Rewards**: If agent doesn't learn, modify reward values
5. **Frame Rate**: Target 30+ FPS for stable training

## 🐛 Troubleshooting

### ADB Issues
```bash
# Restart ADB server
adb kill-server
adb start-server
adb devices
```

### CUDA Out of Memory
- Reduce batch_size in `ppo_config.yaml`
- Enable FP16 in config

### Slow Capture
- Check USB connection (use USB 3.0)
- Reduce screen capture resolution

## 📈 Expected Results

- **Random baseline**: ~-50 reward per episode
- **Trained agent (1M steps)**: ~+20-50 reward per episode
- **Expert agent (5M+ steps)**: ~+100+ reward per episode

## 📚 References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## 📝 License

MIT License

---

**Built with ❤️ using PyTorch, Stable-Baselines3, and Gymnasium**
