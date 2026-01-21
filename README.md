# Tower Bloxx RL Agent 🏗️🤖

A reinforcement learning system that learns to play the mobile game **Tower Bloxx** by connecting to an Android device via USB/ADB or using the Pygame simulation.

<img width="1254" height="956" alt="image" src="https://github.com/user-attachments/assets/307a76cf-d429-415b-a61f-76e706e6565e" />


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

## 🎮 Two Ways to Train

### 1. 🖥️ Pygame Simulator (Fast & Recommended)
We have built a custom Pygame simulator from scratch that replicates the Tower Bloxx physics (swinging pendulum mechanics, collision, gravity). 
- **Pros**: Super fast (no USB latency), stable, can train in background.
- **Cons**: Physics might slightly differ from the original game.

### 2. 📱 Real Device via ADB (Authentic)
Connects to a physical Android phone running the actual game.
- **Pros**: Authentic gameplay interaction.
- **Cons**: Slow (limited by screen capture FPS), requires device setup.

---

## 🚀 Quick Start

### Option A: Train on Simulator 🌟

Run the training immediately on your PC (no phone required).

```bash
# Train with visualization enabled
python train_sim.py --timesteps 100000 --render

# Train in background (faster)
python train_sim.py --timesteps 1000000
```

### Option B: Train on Real Device

#### 1. Connect Device
1. Enable USB debugging on your Android device
2. Connect via USB cable
3. Approve USB debugging prompt on device
4. Open Tower Bloxx game

#### 2. Test Environment
```bash
python scripts/test_environment.py
```

#### 3. Start Training
```bash
# Train with default settings (2M timesteps)
python train.py
```

## 📊 Training Commands

```bash
# Resume training from a checkpoint (Works for both Sim and Real)
python train_sim.py --load ./logs/checkpoints/ppo_towerblox_final.zip --render

# Monitor training progress
tensorboard --logdir=./logs/tensorboard
```

## 📁 Project Structure

```
towerblox-rl-agent/
├── configs/             # Configuration files
├── src/
│   ├── simulation/      # Pygame Simulator logic
│   ├── environment/     # Gym Environment & ADB Logic
│   ├── preprocessing/   # Image processing pipeline
│   ├── training/        # PPO Training loop
│   ├── models/          # Neural Network definitions
│   └── utils/           # Helper scripts
├── scripts/             # Testing scripts
├── train.py             # Entry point for Real Device training
├── train_sim.py         # Entry point for Simulator training
└── logs/                # Checkpoints & TensorBoard logs
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

## 🧮 Reward Structure

The agent receives rewards based on:
- **Alignment**: How close the block is to the center (0.0 - 1.0 scale).
- **Perfect Drop**: +2.0 bonus for perfect alignment.
- **Successful Landing**: +1.0 base reward.
- **Game Over**: -1.0 penalty.

## 🐛 Troubleshooting

### Simulator Issues
- If you get "Not Responding", the training script handles it automatically.
- Ensure `pygame` is installed: `pip install pygame`

### ADB Issues
```bash
# Restart ADB server
adb kill-server; adb start-server
```

## 📈 Expected Results
- **Simulator**: Agent typically learns to stack 20+ blocks within 200k steps.
- **Real Device**: Training is slower; expect results after 6-12 hours.

## 📚 References
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## 📝 License
MIT License

---
**Built with ❤️ using PyTorch, Stable-Baselines3, Gymnasium, and Pygame**
