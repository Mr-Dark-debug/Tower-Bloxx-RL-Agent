"""
Trainer Module
Main training loop and model management.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from ..environment.mobile_game_env import TowerBloxEnv
from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger, setup_logger
from ..utils.gpu_monitor import GPUMonitor
from .callbacks import (
    CheckpointCallback,
    GPUMonitorCallback,
    TrainingLoggerCallback,
)

logger = get_logger(__name__)


class Trainer:
    """
    Main trainer for Tower Bloxx RL agent.
    
    Handles:
    - Environment creation and wrapping
    - Model initialization and configuration
    - Training loop with callbacks
    - Checkpointing and evaluation
    
    Attributes:
        config_dir: Path to configuration directory
        model: PPO model instance
        env: Vectorized environment
    """
    
    def __init__(
        self,
        config_dir: Optional[str] = None,
        device: str = "auto",
        log_dir: str = "./logs",
    ):
        """
        Initialize trainer.
        
        Args:
            config_dir: Path to configuration directory
            device: Device to use ('auto', 'cuda', 'cpu')
            log_dir: Directory for logs and checkpoints
        """
        # Setup logging
        setup_logger(log_dir=os.path.join(log_dir, "training_logs"))
        
        # Load configuration
        self.config_loader = ConfigLoader(config_dir)
        self.ppo_config = self.config_loader.get_ppo_config()
        self.env_config = self.config_loader.get_env_config()
        
        # Setup directories
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.tensorboard_dir = self.log_dir / "tensorboard"
        self.tensorboard_dir.mkdir(exist_ok=True)
        
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Determine device
        self.device = self._determine_device(device)
        
        # Track training state
        self.model: Optional[PPO] = None
        self.env: Optional[DummyVecEnv] = None
        self.eval_env: Optional[DummyVecEnv] = None
        
        # GPU monitoring
        self.gpu_monitor = GPUMonitor()
        
        # Training statistics
        self._training_start_time: float = 0.0
        self._total_timesteps: int = 0
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                device = "cpu"
                logger.warning("CUDA not available, using CPU")
        return device
    
    def create_env(
        self, 
        render_mode: Optional[str] = None,
        show_visualization: bool = False,  # Disable for training performance
    ) -> DummyVecEnv:
        """
        Create and wrap the training environment.
        
        Args:
            render_mode: Rendering mode for environment
            show_visualization: Show visualization window
            
        Returns:
            Vectorized environment
        """
        def make_env():
            env = TowerBloxEnv(
                config_dir=str(self.config_loader.config_dir),
                render_mode=render_mode,
                show_visualization=show_visualization,
            )
            return env
        
        # Create vectorized environment
        env = DummyVecEnv([make_env])
        
        # Transpose image for CNN policy (channel-last to channel-first)
        env = VecTransposeImage(env)
        
        logger.info(f"Environment created: obs={env.observation_space}, act={env.action_space}")
        
        return env
    
    def create_model(
        self,
        env: DummyVecEnv,
        load_path: Optional[str] = None,
    ) -> PPO:
        """
        Create or load PPO model.
        
        Args:
            env: Vectorized environment
            load_path: Path to load existing model from
            
        Returns:
            PPO model instance
        """
        if load_path:
            logger.info(f"Loading model from: {load_path}")
            model = PPO.load(load_path, env=env, device=self.device)
        else:
            # Get hyperparameters from config
            hyperparams = self.ppo_config.get('hyperparameters', {})
            
            model = PPO(
                policy="CnnPolicy",
                env=env,
                learning_rate=hyperparams.get('learning_rate', 3e-4),
                n_steps=hyperparams.get('n_steps', 2048),
                batch_size=hyperparams.get('batch_size', 64),
                n_epochs=hyperparams.get('n_epochs', 10),
                gamma=hyperparams.get('gamma', 0.99),
                gae_lambda=hyperparams.get('gae_lambda', 0.95),
                clip_range=hyperparams.get('clip_range', 0.2),
                clip_range_vf=hyperparams.get('clip_range_vf'),
                ent_coef=hyperparams.get('ent_coef', 0.01),
                vf_coef=hyperparams.get('vf_coef', 0.5),
                max_grad_norm=hyperparams.get('max_grad_norm', 0.5),
                device=self.device,
                verbose=1,
                tensorboard_log=str(self.tensorboard_dir),
            )
            
            logger.info("Created new PPO model with CnnPolicy")
        
        return model
    
    def create_callbacks(
        self,
        eval_env: Optional[DummyVecEnv] = None,
    ) -> CallbackList:
        """
        Create training callbacks.
        
        Args:
            eval_env: Optional evaluation environment
            
        Returns:
            Callback list for training
        """
        training_config = self.ppo_config.get('training', {})
        
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=training_config.get('save_freq', 50000),
            save_path=str(self.checkpoint_dir),
            name_prefix="ppo_towerblox",
            keep_n_best=training_config.get('keep_n_best', 5),
        )
        callbacks.append(checkpoint_callback)
        
        # GPU monitor callback
        if self.gpu_monitor.is_available():
            gpu_callback = GPUMonitorCallback(
                log_freq=1000,
                memory_warning_threshold=90.0,
            )
            callbacks.append(gpu_callback)
        
        # Training logger callback
        logger_callback = TrainingLoggerCallback(
            log_freq=training_config.get('log_interval', 10),
        )
        callbacks.append(logger_callback)
        
        # Evaluation callback
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(self.checkpoint_dir / "best"),
                log_path=str(self.log_dir / "eval_logs"),
                eval_freq=training_config.get('eval_freq', 25000),
                n_eval_episodes=training_config.get('n_eval_episodes', 10),
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)
        
        return CallbackList(callbacks)
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        load_path: Optional[str] = None,
        reset_timesteps: bool = False,
    ) -> PPO:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training steps (uses config if None)
            load_path: Path to load existing model
            reset_timesteps: Reset timestep counter when loading
            
        Returns:
            Trained PPO model
        """
        # Get total timesteps from config if not provided
        if total_timesteps is None:
            training_config = self.ppo_config.get('training', {})
            total_timesteps = training_config.get('total_timesteps', 1000000)
        
        logger.info(f"Starting training for {total_timesteps:,} timesteps")
        
        # Log GPU info
        if self.gpu_monitor.is_available():
            stats = self.gpu_monitor.get_stats()
            logger.info(
                f"GPU Memory: {stats['memory_used_mb']:.0f}/{stats['memory_total_mb']:.0f} MB"
            )
        
        # Create environments
        self.env = self.create_env()
        self.eval_env = self.create_env()
        
        # Create or load model
        self.model = self.create_model(self.env, load_path=load_path)
        
        # Create callbacks
        callbacks = self.create_callbacks(eval_env=self.eval_env)
        
        # Start training
        self._training_start_time = time.time()
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                reset_num_timesteps=reset_timesteps,
                tb_log_name="PPO",
                progress_bar=True,
            )
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        finally:
            # Save final model
            final_path = self.checkpoint_dir / "ppo_towerblox_final"
            self.model.save(str(final_path))
            logger.info(f"Final model saved: {final_path}")
            
            # Training summary
            training_time = time.time() - self._training_start_time
            logger.info(
                f"Training completed: {self.model.num_timesteps:,} steps "
                f"in {training_time/3600:.1f} hours"
            )
        
        return self.model
    
    def evaluate(
        self,
        model_path: str,
        n_episodes: int = 100,
        render: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to model file
            n_episodes: Number of evaluation episodes
            render: Whether to render episodes
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model: {model_path}")
        
        # Create evaluation environment
        render_mode = 'human' if render else None
        eval_env = self.create_env(render_mode=render_mode)
        
        # Load model
        model = PPO.load(model_path, env=eval_env, device=self.device)
        
        # Run evaluation
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            logger.info(
                f"Episode {episode + 1}/{n_episodes}: "
                f"reward={episode_reward:.2f}, length={episode_length}"
            )
        
        # Calculate statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'n_episodes': n_episodes,
        }
        
        logger.info(
            f"Evaluation complete: "
            f"reward={results['mean_reward']:.2f} ± {results['std_reward']:.2f}"
        )
        
        eval_env.close()
        
        return results
    
    def close(self) -> None:
        """Clean up trainer resources."""
        if self.env:
            self.env.close()
        if self.eval_env:
            self.eval_env.close()
        
        self.gpu_monitor.shutdown()
        
        logger.info("Trainer closed")
