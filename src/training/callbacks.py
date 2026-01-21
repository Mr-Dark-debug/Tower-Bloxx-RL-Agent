"""
Training Callbacks Module
Custom callbacks for Stable-Baselines3 training.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from ..utils.logger import get_logger, TrainingMetricsLogger
from ..utils.gpu_monitor import GPUMonitor

logger = get_logger(__name__)


class CheckpointCallback(BaseCallback):
    """
    Callback for saving model checkpoints during training.
    
    Saves checkpoints at regular intervals and keeps track of best models.
    
    Attributes:
        save_freq: Save checkpoint every N steps
        save_path: Directory to save checkpoints
        keep_n_best: Number of best models to keep
    """
    
    def __init__(
        self,
        save_freq: int = 50000,
        save_path: str = "./logs/checkpoints",
        name_prefix: str = "model",
        keep_n_best: int = 5,
        verbose: int = 1,
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            save_freq: Checkpoint frequency in steps
            save_path: Directory for checkpoints
            name_prefix: Prefix for checkpoint names
            keep_n_best: Number of best models to keep
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.keep_n_best = keep_n_best
        
        # Track best models
        self._best_rewards: list = []
        self._best_paths: list = []
        
        # Ensure save directory exists
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.save_freq == 0:
            self._save_checkpoint()
        return True
    
    def _save_checkpoint(self) -> None:
        """Save a checkpoint."""
        step = self.num_timesteps
        path = self.save_path / f"{self.name_prefix}_{step}_steps"
        
        self.model.save(str(path))
        
        if self.verbose > 0:
            logger.info(f"Checkpoint saved: {path}")
    
    def save_best(self, mean_reward: float) -> None:
        """
        Save model if it's among the best.
        
        Args:
            mean_reward: Mean episode reward
        """
        # Check if this is a new best
        should_save = len(self._best_rewards) < self.keep_n_best
        
        if not should_save and self._best_rewards:
            should_save = mean_reward > min(self._best_rewards)
        
        if should_save:
            step = self.num_timesteps
            path = self.save_path / f"{self.name_prefix}_best_{step}_steps"
            
            self.model.save(str(path))
            
            self._best_rewards.append(mean_reward)
            self._best_paths.append(str(path))
            
            # Remove worst if exceeding limit
            if len(self._best_rewards) > self.keep_n_best:
                min_idx = np.argmin(self._best_rewards)
                
                # Delete old file
                old_path = Path(self._best_paths[min_idx])
                if old_path.exists():
                    old_path.unlink()
                if old_path.with_suffix('.zip').exists():
                    old_path.with_suffix('.zip').unlink()
                
                del self._best_rewards[min_idx]
                del self._best_paths[min_idx]
            
            if self.verbose > 0:
                logger.info(f"New best model saved: {path} (reward: {mean_reward:.2f})")


class GPUMonitorCallback(BaseCallback):
    """
    Callback for monitoring GPU usage during training.
    
    Logs GPU utilization and memory usage to TensorBoard
    and provides warnings when memory is high.
    """
    
    def __init__(
        self,
        log_freq: int = 1000,
        memory_warning_threshold: float = 90.0,
        verbose: int = 1,
    ):
        """
        Initialize GPU monitor callback.
        
        Args:
            log_freq: Logging frequency in steps
            memory_warning_threshold: Memory % to trigger warning
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.log_freq = log_freq
        self.memory_warning_threshold = memory_warning_threshold
        self.gpu_monitor = GPUMonitor()
    
    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.log_freq == 0:
            if self.gpu_monitor.is_available():
                stats = self.gpu_monitor.get_stats()
                
                # Log to TensorBoard
                self.logger.record("gpu/utilization", stats['utilization'])
                self.logger.record("gpu/memory_used_mb", stats['memory_used_mb'])
                self.logger.record("gpu/memory_percent", stats['memory_percent'])
                self.logger.record("gpu/temperature", stats['temperature'])
                
                # Check for high memory usage
                if stats['memory_percent'] > self.memory_warning_threshold:
                    logger.warning(
                        f"High GPU memory usage: {stats['memory_percent']:.1f}%"
                    )
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at end of training."""
        self.gpu_monitor.shutdown()


class TrainingLoggerCallback(BaseCallback):
    """
    Callback for detailed training logging.
    
    Logs episode rewards, training metrics, and provides
    formatted console output.
    """
    
    def __init__(
        self,
        log_freq: int = 10,
        verbose: int = 1,
    ):
        """
        Initialize training logger callback.
        
        Args:
            log_freq: Logging frequency (episodes)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.log_freq = log_freq
        self.metrics_logger = TrainingMetricsLogger()
        
        # Episode tracking
        self._episode_rewards: list = []
        self._episode_lengths: list = []
        self._episode_count = 0
        
        # Timing
        self._start_time = time.time()
        self._last_log_time = time.time()
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Check for episode completion
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    self._episode_count += 1
                    
                    # Get episode info
                    infos = self.locals.get('infos', [])
                    if i < len(infos) and 'episode' in infos[i]:
                        ep_info = infos[i]['episode']
                        reward = ep_info.get('r', 0)
                        length = ep_info.get('l', 0)
                        
                        self._episode_rewards.append(reward)
                        self._episode_lengths.append(length)
        
        # Log periodically
        if self._episode_count > 0 and self._episode_count % self.log_freq == 0:
            self._log_progress()
        
        return True
    
    def _log_progress(self) -> None:
        """Log training progress."""
        if not self._episode_rewards:
            return
        
        # Calculate statistics
        recent_rewards = self._episode_rewards[-100:]
        recent_lengths = self._episode_lengths[-100:]
        
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        mean_length = np.mean(recent_lengths)
        
        # Calculate FPS
        elapsed = time.time() - self._last_log_time
        steps_since_log = self.num_timesteps
        fps = steps_since_log / max(elapsed, 1)
        
        # Log to console
        self.metrics_logger.log_evaluation(
            step=self.num_timesteps,
            mean_reward=mean_reward,
            std_reward=std_reward,
            n_episodes=len(recent_rewards),
        )
        
        # Log to TensorBoard
        self.logger.record("rollout/ep_rew_mean", mean_reward)
        self.logger.record("rollout/ep_rew_std", std_reward)
        self.logger.record("rollout/ep_len_mean", mean_length)
        self.logger.record("time/fps", fps)
        self.logger.record("time/episodes", self._episode_count)
        
        self._last_log_time = time.time()


class EarlyStoppingCallback(BaseCallback):
    """
    Callback for early stopping when training plateaus.
    
    Stops training if mean reward hasn't improved for N evaluations.
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.01,
        verbose: int = 1,
    ):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of evaluations without improvement
            min_delta: Minimum improvement to reset patience
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.patience = patience
        self.min_delta = min_delta
        
        self._best_reward = -np.inf
        self._patience_counter = 0
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Check if this is an evaluation step
        # (This should be used with EvalCallback)
        if 'eval_mean_reward' in self.locals:
            mean_reward = self.locals['eval_mean_reward']
            
            if mean_reward > self._best_reward + self.min_delta:
                self._best_reward = mean_reward
                self._patience_counter = 0
                
                if self.verbose > 0:
                    logger.info(f"New best reward: {mean_reward:.2f}")
            else:
                self._patience_counter += 1
                
                if self._patience_counter >= self.patience:
                    if self.verbose > 0:
                        logger.info(
                            f"Early stopping: no improvement for {self.patience} evaluations"
                        )
                    return False  # Stop training
        
        return True


class WandbCallback(BaseCallback):
    """
    Callback for Weights & Biases logging.
    
    Logs training metrics to wandb for experiment tracking.
    """
    
    def __init__(
        self,
        project_name: str = "towerblox-rl",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        log_freq: int = 100,
        verbose: int = 1,
    ):
        """
        Initialize wandb callback.
        
        Args:
            project_name: Wandb project name
            run_name: Wandb run name
            config: Configuration to log
            log_freq: Logging frequency in steps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.project_name = project_name
        self.run_name = run_name
        self.config = config
        self.log_freq = log_freq
        
        self._wandb_initialized = False
    
    def _on_training_start(self) -> None:
        """Called at start of training."""
        try:
            import wandb
            
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=self.config,
                reinit=True,
            )
            
            self._wandb_initialized = True
            logger.info(f"Wandb initialized: {self.project_name}/{self.run_name}")
            
        except ImportError:
            logger.warning("wandb not installed. Wandb logging disabled.")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    def _on_step(self) -> bool:
        """Called at each step."""
        if not self._wandb_initialized:
            return True
        
        if self.n_calls % self.log_freq == 0:
            import wandb
            
            # Log basic metrics
            wandb.log({
                'timesteps': self.num_timesteps,
            }, step=self.num_timesteps)
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at end of training."""
        if self._wandb_initialized:
            import wandb
            wandb.finish()
