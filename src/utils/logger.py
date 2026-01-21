"""
Logging Module
Custom logging setup with colored console output and file logging.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from colorama import Fore, Style, init as colorama_init

# Initialize colorama for Windows support
colorama_init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log levels in console output.
    """
    
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get color for log level
        color = self.COLORS.get(record.levelno, '')
        
        # Format the message
        log_message = super().format(record)
        
        # Add color to level name
        if color:
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
            log_message = super().format(record)
        
        return log_message


class TowerBloxLogger:
    """
    Custom logger for Tower Bloxx RL Agent.
    
    Provides structured logging with:
    - Colored console output
    - File logging with rotation
    - Training metrics logging
    """
    
    _loggers: dict = {}
    _initialized: bool = False
    _log_dir: Optional[Path] = None
    
    @classmethod
    def setup(
        cls,
        log_dir: Optional[str] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ) -> None:
        """
        Set up the logging system.
        
        Args:
            log_dir: Directory for log files. Defaults to ./logs/training_logs
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        if cls._initialized:
            return
        
        # Set up log directory
        if log_dir is None:
            cls._log_dir = Path(__file__).parent.parent.parent / "logs" / "training_logs"
        else:
            cls._log_dir = Path(log_dir)
        
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = cls._log_dir / f"training_{timestamp}.log"
        
        # Set up root logger
        root_logger = logging.getLogger("towerblox")
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        console_handler.setFormatter(ColoredFormatter(console_format, datefmt="%H:%M:%S"))
        root_logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        file_handler.setFormatter(logging.Formatter(file_format))
        root_logger.addHandler(file_handler)
        
        cls._initialized = True
        root_logger.info(f"Logging initialized. Log file: {log_file}")
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name (usually __name__ of the calling module)
            
        Returns:
            Logger instance
        """
        if not cls._initialized:
            cls.setup()
        
        if name not in cls._loggers:
            logger = logging.getLogger(f"towerblox.{name}")
            cls._loggers[name] = logger
        
        return cls._loggers[name]


def setup_logger(
    log_dir: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """
    Initialize the logging system.
    
    Args:
        log_dir: Directory for log files
        console_level: Console logging level
        file_level: File logging level
    """
    TowerBloxLogger.setup(log_dir, console_level, file_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Configured logger
    """
    return TowerBloxLogger.get_logger(name)


class TrainingMetricsLogger:
    """
    Specialized logger for training metrics.
    
    Logs training progress, rewards, and performance metrics
    in a structured format for easy parsing.
    """
    
    def __init__(self, name: str = "training"):
        """
        Initialize training metrics logger.
        
        Args:
            name: Logger name
        """
        self.logger = get_logger(name)
        self.episode_count = 0
        self.step_count = 0
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        info: Optional[dict] = None,
    ) -> None:
        """
        Log episode completion.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length in steps
            info: Additional info dictionary
        """
        self.episode_count = episode
        msg = f"Episode {episode:6d} | Reward: {reward:8.2f} | Length: {length:5d}"
        
        if info:
            for key, value in info.items():
                if isinstance(value, float):
                    msg += f" | {key}: {value:.4f}"
                else:
                    msg += f" | {key}: {value}"
        
        self.logger.info(msg)
    
    def log_step(
        self,
        step: int,
        reward: float,
        action: int,
        info: Optional[dict] = None,
    ) -> None:
        """
        Log individual step (at DEBUG level).
        
        Args:
            step: Step number
            reward: Step reward
            action: Action taken
            info: Additional info
        """
        self.step_count = step
        msg = f"Step {step:8d} | Action: {action} | Reward: {reward:6.3f}"
        
        if info:
            msg += f" | {info}"
        
        self.logger.debug(msg)
    
    def log_training_update(
        self,
        step: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        learning_rate: float,
    ) -> None:
        """
        Log PPO training update.
        
        Args:
            step: Current training step
            policy_loss: Policy loss value
            value_loss: Value function loss
            entropy: Entropy bonus
            learning_rate: Current learning rate
        """
        self.logger.info(
            f"Update @ {step:8d} | "
            f"Policy Loss: {policy_loss:.6f} | "
            f"Value Loss: {value_loss:.6f} | "
            f"Entropy: {entropy:.6f} | "
            f"LR: {learning_rate:.2e}"
        )
    
    def log_evaluation(
        self,
        step: int,
        mean_reward: float,
        std_reward: float,
        n_episodes: int,
    ) -> None:
        """
        Log evaluation results.
        
        Args:
            step: Current training step
            mean_reward: Mean episode reward
            std_reward: Standard deviation of rewards
            n_episodes: Number of evaluation episodes
        """
        self.logger.info(
            f"Evaluation @ {step:8d} | "
            f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f} | "
            f"Episodes: {n_episodes}"
        )
    
    def log_checkpoint(self, step: int, path: str) -> None:
        """
        Log checkpoint save.
        
        Args:
            step: Training step
            path: Path where checkpoint was saved
        """
        self.logger.info(f"Checkpoint saved @ step {step}: {path}")
    
    def log_gpu_stats(self, gpu_util: float, memory_used: float, memory_total: float) -> None:
        """
        Log GPU statistics.
        
        Args:
            gpu_util: GPU utilization percentage
            memory_used: GPU memory used in MB
            memory_total: Total GPU memory in MB
        """
        self.logger.debug(
            f"GPU Stats | Utilization: {gpu_util:.1f}% | "
            f"Memory: {memory_used:.0f}/{memory_total:.0f} MB ({100*memory_used/memory_total:.1f}%)"
        )
