"""
Visualizer Module
Visualization tools for RL agent performance.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Try to import visualization libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class Visualizer:
    """
    Visualization tools for training and evaluation.
    
    Provides:
    - Training curve plots
    - Action distribution charts
    - Gameplay recording
    - Performance comparisons
    """
    
    def __init__(self, output_dir: str = "./logs/visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curve(
        self,
        rewards: List[float],
        window_size: int = 100,
        title: str = "Training Progress",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot training reward curve.
        
        Args:
            rewards: List of episode rewards
            window_size: Rolling average window
            title: Plot title
            save_path: Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot raw rewards
        episodes = list(range(1, len(rewards) + 1))
        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        
        # Plot rolling average
        if len(rewards) >= window_size:
            rolling_avg = np.convolve(
                rewards,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            rolling_episodes = list(range(window_size, len(rewards) + 1))
            ax.plot(
                rolling_episodes,
                rolling_avg,
                color='red',
                linewidth=2,
                label=f'{window_size}-Episode Average'
            )
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Training curve saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_action_distribution(
        self,
        action_counts: Dict[str, int],
        title: str = "Action Distribution",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot action distribution as pie chart.
        
        Args:
            action_counts: Dictionary of action names to counts
            title: Plot title
            save_path: Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        labels = list(action_counts.keys())
        sizes = list(action_counts.values())
        colors = ['#66b3ff', '#99ff99', '#ffcc99', '#ff9999']
        
        ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors[:len(labels)],
            startangle=90,
        )
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Action distribution saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_evaluation_results(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create comprehensive evaluation visualization.
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available for plotting")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # Reward distribution
        ax1 = fig.add_subplot(gs[0, 0])
        rewards = results.get('rewards', [])
        ax1.hist(rewards, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(
            results['mean_reward'],
            color='red',
            linestyle='--',
            label=f"Mean: {results['mean_reward']:.2f}"
        )
        ax1.set_xlabel('Reward')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Reward Distribution')
        ax1.legend()
        
        # Episode lengths
        ax2 = fig.add_subplot(gs[0, 1])
        lengths = results.get('lengths', [])
        ax2.hist(lengths, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax2.axvline(
            results['mean_length'],
            color='red',
            linestyle='--',
            label=f"Mean: {results['mean_length']:.1f}"
        )
        ax2.set_xlabel('Episode Length')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Episode Length Distribution')
        ax2.legend()
        
        # Action distribution
        ax3 = fig.add_subplot(gs[0, 2])
        action_dist = results.get('action_distribution', {})
        if action_dist:
            actions = list(action_dist.keys())
            counts = list(action_dist.values())
            ax3.bar(actions, counts, color=['#66b3ff', '#99ff99'])
            ax3.set_xlabel('Action')
            ax3.set_ylabel('Count')
            ax3.set_title('Action Distribution')
        
        # Reward over episodes
        ax4 = fig.add_subplot(gs[1, :])
        if rewards:
            ax4.plot(rewards, alpha=0.5, color='blue')
            
            # Add rolling average
            window = min(10, len(rewards) // 4) or 1
            rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax4.plot(
                range(window-1, len(rewards)),
                rolling,
                color='red',
                linewidth=2,
                label=f'{window}-Episode Average'
            )
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Reward')
            ax4.set_title('Evaluation Progress')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(
            f"Evaluation Results: {results['n_episodes']} Episodes | "
            f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}",
            fontsize=14,
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Evaluation plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_models_plot(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot model comparison bar chart.
        
        Args:
            results: Dictionary mapping model names to results
            save_path: Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        model_names = [Path(p).stem for p in results.keys()]
        means = [r['mean_reward'] for r in results.values()]
        stds = [r['std_reward'] for r in results.values()]
        
        x = np.arange(len(model_names))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        
        # Color best model
        best_idx = np.argmax(means)
        bars[best_idx].set_color('green')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Model comparison saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 30,
    ) -> None:
        """
        Save frames as video.
        
        Args:
            frames: List of RGB frames
            output_path: Output video path
            fps: Frames per second
        """
        if not CV2_AVAILABLE:
            logger.warning("cv2 not available for video saving")
            return
        
        if not frames:
            logger.warning("No frames to save")
            return
        
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        
        out.release()
        logger.info(f"Video saved: {output_path} ({len(frames)} frames)")
