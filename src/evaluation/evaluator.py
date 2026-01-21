"""
Evaluator Module
Test and evaluate trained RL models.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

from ..environment.mobile_game_env import TowerBloxEnv
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Evaluates trained RL models on Tower Bloxx.
    
    Provides methods for:
    - Single model evaluation
    - Model comparison
    - Performance analysis
    - Recording gameplay
    
    Attributes:
        model_path: Path to model file
        model: Loaded PPO model
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to model file
            device: Device for inference
        """
        self.model_path = model_path
        self.device = device
        self.model: Optional[PPO] = None
        self.env: Optional[TowerBloxEnv] = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a model from file.
        
        Args:
            model_path: Path to model file
        """
        logger.info(f"Loading model: {model_path}")
        self.model_path = model_path
        self.model = PPO.load(model_path, device=self.device)
    
    def create_env(self, render_mode: Optional[str] = None) -> TowerBloxEnv:
        """
        Create evaluation environment.
        
        Args:
            render_mode: Rendering mode
            
        Returns:
            Tower Bloxx environment
        """
        env = TowerBloxEnv(render_mode=render_mode)
        return env
    
    def run_episode(
        self,
        deterministic: bool = True,
        render: bool = False,
        record: bool = False,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Run a single evaluation episode.
        
        Args:
            deterministic: Use deterministic actions
            render: Render the episode
            record: Record frames
            
        Returns:
            Tuple of (reward, length, info)
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        render_mode = 'human' if render else None
        env = self.create_env(render_mode=render_mode)
        
        obs, info = env.reset()
        done = False
        truncated = False
        
        episode_reward = 0.0
        episode_length = 0
        frames = [] if record else None
        actions = []
        rewards = []
        
        while not (done or truncated):
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=deterministic)
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            actions.append(int(action))
            rewards.append(float(reward))
            
            if record:
                frames.append(env.render())
            
            if render:
                time.sleep(0.03)  # Slow down for human viewing
        
        env.close()
        
        episode_info = {
            'reward': episode_reward,
            'length': episode_length,
            'actions': actions,
            'rewards': rewards,
            'frames': frames,
            'final_info': info,
        }
        
        return episode_reward, episode_length, episode_info
    
    def evaluate(
        self,
        n_episodes: int = 100,
        deterministic: bool = True,
        render: bool = False,
        progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model over multiple episodes.
        
        Args:
            n_episodes: Number of episodes
            deterministic: Use deterministic actions
            render: Render episodes
            progress: Show progress
            
        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        logger.info(f"Evaluating over {n_episodes} episodes...")
        
        rewards = []
        lengths = []
        action_counts = {0: 0, 1: 0}
        
        start_time = time.time()
        
        for i in range(n_episodes):
            reward, length, info = self.run_episode(
                deterministic=deterministic,
                render=render,
            )
            
            rewards.append(reward)
            lengths.append(length)
            
            # Count actions
            for action in info.get('actions', []):
                action_counts[action] = action_counts.get(action, 0) + 1
            
            if progress and (i + 1) % 10 == 0:
                logger.info(
                    f"Progress: {i + 1}/{n_episodes} | "
                    f"Mean reward: {np.mean(rewards):.2f}"
                )
        
        elapsed = time.time() - start_time
        
        # Calculate statistics
        results = {
            'n_episodes': n_episodes,
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'median_reward': float(np.median(rewards)),
            'mean_length': float(np.mean(lengths)),
            'std_length': float(np.std(lengths)),
            'total_steps': int(sum(lengths)),
            'elapsed_time': elapsed,
            'episodes_per_second': n_episodes / elapsed,
            'action_distribution': {
                'wait': action_counts.get(0, 0),
                'tap': action_counts.get(1, 0),
            },
            'rewards': rewards,
            'lengths': lengths,
        }
        
        logger.info(
            f"Evaluation complete: "
            f"reward={results['mean_reward']:.2f} ± {results['std_reward']:.2f}, "
            f"length={results['mean_length']:.1f}"
        )
        
        return results
    
    def compare_models(
        self,
        model_paths: List[str],
        n_episodes: int = 50,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models.
        
        Args:
            model_paths: List of model file paths
            n_episodes: Episodes per model
            
        Returns:
            Dictionary mapping model paths to results
        """
        results = {}
        
        for path in model_paths:
            logger.info(f"Evaluating: {path}")
            self.load_model(path)
            results[path] = self.evaluate(n_episodes=n_episodes, render=False)
        
        # Rank models
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]['mean_reward'],
            reverse=True,
        )
        
        logger.info("\n=== Model Comparison ===")
        for rank, (path, result) in enumerate(sorted_models, 1):
            logger.info(
                f"{rank}. {Path(path).name}: "
                f"reward={result['mean_reward']:.2f} ± {result['std_reward']:.2f}"
            )
        
        return results
    
    def test_random_baseline(self, n_episodes: int = 100) -> Dict[str, Any]:
        """
        Test random action baseline.
        
        Args:
            n_episodes: Number of episodes
            
        Returns:
            Baseline results
        """
        logger.info("Testing random baseline...")
        
        env = self.create_env()
        
        rewards = []
        lengths = []
        
        for i in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            episode_length = 0
            
            while not (done or truncated):
                action = env.action_space.sample()
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
        
        env.close()
        
        results = {
            'n_episodes': n_episodes,
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'mean_length': float(np.mean(lengths)),
        }
        
        logger.info(
            f"Random baseline: "
            f"reward={results['mean_reward']:.2f} ± {results['std_reward']:.2f}"
        )
        
        return results
