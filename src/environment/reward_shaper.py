"""
Reward Shaper Module
Calculates rewards based on game state changes.
"""

from typing import Dict, Optional, Tuple
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RewardShaper:
    """
    Shapes and calculates rewards for Tower Bloxx environment.
    
    Reward components:
    - Block placement quality (perfect, good, ok, wobbly)
    - Height bonus (building higher tower)
    - Coin collection
    - Step penalty (encourage faster decisions)
    - Game over penalty
    
    Attributes:
        config: Reward configuration dictionary
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize reward shaper.
        
        Args:
            config: Reward configuration. Uses defaults if None.
        """
        # Default reward values
        self.rewards = {
            'perfect_placement': 5.0,
            'good_placement': 2.0,
            'ok_placement': 1.0,
            'wobbly_placement': 0.3,
            'game_over': -10.0,
            'step_penalty': -0.01,
            'height_bonus': 0.5,
            'coin_collected': 1.0,
            'idle_penalty': -0.05,  # Penalty for too much waiting
        }
        
        # Override with config if provided
        if config is not None:
            self.rewards.update(config)
        
        # State tracking
        self._previous_height = 0
        self._previous_coins = 0
        self._steps_since_action = 0
        self._total_reward = 0.0
        
        # Statistics
        self._reward_components: Dict[str, float] = {}
        self._episode_rewards: list = []
        
        logger.info(f"Reward shaper initialized with config: {self.rewards}")
    
    def calculate_reward(
        self,
        current_state: Dict,
        previous_state: Optional[Dict],
        action: int,
        done: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward for a step.
        
        Args:
            current_state: Current game state dictionary
            previous_state: Previous game state dictionary
            action: Action taken
            done: Whether episode is done
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        components = {}
        total_reward = 0.0
        
        # Step penalty (always applied)
        step_penalty = self.rewards['step_penalty']
        components['step_penalty'] = step_penalty
        total_reward += step_penalty
        
        # Check for game over
        if done and current_state.get('game_over', False):
            game_over_penalty = self.rewards['game_over']
            components['game_over'] = game_over_penalty
            total_reward += game_over_penalty
            
            self._reset_tracking()
            return total_reward, components
        
        # Track waiting behavior
        if action == 0:  # Wait action
            self._steps_since_action += 1
            
            # Penalty for excessive waiting
            if self._steps_since_action > 30:  # ~1 second at 30 FPS
                idle_penalty = self.rewards['idle_penalty']
                components['idle_penalty'] = idle_penalty
                total_reward += idle_penalty
        else:
            self._steps_since_action = 0
        
        # Height-based reward (building taller tower)
        current_height = current_state.get('tower_height', 0)
        if current_height > self._previous_height:
            height_diff = current_height - self._previous_height
            height_bonus = self.rewards['height_bonus'] * height_diff
            components['height_bonus'] = height_bonus
            total_reward += height_bonus
            self._previous_height = current_height
        
        # Block placement quality (when a tap action is taken)
        if action == 1:  # Tap action
            placement_quality = current_state.get('placement_quality', None)
            
            if placement_quality == 'perfect':
                reward = self.rewards['perfect_placement']
                components['perfect_placement'] = reward
                total_reward += reward
            elif placement_quality == 'good':
                reward = self.rewards['good_placement']
                components['good_placement'] = reward
                total_reward += reward
            elif placement_quality == 'ok':
                reward = self.rewards['ok_placement']
                components['ok_placement'] = reward
                total_reward += reward
            elif placement_quality == 'wobbly':
                reward = self.rewards['wobbly_placement']
                components['wobbly_placement'] = reward
                total_reward += reward
        
        # Coin collection
        current_coins = current_state.get('coins', 0)
        if current_coins > self._previous_coins:
            coin_diff = current_coins - self._previous_coins
            coin_reward = self.rewards['coin_collected'] * coin_diff
            components['coin_collected'] = coin_reward
            total_reward += coin_reward
            self._previous_coins = current_coins
        
        # Update statistics
        self._total_reward += total_reward
        self._reward_components = components
        
        return total_reward, components
    
    def calculate_placement_quality(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray,
    ) -> str:
        """
        Estimate placement quality from frame comparison.
        
        Args:
            current_frame: Current game frame
            previous_frame: Previous game frame
            
        Returns:
            Placement quality: 'perfect', 'good', 'ok', 'wobbly', or 'none'
        """
        # Calculate frame difference
        if len(current_frame.shape) == 3:
            diff = np.abs(current_frame.astype(float) - previous_frame.astype(float))
            diff = np.mean(diff, axis=-1)  # Average across channels
        else:
            diff = np.abs(current_frame.astype(float) - previous_frame.astype(float))
        
        # Measure change in different regions
        h, w = diff.shape[:2]
        
        # Tower region (center-bottom of frame)
        tower_region = diff[h//2:, w//4:3*w//4]
        tower_change = np.mean(tower_region)
        
        # If minimal change, no placement occurred
        if tower_change < 0.01:
            return 'none'
        
        # Estimate quality based on change pattern
        # High concentrated change = perfect alignment
        # Spread out change = wobble
        change_variance = np.var(tower_region)
        change_max = np.max(tower_region)
        
        if change_variance < 0.01 and change_max > 0.1:
            return 'perfect'
        elif change_variance < 0.02:
            return 'good'
        elif change_variance < 0.05:
            return 'ok'
        else:
            return 'wobbly'
    
    def estimate_tower_height(self, frame: np.ndarray) -> int:
        """
        Estimate tower height from frame.
        
        Args:
            frame: Game frame (grayscale or color)
            
        Returns:
            Estimated tower height in blocks
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = np.mean(frame, axis=-1)
        else:
            gray = frame
        
        h, w = gray.shape
        
        # Look for horizontal lines of consistent color (block edges)
        # in the lower 2/3 of the frame
        tower_region = gray[h//3:, w//4:3*w//4]
        
        # Edge detection - look for horizontal edges
        edges = np.abs(np.diff(tower_region, axis=0))
        
        # Count significant horizontal edges
        edge_threshold = 0.1 if gray.max() <= 1.0 else 25
        significant_edges = np.sum(edges > edge_threshold, axis=1)
        
        # Count rows with significant edge activity
        active_rows = np.sum(significant_edges > tower_region.shape[1] * 0.3)
        
        # Estimate blocks (rough approximation)
        # Each block is roughly 1/15 of the tower region height
        block_size = tower_region.shape[0] / 15
        estimated_height = max(0, int(active_rows / block_size))
        
        return estimated_height
    
    def detect_game_over(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray,
    ) -> bool:
        """
        Detect if game over occurred.
        
        Args:
            current_frame: Current game frame
            previous_frame: Previous game frame
            
        Returns:
            True if game over detected
        """
        # Game over typically shows dark overlay or significant screen change
        if len(current_frame.shape) == 3:
            current_brightness = np.mean(current_frame)
            previous_brightness = np.mean(previous_frame)
        else:
            current_brightness = np.mean(current_frame)
            previous_brightness = np.mean(previous_frame)
        
        # Significant darkening often indicates game over overlay
        brightness_drop = previous_brightness - current_brightness
        relative_drop = brightness_drop / (previous_brightness + 0.001)
        
        if relative_drop > 0.3:  # 30% brightness drop
            return True
        
        # Check for specific patterns (buttons appearing)
        # This would need calibration based on actual game over screens
        
        return False
    
    def _reset_tracking(self) -> None:
        """Reset state tracking for new episode."""
        self._previous_height = 0
        self._previous_coins = 0
        self._steps_since_action = 0
    
    def reset(self) -> None:
        """Reset reward shaper for new episode."""
        self._reset_tracking()
        self._total_reward = 0.0
        self._reward_components = {}
    
    def get_episode_reward(self) -> float:
        """Get total reward for current episode."""
        return self._total_reward
    
    def get_reward_breakdown(self) -> Dict[str, float]:
        """Get breakdown of reward components."""
        return self._reward_components.copy()
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get reward statistics.
        
        Returns:
            Dictionary with reward statistics
        """
        return {
            'episode_reward': self._total_reward,
            'previous_height': self._previous_height,
            'previous_coins': self._previous_coins,
            'steps_since_action': self._steps_since_action,
        }
