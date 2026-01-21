"""
Frame Stacker Wrapper
Gymnasium wrapper for frame stacking.
"""

from collections import deque
from typing import Any, Dict, Optional, Tuple, SupportsFloat

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FrameStackWrapper(gym.ObservationWrapper):
    """
    Gymnasium wrapper that stacks consecutive frames.
    
    Stacking frames provides temporal information to the network,
    allowing it to perceive motion and velocity.
    
    Attributes:
        n_frames: Number of frames to stack
    """
    
    def __init__(
        self,
        env: gym.Env,
        n_frames: int = 4,
        stack_axis: int = -1,
    ):
        """
        Initialize frame stack wrapper.
        
        Args:
            env: Gymnasium environment to wrap
            n_frames: Number of frames to stack
            stack_axis: Axis along which to stack (-1 for last)
        """
        super().__init__(env)
        
        self.n_frames = n_frames
        self.stack_axis = stack_axis
        self._frames: deque = deque(maxlen=n_frames)
        
        # Update observation space
        old_space = env.observation_space
        
        if isinstance(old_space, spaces.Box):
            low = np.repeat(old_space.low, n_frames, axis=stack_axis)
            high = np.repeat(old_space.high, n_frames, axis=stack_axis)
            
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=old_space.dtype,
            )
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Stack observations.
        
        Args:
            observation: Single observation
            
        Returns:
            Stacked observations
        """
        self._frames.append(observation)
        return np.concatenate(list(self._frames), axis=self.stack_axis)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment and frame buffer.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (stacked_observation, info)
        """
        observation, info = self.env.reset(seed=seed, options=options)
        
        # Fill buffer with initial observation
        self._frames.clear()
        for _ in range(self.n_frames):
            self._frames.append(observation)
        
        return self.observation(observation), info


class SkipFrameWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that repeats actions for n frames.
    
    Reduces computational cost and provides more meaningful
    state transitions for many games.
    """
    
    def __init__(
        self,
        env: gym.Env,
        skip: int = 4,
    ):
        """
        Initialize frame skip wrapper.
        
        Args:
            env: Gymnasium environment to wrap
            skip: Number of frames to skip (repeat action)
        """
        super().__init__(env)
        self._skip = skip
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute action for skip frames.
        
        Args:
            action: Action to repeat
            
        Returns:
            Tuple of (observation, accumulated_reward, terminated, truncated, info)
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for i in range(self._skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return observation, total_reward, terminated, truncated, info


class MaxAndSkipFrameWrapper(gym.Wrapper):
    """
    Wrapper that takes max over last observations and skips frames.
    
    Useful for games with flickering sprites (like Atari).
    """
    
    def __init__(
        self,
        env: gym.Env,
        skip: int = 4,
    ):
        """
        Initialize max and skip wrapper.
        
        Args:
            env: Gymnasium environment to wrap
            skip: Number of frames to skip
        """
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.float32)
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute action and return max of last 2 frames.
        
        Args:
            action: Action to execute
            
        Returns:
            Environment step tuple
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for i in range(self._skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            # Store last 2 observations
            if i == self._skip - 2:
                self._obs_buffer[0] = observation
            elif i == self._skip - 1:
                self._obs_buffer[1] = observation
            
            if terminated or truncated:
                break
        
        # Take maximum over last 2 frames
        max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        
        return max_frame, total_reward, terminated, truncated, info


def make_env(
    env_id: str = 'TowerBlox-v0',
    frame_stack: int = 4,
    frame_skip: int = 4,
    **kwargs,
) -> gym.Env:
    """
    Create and wrap Tower Bloxx environment.
    
    Args:
        env_id: Environment ID
        frame_stack: Number of frames to stack
        frame_skip: Number of frames to skip
        **kwargs: Additional environment arguments
        
    Returns:
        Wrapped Gymnasium environment
    """
    # Create base environment
    env = gym.make(env_id, **kwargs)
    
    # Apply frame skip
    if frame_skip > 1:
        env = SkipFrameWrapper(env, skip=frame_skip)
    
    # Apply frame stacking
    if frame_stack > 1:
        env = FrameStackWrapper(env, n_frames=frame_stack)
    
    return env
