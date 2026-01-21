"""
Tower Bloxx Mobile Game Environment
Main Gymnasium environment for RL training.
"""

import time
from typing import Any, Dict, Optional, Tuple, SupportsFloat

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..utils.adb_manager import ADBManager
from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger
from .screen_capture import ScreenCapture, FrameStacker
from .action_executor import ActionExecutor
from .reward_shaper import RewardShaper
from .agent_visualizer import AgentVisualizer

logger = get_logger(__name__)


class TowerBloxEnv(gym.Env):
    """
    Gymnasium environment for Tower Bloxx mobile game.
    
    Connects to an Android device via ADB, captures screen frames,
    executes touch actions, and calculates rewards for RL training.
    
    Observation Space:
        Box(0, 1, shape=(84, 84, 4), dtype=float32)
        - 4 stacked grayscale frames
        - Normalized to [0, 1]
    
    Action Space:
        Discrete(2)
        - 0: Wait (do nothing)
        - 1: Tap (release block)
    
    Attributes:
        render_mode: Rendering mode ('human' or 'rgb_array')
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30,
    }
    
    def __init__(
        self,
        config_dir: Optional[str] = None,
        render_mode: Optional[str] = None,
        device_serial: Optional[str] = None,
        frame_stack: int = 4,
        frame_skip: int = 4,
        max_episode_steps: int = 500,
        show_visualization: bool = True,  # Enable by default for debugging
    ):
        """
        Initialize Tower Bloxx environment.
        
        Args:
            config_dir: Path to configuration directory
            render_mode: Rendering mode ('human' or 'rgb_array')
            device_serial: Android device serial number
            frame_stack: Number of frames to stack
            frame_skip: Number of frames to skip per action
            max_episode_steps: Maximum steps per episode
            show_visualization: Show real-time visualization window
        """
        super().__init__()
        
        # Load configuration
        self.config_loader = ConfigLoader(config_dir)
        self.env_config = self.config_loader.get_env_config()
        self.device_config = self.config_loader.get_device_config()
        
        # Override with parameters
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.show_visualization = show_visualization
        
        # Frame dimensions
        obs_config = self.env_config.get('observation', {})
        self.frame_width = obs_config.get('frame_width', 84)
        self.frame_height = obs_config.get('frame_height', 84)
        
        # Initialize ADB manager
        serial = device_serial or self.device_config.get('device', {}).get('serial')
        self.adb = ADBManager(serial=serial)
        
        # Get screen dimensions
        self.screen_width, self.screen_height = self.adb.get_screen_size()
        
        # Initialize components
        self.screen_capture = ScreenCapture(
            adb=self.adb,
            target_size=(self.frame_width, self.frame_height),
            grayscale=True,
            normalize=True,
        )
        
        self.action_executor = ActionExecutor(
            adb=self.adb,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
        )
        
        reward_config = self.env_config.get('rewards', {})
        self.reward_shaper = RewardShaper(config=reward_config)
        
        self.frame_stacker = FrameStacker(n_frames=frame_stack)
        
        # Initialize visualizer if enabled
        self.visualizer: Optional[AgentVisualizer] = None
        if show_visualization:
            self.visualizer = AgentVisualizer(
                window_name="Tower Bloxx RL Agent",
                display_width=400,
                show_preprocessed=True,
            )
        
        # Store raw frame for visualization
        self._raw_frame: Optional[np.ndarray] = None
        
        # Define spaces
        # Observation: stacked grayscale frames
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.frame_height, self.frame_width, frame_stack),
            dtype=np.float32,
        )
        
        # Action: discrete (wait or tap)
        self.action_space = spaces.Discrete(2)
        
        # Episode state
        self._current_step = 0
        self._episode_reward = 0.0
        self._previous_frame: Optional[np.ndarray] = None
        self._current_frame: Optional[np.ndarray] = None
        self._game_state: Dict[str, Any] = {}
        self._episode_count = 0
        
        # Timing
        self._step_start_time = 0.0
        self._episode_start_time = 0.0
        
        logger.info(
            f"TowerBloxEnv initialized: "
            f"observation_space={self.observation_space.shape}, "
            f"action_space={self.action_space.n}, "
            f"visualization={'enabled' if show_visualization else 'disabled'}"
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        self._episode_count += 1
        logger.info(f"Starting episode {self._episode_count}")
        
        # Reset state
        self._current_step = 0
        self._episode_reward = 0.0
        self._episode_start_time = time.time()
        
        # Reset reward shaper
        self.reward_shaper.reset()
        
        # Navigate to game
        self._ensure_game_running()
        
        # Capture initial frame
        initial_frame = self.screen_capture.capture()
        self._current_frame = initial_frame
        self._previous_frame = initial_frame.copy()
        
        # Initialize frame stacker
        self.frame_stacker.reset(initial_frame)
        
        # Get stacked observation
        observation = self.frame_stacker.get_stacked()
        
        # Build info dictionary
        info = {
            'episode': self._episode_count,
            'step': 0,
            'game_state': 'playing',
        }
        
        return observation.astype(np.float32), info
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action to execute (0=wait, 1=tap)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._step_start_time = time.time()
        self._current_step += 1
        
        # Execute action with frame skip
        reward_accumulator = 0.0
        
        for _ in range(self.frame_skip):
            # Execute action
            self.action_executor.execute(action)
            
            # Small delay to let game respond
            time.sleep(0.03)  # ~30 FPS timing
            
            # Capture new frame
            self._previous_frame = self._current_frame
            self._current_frame = self.screen_capture.capture()
            
            # Update game state
            self._update_game_state()
            
            # Calculate reward
            reward, components = self.reward_shaper.calculate_reward(
                current_state=self._game_state,
                previous_state=None,
                action=action,
                done=self._game_state.get('game_over', False),
            )
            reward_accumulator += reward
            
            # Check for early termination
            if self._game_state.get('game_over', False):
                break
        
        # Add frame to stack
        observation = self.frame_stacker.add(self._current_frame)
        
        # Update episode reward
        self._episode_reward += reward_accumulator
        
        # Check termination conditions
        terminated = self._game_state.get('game_over', False)
        truncated = self._current_step >= self.max_episode_steps
        
        # Build info dictionary
        step_time = time.time() - self._step_start_time
        info = {
            'episode': self._episode_count,
            'step': self._current_step,
            'action': action,
            'action_name': self.action_executor.get_action_name(action),
            'reward': reward_accumulator,
            'episode_reward': self._episode_reward,
            'game_state': 'game_over' if terminated else 'playing',
            'step_time': step_time,
            'fps': 1.0 / step_time if step_time > 0 else 0,
            'tower_height': self._game_state.get('tower_height', 0),
        }
        
        if terminated or truncated:
            episode_time = time.time() - self._episode_start_time
            logger.info(
                f"Episode {self._episode_count} ended: "
                f"reward={self._episode_reward:.2f}, "
                f"steps={self._current_step}, "
                f"time={episode_time:.1f}s"
            )
        
        # Update visualization if enabled
        if self.visualizer is not None:
            # Capture raw frame for visualization
            self._raw_frame = self.screen_capture.capture_raw()
            
            self.visualizer.update(
                raw_frame=self._raw_frame,
                observation=observation,
                tap_position=self.action_executor.last_tap_position if action == 1 else None,
                action=action,
                reward=reward_accumulator,
                episode_reward=self._episode_reward,
                step=self._current_step,
                game_state='game_over' if terminated else 'playing',
            )
        
        return (
            observation.astype(np.float32),
            float(reward_accumulator),
            terminated,
            truncated,
            info,
        )
    
    def _update_game_state(self) -> None:
        """Update internal game state from current frame."""
        # Detect game over
        game_over = self.reward_shaper.detect_game_over(
            self._current_frame,
            self._previous_frame,
        )
        
        # Estimate tower height
        tower_height = self.reward_shaper.estimate_tower_height(self._current_frame)
        
        # Estimate placement quality if action was tap
        placement_quality = self.reward_shaper.calculate_placement_quality(
            self._current_frame,
            self._previous_frame,
        )
        
        self._game_state = {
            'game_over': game_over,
            'tower_height': tower_height,
            'placement_quality': placement_quality,
            'coins': 0,  # TODO: Implement coin detection via OCR
        }
    
    def _ensure_game_running(self) -> None:
        """Ensure the game is running and ready to play."""
        # Capture current screen
        frame = self.screen_capture.capture_raw()
        
        # Detect current state
        state = self.screen_capture.detect_game_state(frame)
        
        if state == 'menu':
            logger.info("Detected menu, starting quick game...")
            self.action_executor.start_quick_game()
            time.sleep(2.0)  # Wait for game to load
            
        elif state == 'game_over':
            logger.info("Detected game over, restarting...")
            self.action_executor.restart_game()
            time.sleep(1.5)
            
        elif state == 'playing':
            logger.debug("Game is already running")
            
        else:
            logger.warning(f"Unknown game state: {state}, attempting to start game")
            # Try to start quick game anyway
            self.action_executor.start_quick_game()
            time.sleep(2.0)
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB frame if render_mode='rgb_array', None otherwise
        """
        if self.render_mode == 'rgb_array':
            # Return current raw frame
            return self.screen_capture.capture_raw()
        
        elif self.render_mode == 'human':
            # Display frame in window
            import cv2
            frame = self.screen_capture.capture_raw()
            
            # Resize for display
            display_height = 800
            aspect = frame.shape[1] / frame.shape[0]
            display_width = int(display_height * aspect)
            
            frame_display = cv2.resize(frame, (display_width, display_height))
            frame_display = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
            
            # Add info overlay
            cv2.putText(
                frame_display,
                f"Step: {self._current_step} | Reward: {self._episode_reward:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            
            cv2.imshow('Tower Bloxx RL', frame_display)
            cv2.waitKey(1)
        
        return None
    
    def close(self) -> None:
        """Clean up environment resources."""
        logger.info("Closing TowerBloxEnv")
        
        # Close visualizer
        if self.visualizer is not None:
            self.visualizer.close()
        
        # Close any display windows
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        
        # Stop frame streaming if active
        self.screen_capture.stop_streaming()
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get statistics for current episode.
        
        Returns:
            Dictionary with episode statistics
        """
        return {
            'episode': self._episode_count,
            'steps': self._current_step,
            'reward': self._episode_reward,
            'duration': time.time() - self._episode_start_time,
            'capture_fps': self.screen_capture.get_fps(),
            'action_stats': self.action_executor.get_action_stats(),
            'reward_stats': self.reward_shaper.get_statistics(),
        }


# Register environment with Gymnasium
try:
    gym.register(
        id='TowerBlox-v0',
        entry_point='src.environment.mobile_game_env:TowerBloxEnv',
        max_episode_steps=500,
    )
except Exception:
    pass  # Already registered
