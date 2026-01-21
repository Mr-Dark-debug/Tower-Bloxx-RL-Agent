"""
Agent Visualizer Module
Real-time visualization of what the RL agent sees and does.
"""

import threading
import time
from collections import deque
from typing import Optional, Tuple
import cv2
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AgentVisualizer:
    """
    Real-time visualization of agent's perception and actions.
    
    Shows:
    - Raw screen capture from device
    - Preprocessed observation the agent sees
    - Tap/click positions
    - Game state info overlay
    
    Uses OpenCV for display window.
    """
    
    def __init__(
        self,
        window_name: str = "Tower Bloxx RL Agent",
        display_width: int = 400,
        show_preprocessed: bool = True,
    ):
        """
        Initialize visualizer.
        
        Args:
            window_name: Name of the display window
            display_width: Width of display window
            show_preprocessed: Show preprocessed observation alongside raw
        """
        self.window_name = window_name
        self.display_width = display_width
        self.show_preprocessed = show_preprocessed
        
        self._running = False
        self._current_frame: Optional[np.ndarray] = None
        self._current_obs: Optional[np.ndarray] = None
        self._last_tap: Optional[Tuple[int, int]] = None
        self._tap_time: float = 0
        self._info_text: str = ""
        self._episode_reward: float = 0
        self._current_step: int = 0
        self._action: int = 0
        
        # Frame history for animation
        self._tap_history: deque = deque(maxlen=10)
        
        logger.info(f"Visualizer initialized: {window_name}")
    
    def update(
        self,
        raw_frame: np.ndarray,
        observation: Optional[np.ndarray] = None,
        tap_position: Optional[Tuple[int, int]] = None,
        action: int = 0,
        reward: float = 0,
        episode_reward: float = 0,
        step: int = 0,
        game_state: str = "unknown",
    ) -> None:
        """
        Update visualization with new frame and info.
        
        Args:
            raw_frame: Raw RGB frame from device
            observation: Preprocessed observation (grayscale, stacked)
            tap_position: Position of last tap (x, y)
            action: Action taken (0=wait, 1=tap)
            reward: Step reward
            episode_reward: Cumulative episode reward
            step: Current step number
            game_state: Current game state string
        """
        self._current_frame = raw_frame.copy()
        self._current_obs = observation
        self._action = action
        self._episode_reward = episode_reward
        self._current_step = step
        
        if tap_position is not None:
            self._last_tap = tap_position
            self._tap_time = time.time()
            self._tap_history.append((tap_position, time.time()))
        
        self._info_text = (
            f"Step: {step} | Action: {'TAP' if action == 1 else 'WAIT'} | "
            f"Reward: {reward:.2f} | Total: {episode_reward:.2f} | "
            f"State: {game_state}"
        )
        
        # Display the frame
        self._display_frame()
    
    def _display_frame(self) -> None:
        """Display current frame with overlays."""
        if self._current_frame is None:
            return
        
        # Create display frame
        frame = self._current_frame.copy()
        
        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Calculate display size maintaining aspect ratio
        h, w = frame.shape[:2]
        aspect = w / h
        display_h = int(self.display_width / aspect)
        
        # Resize for display
        frame_display = cv2.resize(frame, (self.display_width, display_h))
        
        # Scale factor for tap position
        scale_x = self.display_width / w
        scale_y = display_h / h
        
        # Draw tap history with fading circles
        current_time = time.time()
        for tap_pos, tap_time in list(self._tap_history):
            age = current_time - tap_time
            if age < 2.0:  # Show taps for 2 seconds
                alpha = 1.0 - (age / 2.0)
                radius = int(20 * alpha) + 5
                
                x = int(tap_pos[0] * scale_x)
                y = int(tap_pos[1] * scale_y)
                
                # Draw circles with color based on age
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))  # Green to red fade
                cv2.circle(frame_display, (x, y), radius, color, 2)
                cv2.circle(frame_display, (x, y), 3, (0, 0, 255), -1)  # Red center dot
        
        # Draw current tap with crosshair
        if self._last_tap and (current_time - self._tap_time) < 0.5:
            x = int(self._last_tap[0] * scale_x)
            y = int(self._last_tap[1] * scale_y)
            
            # Crosshair
            cv2.line(frame_display, (x - 20, y), (x + 20, y), (0, 255, 0), 2)
            cv2.line(frame_display, (x, y - 20), (x, y + 20), (0, 255, 0), 2)
            
            # Position text
            cv2.putText(
                frame_display,
                f"TAP ({self._last_tap[0]}, {self._last_tap[1]})",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
        
        # Add info overlay at top
        overlay = frame_display.copy()
        cv2.rectangle(overlay, (0, 0), (self.display_width, 60), (0, 0, 0), -1)
        frame_display = cv2.addWeighted(overlay, 0.7, frame_display, 0.3, 0)
        
        # Draw info text
        cv2.putText(
            frame_display,
            self._info_text[:60],  # First line
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
        
        if len(self._info_text) > 60:
            cv2.putText(
                frame_display,
                self._info_text[60:],
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
        
        # Action indicator
        action_color = (0, 0, 255) if self._action == 1 else (128, 128, 128)
        action_text = "TAP!" if self._action == 1 else "WAIT"
        cv2.putText(
            frame_display,
            action_text,
            (self.display_width - 60, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            action_color,
            2,
        )
        
        # If showing preprocessed observation
        if self.show_preprocessed and self._current_obs is not None:
            obs_display = self._create_obs_display()
            if obs_display is not None:
                # Stack horizontally
                if obs_display.shape[0] != frame_display.shape[0]:
                    obs_display = cv2.resize(
                        obs_display,
                        (obs_display.shape[1], frame_display.shape[0])
                    )
                frame_display = np.hstack([frame_display, obs_display])
        
        # Show window
        cv2.imshow(self.window_name, frame_display)
        cv2.waitKey(1)
    
    def _create_obs_display(self) -> Optional[np.ndarray]:
        """Create display for preprocessed observation."""
        if self._current_obs is None:
            return None
        
        obs = self._current_obs
        
        # Handle stacked frames (84, 84, 4)
        if len(obs.shape) == 3:
            # Show all 4 stacked frames in a 2x2 grid
            n_frames = obs.shape[-1]
            frame_size = obs.shape[0]
            
            grid_size = 2  # 2x2 grid
            grid = np.zeros((frame_size * grid_size, frame_size * grid_size), dtype=np.float32)
            
            for i in range(min(n_frames, 4)):
                row = i // grid_size
                col = i % grid_size
                
                frame = obs[:, :, i]
                
                # Place in grid
                y_start = row * frame_size
                x_start = col * frame_size
                grid[y_start:y_start + frame_size, x_start:x_start + frame_size] = frame
            
            obs_display = grid
        else:
            obs_display = obs
        
        # Normalize to 0-255
        if obs_display.max() <= 1.0:
            obs_display = (obs_display * 255).astype(np.uint8)
        else:
            obs_display = obs_display.astype(np.uint8)
        
        # Convert to BGR for display
        obs_display = cv2.cvtColor(obs_display, cv2.COLOR_GRAY2BGR)
        
        # Add label
        cv2.putText(
            obs_display,
            "Agent View (4 frames)",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
        
        return obs_display
    
    def close(self) -> None:
        """Close visualization window."""
        cv2.destroyWindow(self.window_name)
        cv2.destroyAllWindows()
        logger.info("Visualizer closed")
    
    def is_open(self) -> bool:
        """Check if window is still open."""
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except Exception:
            return False


def create_debug_overlay(
    frame: np.ndarray,
    tap_pos: Optional[Tuple[int, int]] = None,
    info_text: str = "",
) -> np.ndarray:
    """
    Create a debug overlay on a frame.
    
    Args:
        frame: Input frame (RGB or BGR)
        tap_pos: Tap position to highlight
        info_text: Text to overlay
        
    Returns:
        Frame with overlay
    """
    display = frame.copy()
    
    # Convert if needed
    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    
    # Draw tap position
    if tap_pos is not None:
        cv2.circle(display, tap_pos, 20, (0, 255, 0), 2)
        cv2.circle(display, tap_pos, 5, (0, 0, 255), -1)
        cv2.line(display, (tap_pos[0] - 30, tap_pos[1]), (tap_pos[0] + 30, tap_pos[1]), (0, 255, 0), 1)
        cv2.line(display, (tap_pos[0], tap_pos[1] - 30), (tap_pos[0], tap_pos[1] + 30), (0, 255, 0), 1)
    
    # Add text overlay
    if info_text:
        cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return display
