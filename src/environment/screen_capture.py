"""
Screen Capture Module
Fast screen capture from Android device using ADB.
"""

import time
from collections import deque
from typing import Optional, Tuple
import threading

import cv2
import numpy as np

from ..utils.adb_manager import ADBManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ScreenCapture:
    """
    Fast screen capture from Android device.
    
    Provides methods for:
    - Single frame capture
    - Continuous frame streaming
    - Frame preprocessing (resize, grayscale)
    
    Uses ADB exec-out for optimal performance.
    
    Attributes:
        adb: ADB manager instance
        target_size: Target frame dimensions
    """
    
    def __init__(
        self,
        adb: ADBManager,
        target_size: Tuple[int, int] = (84, 84),
        grayscale: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize screen capture.
        
        Args:
            adb: ADB manager instance
            target_size: Target (width, height) for captured frames
            grayscale: Convert to grayscale
            normalize: Normalize pixel values to [0, 1]
        """
        self.adb = adb
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
        
        # Capture statistics
        self._frame_count = 0
        self._total_capture_time = 0.0
        self._last_capture_time = 0.0
        
        # Frame buffer for streaming
        self._frame_buffer: deque = deque(maxlen=5)
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        
        # Screen dimensions
        self._screen_size = adb.get_screen_size()
        logger.info(f"Screen capture initialized: {self._screen_size} -> {target_size}")
    
    def capture_raw(self) -> np.ndarray:
        """
        Capture raw frame from device (no preprocessing).
        
        Returns:
            RGB image as numpy array (H, W, 3)
        """
        start_time = time.time()
        
        frame = self.adb.capture_screen()
        
        capture_time = time.time() - start_time
        self._last_capture_time = capture_time
        self._total_capture_time += capture_time
        self._frame_count += 1
        
        return frame
    
    def capture(self) -> np.ndarray:
        """
        Capture and preprocess frame from device.
        
        Returns:
            Preprocessed frame as numpy array
            - If grayscale: (H, W) or (H, W, 1)
            - If color: (H, W, 3)
            - Normalized to [0, 1] if normalize=True
        """
        # Capture raw frame
        frame = self.capture_raw()
        
        # Preprocess
        frame = self.preprocess(frame)
        
        return frame
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame.
        
        Args:
            frame: Raw RGB frame (H, W, 3)
            
        Returns:
            Preprocessed frame
        """
        # Resize to target size
        if self.target_size is not None:
            frame = cv2.resize(
                frame,
                self.target_size,
                interpolation=cv2.INTER_AREA
            )
        
        # Convert to grayscale
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Normalize to [0, 1]
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Capture and crop a specific region of the screen.
        
        Args:
            x: Left edge
            y: Top edge
            width: Region width
            height: Region height
            
        Returns:
            Cropped frame
        """
        frame = self.capture_raw()
        
        # Crop region
        cropped = frame[y:y+height, x:x+width]
        
        # Preprocess
        return self.preprocess(cropped)
    
    def get_fps(self) -> float:
        """
        Get average capture FPS.
        
        Returns:
            Average frames per second
        """
        if self._frame_count == 0 or self._total_capture_time == 0:
            return 0.0
        
        return self._frame_count / self._total_capture_time
    
    def get_last_capture_time(self) -> float:
        """
        Get time taken for last capture.
        
        Returns:
            Last capture time in seconds
        """
        return self._last_capture_time
    
    def reset_stats(self) -> None:
        """Reset capture statistics."""
        self._frame_count = 0
        self._total_capture_time = 0.0
    
    def start_streaming(self, fps: int = 30) -> None:
        """
        Start continuous frame capture in background thread.
        
        Args:
            fps: Target frames per second
        """
        if self._streaming:
            return
        
        self._streaming = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(fps,),
            daemon=True,
        )
        self._stream_thread.start()
        logger.info(f"Frame streaming started at {fps} FPS target")
    
    def stop_streaming(self) -> None:
        """Stop frame streaming."""
        self._streaming = False
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None
        logger.info("Frame streaming stopped")
    
    def _stream_loop(self, fps: int) -> None:
        """Background streaming loop."""
        frame_time = 1.0 / fps
        
        while self._streaming:
            start = time.time()
            
            try:
                frame = self.capture()
                self._frame_buffer.append(frame)
            except Exception as e:
                logger.warning(f"Frame capture error: {e}")
            
            # Maintain target FPS
            elapsed = time.time() - start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest streamed frame.
        
        Returns:
            Latest frame or None if buffer is empty
        """
        if self._frame_buffer:
            return self._frame_buffer[-1]
        return None
    
    def detect_game_state(self, frame: np.ndarray) -> str:
        """
        Detect current game state from frame.
        
        Args:
            frame: Preprocessed frame
            
        Returns:
            Game state: 'menu', 'playing', 'game_over', 'unknown'
        """
        # Convert to raw for color detection if grayscale
        if len(frame.shape) == 2:
            # Need raw frame for color detection
            raw_frame = self.capture_raw()
        else:
            raw_frame = frame
            if self.normalize:
                raw_frame = (raw_frame * 255).astype(np.uint8)
        
        # Sample regions for detection
        h, w = raw_frame.shape[:2]
        
        # Check for "GAME OVER" screen
        # The Game Over screen has:
        # - Yellow "GAME OVER" text at top of blue panel
        # - Bright blue panel in center (similar to menu blue but with darker edges)
        # Yellow text color around RGB(255, 200, 50)
        # Blue panel around RGB(80, 160, 220)
        
        # Check upper-middle region for yellow "GAME OVER" text
        upper_region = raw_frame[h//4:h//3, w//4:3*w//4]
        yellow_mask = np.all([
            upper_region[:,:,0] > 200,  # High red
            upper_region[:,:,1] > 150,  # High green  
            upper_region[:,:,2] < 100,  # Low blue
        ], axis=0)
        yellow_ratio = np.mean(yellow_mask)
        
        # Check for blue panel
        center_region = raw_frame[h//3:2*h//3, w//4:3*w//4]
        blue_panel_mask = np.all([
            center_region[:,:,0] > 60,
            center_region[:,:,0] < 150,
            center_region[:,:,1] > 130,
            center_region[:,:,1] < 200,
            center_region[:,:,2] > 180,
            center_region[:,:,2] < 255,
        ], axis=0)
        blue_panel_ratio = np.mean(blue_panel_mask)
        
        # Game Over: yellow text visible AND blue panel visible
        if yellow_ratio > 0.01 and blue_panel_ratio > 0.2:
            return 'game_over'
        
        # Check for main menu (light blue background covers most of screen)
        # Menu has light blue color around RGB(148, 187, 211)
        light_blue_mask = np.all([
            raw_frame[:,:,0] > 100,
            raw_frame[:,:,0] < 180,
            raw_frame[:,:,1] > 150,
            raw_frame[:,:,1] < 220,
            raw_frame[:,:,2] > 180,
            raw_frame[:,:,2] < 240,
        ], axis=0)
        light_blue_ratio = np.mean(light_blue_mask)
        
        if light_blue_ratio > 0.4:
            return 'menu'
        
        # Check for active gameplay (brown/tan ground at bottom, buildings)
        # Ground is around RGB(160, 100, 50) brown color
        bottom_region = raw_frame[3*h//4:, :]
        ground_mask = np.all([
            bottom_region[:,:,0] > 100,
            bottom_region[:,:,0] < 200,
            bottom_region[:,:,1] > 60,
            bottom_region[:,:,1] < 140,
            bottom_region[:,:,2] > 20,
            bottom_region[:,:,2] < 100,
        ], axis=0)
        ground_ratio = np.mean(ground_mask)
        
        # Also check for crane/orange blocks visible
        orange_mask = np.all([
            raw_frame[:,:,0] > 200,
            raw_frame[:,:,1] > 100,
            raw_frame[:,:,1] < 200,
            raw_frame[:,:,2] < 80,
        ], axis=0)
        orange_ratio = np.mean(orange_mask)
        
        if ground_ratio > 0.1 or orange_ratio > 0.005:
            return 'playing'
        
        return 'unknown'


class FrameStacker:
    """
    Stacks consecutive frames for temporal information.
    
    Stacking frames allows the network to perceive motion
    and velocity from a sequence of observations.
    """
    
    def __init__(self, n_frames: int = 4):
        """
        Initialize frame stacker.
        
        Args:
            n_frames: Number of frames to stack
        """
        self.n_frames = n_frames
        self._frames: deque = deque(maxlen=n_frames)
    
    def reset(self, initial_frame: Optional[np.ndarray] = None) -> None:
        """
        Reset the frame stack.
        
        Args:
            initial_frame: Optional frame to fill stack with
        """
        self._frames.clear()
        
        if initial_frame is not None:
            # Fill stack with initial frame
            for _ in range(self.n_frames):
                self._frames.append(initial_frame.copy())
    
    def add(self, frame: np.ndarray) -> np.ndarray:
        """
        Add a frame and return stacked observation.
        
        Args:
            frame: New frame to add
            
        Returns:
            Stacked frames with shape (H, W, n_frames) or (n_frames, H, W)
        """
        self._frames.append(frame)
        
        # If buffer not full, pad with copies of first frame
        while len(self._frames) < self.n_frames:
            self._frames.appendleft(frame.copy())
        
        # Stack frames along new axis
        stacked = np.stack(list(self._frames), axis=-1)
        
        return stacked
    
    def get_stacked(self) -> np.ndarray:
        """
        Get current stacked frames.
        
        Returns:
            Stacked frames
        """
        if len(self._frames) == 0:
            raise ValueError("No frames in stack. Call add() first.")
        
        return np.stack(list(self._frames), axis=-1)
