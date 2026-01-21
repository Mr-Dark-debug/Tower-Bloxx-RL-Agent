"""
Frame Processor Module
Preprocessing pipeline for game frames.
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np


class FrameProcessor:
    """
    Processes raw game frames for RL consumption.
    
    Pipeline:
    1. Crop to game area (remove UI)
    2. Resize to target dimensions
    3. Convert to grayscale (optional)
    4. Normalize pixel values
    
    Attributes:
        target_size: Target (width, height) for output
        grayscale: Whether to convert to grayscale
        normalize: Whether to normalize to [0, 1]
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (84, 84),
        grayscale: bool = True,
        normalize: bool = True,
        crop_region: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        Initialize frame processor.
        
        Args:
            target_size: Target (width, height)
            grayscale: Convert to grayscale
            normalize: Normalize to [0, 1]
            crop_region: Optional (x, y, width, height) to crop
        """
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
        self.crop_region = crop_region
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.
        
        Args:
            frame: Input frame (H, W, C) or (H, W)
            
        Returns:
            Processed frame
        """
        # Crop if region specified
        if self.crop_region is not None:
            x, y, w, h = self.crop_region
            frame = frame[y:y+h, x:x+w]
        
        # Resize
        frame = cv2.resize(
            frame,
            self.target_size,
            interpolation=cv2.INTER_AREA
        )
        
        # Convert to grayscale
        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Normalize
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def batch_process(self, frames: np.ndarray) -> np.ndarray:
        """
        Process a batch of frames.
        
        Args:
            frames: Batch of frames (N, H, W, C)
            
        Returns:
            Processed frames
        """
        processed = []
        for frame in frames:
            processed.append(self.process(frame))
        return np.stack(processed)
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """
        Get output shape after processing.
        
        Returns:
            Output shape tuple
        """
        if self.grayscale:
            return (self.target_size[1], self.target_size[0])
        else:
            return (self.target_size[1], self.target_size[0], 3)


class LazyFrameProcessor:
    """
    Lazy frame processor that defers processing until needed.
    
    Useful for memory-efficient storage in replay buffers.
    """
    
    def __init__(
        self,
        frame: np.ndarray,
        processor: FrameProcessor,
    ):
        """
        Initialize lazy frame.
        
        Args:
            frame: Raw frame
            processor: Frame processor to use
        """
        self._frame = frame
        self._processor = processor
        self._processed: Optional[np.ndarray] = None
    
    def get(self) -> np.ndarray:
        """Get processed frame (processes on first call)."""
        if self._processed is None:
            self._processed = self._processor.process(self._frame)
        return self._processed
    
    def __array__(self) -> np.ndarray:
        """Allow numpy conversion."""
        return self.get()


def preprocess_atari_style(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (84, 84),
) -> np.ndarray:
    """
    Standard Atari-style preprocessing.
    
    Args:
        frame: Input frame
        target_size: Target dimensions
        
    Returns:
        Preprocessed frame (grayscale, resized, normalized)
    """
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    # Resize
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized
