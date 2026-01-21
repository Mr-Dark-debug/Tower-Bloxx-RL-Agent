"""
Action Executor Module
Translates RL actions to game inputs via ADB.
"""

import time
from typing import Dict, List, Optional, Tuple

from ..utils.adb_manager import ADBManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ActionExecutor:
    """
    Executes game actions on Android device.
    
    Maps discrete RL actions to touch inputs:
    - Action 0: Wait (do nothing)
    - Action 1: Tap (release block)
    
    Handles timing and coordinates for Tower Bloxx game.
    
    Attributes:
        adb: ADB manager instance
        screen_width: Device screen width
        screen_height: Device screen height
    """
    
    # Action constants
    ACTION_WAIT = 0
    ACTION_TAP = 1
    
    # Action names for logging
    ACTION_NAMES = {
        0: "WAIT",
        1: "TAP",
    }
    
    def __init__(
        self,
        adb: ADBManager,
        screen_width: int = 1084,
        screen_height: int = 2412,
        tap_delay_ms: int = 50,
        post_action_delay_ms: int = 50,
    ):
        """
        Initialize action executor.
        
        Args:
            adb: ADB manager instance
            screen_width: Device screen width
            screen_height: Device screen height
            tap_delay_ms: Tap duration in milliseconds
            post_action_delay_ms: Delay after action execution
        """
        self.adb = adb
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.tap_delay_ms = tap_delay_ms
        self.post_action_delay_ms = post_action_delay_ms
        
        # Define tap position (center of screen for Tower Bloxx)
        # The game responds to taps anywhere on the screen
        self.tap_position = (screen_width // 2, screen_height // 2)
        
        # Action statistics
        self._action_counts: Dict[int, int] = {0: 0, 1: 0}
        self._total_actions = 0
        
        # Button positions for menu navigation
        self._button_positions = self._define_button_positions()
        
        logger.info(f"Action executor initialized: screen {screen_width}x{screen_height}")
    
    def _define_button_positions(self) -> Dict[str, Tuple[int, int]]:
        """
        Define UI button positions based on screen resolution.
        
        Returns:
            Dictionary mapping button names to (x, y) coordinates
        """
        # Based on 1084x2412 screen resolution from game screenshots
        # Screen center X = 542
        return {
            # Main menu buttons (centered around x=300)
            'quick_game': (250, 950),   # "Quick Game" button on main menu
            'build_city': (250, 850),   # "Build City" button
            'ranking_menu': (250, 1050),  # "Ranking" button on menu
            'shop': (250, 1150),        # "Shop" button
            
            # Game Over screen buttons (based on uploaded screenshot)
            # The blue panel with buttons is roughly centered
            # "Play" button is on the left, "Home" center, "Ranking" right
            'play_button': (120, 1320),     # "Play" button on Game Over screen (left)
            'home_button': (250, 1320),     # "Home" button on Game Over screen (center)
            'ranking_button': (380, 1320),  # "Ranking" button on Game Over screen (right)
            'revive_button': (250, 1220),   # "Revive" button (above Play/Home/Ranking)
            
            # Legacy names for compatibility
            'play_again': (120, 1320),      # Same as play_button
            'retry': (250, 1200),           # Center of game over panel
            
            # General tap position (for block release during gameplay)
            # Tap in upper-middle area where crane is
            'game_tap': (self.screen_width // 2, self.screen_height // 3),
            
            # Skip/close buttons (for ads, popups)
            'close_top_right': (self.screen_width - 50, 100),
            'close_center': (self.screen_width // 2, self.screen_height // 2),
        }
    
    # Track last tap position for visualization
    last_tap_position: Optional[Tuple[int, int]] = None
    
    def execute(self, action: int) -> bool:
        """
        Execute an action.
        
        Args:
            action: Action index (0=wait, 1=tap)
            
        Returns:
            True if action was executed successfully
        """
        self._total_actions += 1
        self._action_counts[action] = self._action_counts.get(action, 0) + 1
        
        if action == self.ACTION_WAIT:
            # Do nothing - just wait
            return True
        
        elif action == self.ACTION_TAP:
            # Tap to release block - use game_tap position
            x, y = self._button_positions['game_tap']
            self.adb.tap(x, y)
            
            # Track for visualization
            self.last_tap_position = (x, y)
            
            # Small delay to let action register
            if self.post_action_delay_ms > 0:
                time.sleep(self.post_action_delay_ms / 1000.0)
            
            return True
        
        else:
            logger.warning(f"Unknown action: {action}")
            return False
    
    def tap_button(self, button_name: str) -> bool:
        """
        Tap a named button.
        
        Args:
            button_name: Name of button from _button_positions
            
        Returns:
            True if tap was executed
        """
        if button_name not in self._button_positions:
            logger.warning(f"Unknown button: {button_name}")
            return False
        
        x, y = self._button_positions[button_name]
        self.adb.tap(x, y)
        
        logger.debug(f"Tapped button '{button_name}' at ({x}, {y})")
        return True
    
    def tap_at(self, x: int, y: int) -> None:
        """
        Tap at specific coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.adb.tap(x, y)
    
    def start_quick_game(self) -> None:
        """Navigate from menu to start a quick game."""
        logger.info("Starting quick game...")
        self.tap_button('quick_game')
        time.sleep(1.5)  # Wait for game to load
    
    def restart_game(self) -> None:
        """
        Restart the game after game over.
        
        On the Game Over screen, clicks the "Play" button to restart.
        """
        logger.info("Restarting game from Game Over screen...")
        
        # Click the "Play" button on Game Over screen (left button)
        self.tap_button('play_button')
        self.last_tap_position = self._button_positions['play_button']
        time.sleep(2.0)  # Wait for game to load
    
    def go_to_menu(self) -> None:
        """Navigate to main menu."""
        logger.info("Going to menu...")
        
        # Press back button
        self.adb.press_back()
        time.sleep(0.5)
        
        # Press back again if needed
        self.adb.press_back()
        time.sleep(0.5)
    
    def close_popup(self) -> None:
        """Close any popup or ad overlay."""
        logger.debug("Attempting to close popup...")
        
        # Try close button positions
        self.tap_button('close_top_right')
        time.sleep(0.2)
        
        self.tap_button('close_center')
        time.sleep(0.2)
        
        # Also try back button
        self.adb.press_back()
    
    def get_action_name(self, action: int) -> str:
        """
        Get human-readable name for action.
        
        Args:
            action: Action index
            
        Returns:
            Action name string
        """
        return self.ACTION_NAMES.get(action, f"UNKNOWN({action})")
    
    def get_action_stats(self) -> Dict[str, int]:
        """
        Get action execution statistics.
        
        Returns:
            Dictionary with action counts and totals
        """
        return {
            'wait_count': self._action_counts.get(0, 0),
            'tap_count': self._action_counts.get(1, 0),
            'total': self._total_actions,
        }
    
    def reset_stats(self) -> None:
        """Reset action statistics."""
        self._action_counts = {0: 0, 1: 0}
        self._total_actions = 0
    
    @property
    def n_actions(self) -> int:
        """Get number of available actions."""
        return 2  # Wait and Tap


class ActionSpace:
    """
    Defines the action space for the Tower Bloxx environment.
    
    This is a simple discrete action space with 2 actions:
    - 0: Wait (do nothing, let crane swing)
    - 1: Tap (release the block)
    """
    
    def __init__(self):
        """Initialize action space."""
        self.n = 2
        self.actions = [0, 1]
        self.action_names = ['wait', 'tap']
    
    def sample(self) -> int:
        """
        Sample a random action.
        
        Returns:
            Random action index
        """
        import random
        return random.choice(self.actions)
    
    def contains(self, action: int) -> bool:
        """
        Check if action is valid.
        
        Args:
            action: Action index
            
        Returns:
            True if action is valid
        """
        return action in self.actions
