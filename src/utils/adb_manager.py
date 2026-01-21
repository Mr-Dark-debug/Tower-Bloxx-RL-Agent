"""
ADB Manager Module
Handles Android Debug Bridge connections and commands.
"""

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


class ADBError(Exception):
    """Custom exception for ADB-related errors."""
    pass


class ADBManager:
    """
    Manages ADB connections and commands for Android device interaction.
    
    Provides methods for:
    - Device connection/disconnection
    - Screen capture
    - Touch input (tap, swipe)
    - App management
    - Device monitoring
    
    Attributes:
        serial: Device serial number
        adb_path: Path to ADB executable
    """
    
    def __init__(
        self,
        serial: Optional[str] = None,
        adb_path: str = "adb",
        timeout: int = 5000,
        max_retries: int = 3,
    ):
        """
        Initialize ADB manager.
        
        Args:
            serial: Device serial number (None = auto-detect first device)
            adb_path: Path to ADB executable
            timeout: Command timeout in milliseconds
            max_retries: Maximum retry attempts for failed commands
        """
        self.adb_path = adb_path
        self.timeout = timeout / 1000  # Convert to seconds
        self.max_retries = max_retries
        self.serial = serial
        
        self._screen_size: Optional[Tuple[int, int]] = None
        self._device_connected = False
        
        # Verify ADB is available
        self._verify_adb()
        
        # Auto-detect device if serial not provided
        if self.serial is None:
            self._auto_detect_device()
    
    def _verify_adb(self) -> None:
        """Verify ADB is installed and accessible."""
        try:
            result = subprocess.run(
                [self.adb_path, "version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise ADBError(f"ADB not working properly: {result.stderr}")
            logger.debug(f"ADB version: {result.stdout.strip().split(chr(10))[0]}")
        except FileNotFoundError:
            raise ADBError(f"ADB not found at: {self.adb_path}")
        except subprocess.TimeoutExpired:
            raise ADBError("ADB verification timed out")
    
    def _auto_detect_device(self) -> None:
        """Auto-detect connected Android device."""
        devices = self.list_devices()
        if not devices:
            raise ADBError("No Android devices connected")
        
        self.serial = devices[0]['serial']
        logger.info(f"Auto-detected device: {self.serial}")
    
    def _run_command(
        self,
        args: List[str],
        timeout: Optional[float] = None,
        raw_output: bool = False,
    ) -> Union[str, bytes]:
        """
        Run an ADB command.
        
        Args:
            args: Command arguments (without 'adb' prefix)
            timeout: Command timeout in seconds
            raw_output: Return raw bytes instead of decoded string
            
        Returns:
            Command output as string or bytes
            
        Raises:
            ADBError: If command fails after retries
        """
        if timeout is None:
            timeout = self.timeout
        
        cmd = [self.adb_path]
        
        # Add device serial if specified
        if self.serial:
            cmd.extend(["-s", self.serial])
        
        cmd.extend(args)
        
        for attempt in range(self.max_retries):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=timeout,
                )
                
                if result.returncode != 0:
                    error = result.stderr.decode('utf-8', errors='ignore')
                    if attempt < self.max_retries - 1:
                        logger.warning(f"ADB command failed (attempt {attempt + 1}): {error}")
                        time.sleep(0.5)
                        continue
                    raise ADBError(f"ADB command failed: {error}")
                
                if raw_output:
                    return result.stdout
                return result.stdout.decode('utf-8', errors='ignore')
                
            except subprocess.TimeoutExpired:
                if attempt < self.max_retries - 1:
                    logger.warning(f"ADB command timed out (attempt {attempt + 1})")
                    time.sleep(0.5)
                    continue
                raise ADBError("ADB command timed out")
        
        raise ADBError("ADB command failed after max retries")
    
    def list_devices(self) -> List[Dict[str, str]]:
        """
        List all connected Android devices.
        
        Returns:
            List of dictionaries with 'serial' and 'state' keys
        """
        output = self._run_command(["devices"])
        devices = []
        
        for line in output.strip().split('\n')[1:]:
            if '\t' in line:
                parts = line.split('\t')
                devices.append({
                    'serial': parts[0],
                    'state': parts[1] if len(parts) > 1 else 'unknown',
                })
        
        return devices
    
    def is_connected(self) -> bool:
        """Check if the target device is connected."""
        devices = self.list_devices()
        for device in devices:
            if device['serial'] == self.serial and device['state'] == 'device':
                return True
        return False
    
    def get_screen_size(self) -> Tuple[int, int]:
        """
        Get device screen size.
        
        Returns:
            Tuple of (width, height)
        """
        if self._screen_size is not None:
            return self._screen_size
        
        output = self._run_command(["shell", "wm", "size"])
        # Output format: "Physical size: 1080x2340"
        for line in output.strip().split('\n'):
            if 'Physical size:' in line:
                size_str = line.split(':')[1].strip()
                width, height = map(int, size_str.split('x'))
                self._screen_size = (width, height)
                return self._screen_size
        
        raise ADBError("Could not determine screen size")
    
    def capture_screen(self) -> np.ndarray:
        """
        Capture device screen as numpy array.
        
        Uses exec-out for faster raw data transfer.
        
        Returns:
            RGB image as numpy array with shape (height, width, 3)
        """
        # Get screen size
        width, height = self.get_screen_size()
        
        # Capture raw screen data
        raw_data = self._run_command(
            ["exec-out", "screencap", "-p"],
            raw_output=True,
            timeout=10,
        )
        
        # Convert PNG bytes to numpy array
        import cv2
        nparr = np.frombuffer(raw_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ADBError("Failed to decode screen capture")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def capture_screen_raw(self) -> bytes:
        """
        Capture screen and return raw PNG bytes.
        
        Returns:
            PNG image as bytes
        """
        return self._run_command(
            ["exec-out", "screencap", "-p"],
            raw_output=True,
            timeout=10,
        )
    
    def tap(self, x: int, y: int) -> None:
        """
        Perform a tap at specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        self._run_command(["shell", "input", "tap", str(x), str(y)])
        logger.debug(f"Tap at ({x}, {y})")
    
    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> None:
        """
        Perform a swipe gesture.
        
        Args:
            x1: Start X coordinate
            y1: Start Y coordinate
            x2: End X coordinate
            y2: End Y coordinate
            duration_ms: Swipe duration in milliseconds
        """
        self._run_command([
            "shell", "input", "swipe",
            str(x1), str(y1), str(x2), str(y2), str(duration_ms)
        ])
        logger.debug(f"Swipe from ({x1}, {y1}) to ({x2}, {y2})")
    
    def long_press(self, x: int, y: int, duration_ms: int = 500) -> None:
        """
        Perform a long press at specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            duration_ms: Press duration in milliseconds
        """
        # Long press is implemented as a swipe to same position
        self._run_command([
            "shell", "input", "swipe",
            str(x), str(y), str(x), str(y), str(duration_ms)
        ])
        logger.debug(f"Long press at ({x}, {y}) for {duration_ms}ms")
    
    def press_key(self, keycode: Union[int, str]) -> None:
        """
        Press a key.
        
        Args:
            keycode: Android keycode (number or name like 'KEYCODE_BACK')
        """
        self._run_command(["shell", "input", "keyevent", str(keycode)])
        logger.debug(f"Key press: {keycode}")
    
    def press_back(self) -> None:
        """Press the back button."""
        self.press_key(4)  # KEYCODE_BACK
    
    def press_home(self) -> None:
        """Press the home button."""
        self.press_key(3)  # KEYCODE_HOME
    
    def launch_app(self, package_name: str, activity: Optional[str] = None) -> None:
        """
        Launch an application.
        
        Args:
            package_name: App package name
            activity: Optional main activity name
        """
        if activity:
            component = f"{package_name}/{activity}"
            self._run_command(["shell", "am", "start", "-n", component])
        else:
            self._run_command([
                "shell", "monkey", "-p", package_name,
                "-c", "android.intent.category.LAUNCHER", "1"
            ])
        logger.info(f"Launched app: {package_name}")
    
    def stop_app(self, package_name: str) -> None:
        """
        Force stop an application.
        
        Args:
            package_name: App package name
        """
        self._run_command(["shell", "am", "force-stop", package_name])
        logger.info(f"Stopped app: {package_name}")
    
    def is_app_running(self, package_name: str) -> bool:
        """
        Check if an app is currently running.
        
        Args:
            package_name: App package name
            
        Returns:
            True if app is running
        """
        output = self._run_command(["shell", "pidof", package_name])
        return len(output.strip()) > 0
    
    def get_current_activity(self) -> str:
        """
        Get the current foreground activity.
        
        Returns:
            Current activity string
        """
        output = self._run_command([
            "shell", "dumpsys", "activity", "activities"
        ])
        
        for line in output.split('\n'):
            if 'mResumedActivity' in line or 'mCurrentFocus' in line:
                return line.strip()
        
        return "unknown"
    
    def get_battery_level(self) -> int:
        """
        Get device battery level.
        
        Returns:
            Battery percentage (0-100)
        """
        output = self._run_command(["shell", "dumpsys", "battery"])
        
        for line in output.split('\n'):
            if 'level:' in line:
                return int(line.split(':')[1].strip())
        
        return -1
    
    def get_device_temperature(self) -> float:
        """
        Get device temperature.
        
        Returns:
            Temperature in Celsius
        """
        output = self._run_command(["shell", "dumpsys", "battery"])
        
        for line in output.split('\n'):
            if 'temperature:' in line:
                # Temperature is in tenths of degree Celsius
                return int(line.split(':')[1].strip()) / 10.0
        
        return -1.0
    
    def wait_for_device(self, timeout: int = 30) -> bool:
        """
        Wait for device to be connected.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if device connected within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_connected():
                return True
            time.sleep(1)
        
        return False
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the device.
        
        Returns:
            True if reconnection successful
        """
        logger.info("Attempting to reconnect to device...")
        
        # Kill and restart ADB server
        try:
            subprocess.run([self.adb_path, "kill-server"], timeout=5)
            time.sleep(1)
            subprocess.run([self.adb_path, "start-server"], timeout=10)
            time.sleep(2)
        except Exception as e:
            logger.error(f"Failed to restart ADB server: {e}")
            return False
        
        # Wait for device
        return self.wait_for_device(timeout=10)
