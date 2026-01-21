#!/usr/bin/env python
"""
Test ADB Connection
Verifies Android device connectivity and screen capture.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.adb_manager import ADBManager, ADBError
from src.utils.logger import setup_logger, get_logger


def main():
    """Test ADB connection and screen capture."""
    setup_logger()
    logger = get_logger("test_adb")
    
    logger.info("=" * 60)
    logger.info("Tower Bloxx RL Agent - ADB Connection Test")
    logger.info("=" * 60)
    
    try:
        # Test 1: List devices
        logger.info("\n[Test 1] Listing connected devices...")
        adb = ADBManager()
        devices = adb.list_devices()
        
        if not devices:
            logger.error("No devices found! Make sure USB debugging is enabled.")
            return False
        
        logger.info(f"Found {len(devices)} device(s):")
        for device in devices:
            logger.info(f"  - {device['serial']}: {device['state']}")
        
        # Test 2: Get screen size
        logger.info("\n[Test 2] Getting screen size...")
        width, height = adb.get_screen_size()
        logger.info(f"Screen size: {width}x{height}")
        
        # Test 3: Test screen capture
        logger.info("\n[Test 3] Testing screen capture...")
        start_time = time.time()
        frame = adb.capture_screen()
        capture_time = time.time() - start_time
        
        logger.info(f"Captured frame: {frame.shape}")
        logger.info(f"Capture time: {capture_time*1000:.0f}ms")
        
        # Test 4: Capture speed benchmark
        logger.info("\n[Test 4] Screen capture benchmark (10 frames)...")
        times = []
        for i in range(10):
            start = time.time()
            frame = adb.capture_screen()
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        logger.info(f"Average capture time: {avg_time*1000:.0f}ms")
        logger.info(f"Estimated FPS: {fps:.1f}")
        
        # Test 5: Test tap input
        logger.info("\n[Test 5] Testing tap input...")
        center_x = width // 2
        center_y = height // 2
        
        logger.info(f"Tapping center of screen: ({center_x}, {center_y})")
        adb.tap(center_x, center_y)
        logger.info("Tap executed successfully!")
        
        # Test 6: Get device info
        logger.info("\n[Test 6] Getting device info...")
        battery = adb.get_battery_level()
        temp = adb.get_device_temperature()
        logger.info(f"Battery level: {battery}%")
        logger.info(f"Device temperature: {temp}°C")
        
        # Test 7: Save test screenshot
        logger.info("\n[Test 7] Saving test screenshot...")
        import cv2
        output_path = Path("./logs/test_screenshot.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), frame_bgr)
        logger.info(f"Screenshot saved: {output_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("All tests passed! ADB connection is working.")
        logger.info("=" * 60)
        
        return True
        
    except ADBError as e:
        logger.error(f"ADB Error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
