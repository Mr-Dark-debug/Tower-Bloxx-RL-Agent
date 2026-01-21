#!/usr/bin/env python
"""
Setup Environment Script
Verifies all dependencies and system requirements.
"""

import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("\n[1/7] Checking Python version...")
    
    major, minor = sys.version_info[:2]
    print(f"  Python {major}.{minor}.{sys.version_info.micro}")
    
    if major < 3 or (major == 3 and minor < 9):
        print("  ❌ Python 3.9+ required!")
        return False
    
    print("  ✓ Python version OK")
    return True


def check_cuda():
    """Check CUDA availability."""
    print("\n[2/7] Checking CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            
            # Check memory
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024 ** 3)
            print(f"  GPU Memory: {total_gb:.1f} GB")
            
            print("  ✓ CUDA OK")
            return True
        else:
            print("  ⚠ CUDA not available (will use CPU)")
            return True
            
    except ImportError:
        print("  ❌ PyTorch not installed!")
        return False


def check_adb():
    """Check ADB installation."""
    print("\n[3/7] Checking ADB...")
    
    try:
        result = subprocess.run(
            ["adb", "version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            print(f"  {version}")
            
            # List devices
            result = subprocess.run(
                ["adb", "devices"],
                capture_output=True,
                text=True,
            )
            
            lines = result.stdout.strip().split('\n')[1:]
            devices = [l for l in lines if l.strip()]
            
            if devices:
                print(f"  Found {len(devices)} device(s)")
                print("  ✓ ADB OK")
                return True
            else:
                print("  ⚠ No devices connected")
                return True
        else:
            print("  ❌ ADB not working properly")
            return False
            
    except FileNotFoundError:
        print("  ❌ ADB not found in PATH!")
        print("  Install Android Platform Tools and add to PATH")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def check_packages():
    """Check required Python packages."""
    print("\n[4/7] Checking Python packages...")
    
    required = [
        ("torch", "PyTorch"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("gymnasium", "Gymnasium"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("PIL", "Pillow"),
    ]
    
    all_ok = True
    
    for package, name in required:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ❌ {name} not installed")
            all_ok = False
    
    # Optional packages
    optional = [
        ("tensorboard", "TensorBoard"),
        ("matplotlib", "Matplotlib"),
        ("pynvml", "pynvml (GPU monitoring)"),
        ("colorama", "Colorama"),
    ]
    
    for package, name in optional:
        try:
            __import__(package)
            print(f"  ✓ {name} (optional)")
        except ImportError:
            print(f"  ⚠ {name} not installed (optional)")
    
    return all_ok


def check_configs():
    """Check configuration files."""
    print("\n[5/7] Checking configuration files...")
    
    config_dir = Path(__file__).parent.parent / "configs"
    
    required_configs = [
        "env_config.yaml",
        "ppo_config.yaml",
        "device_config.yaml",
    ]
    
    all_ok = True
    
    for config in required_configs:
        path = config_dir / config
        if path.exists():
            print(f"  ✓ {config}")
        else:
            print(f"  ❌ {config} missing!")
            all_ok = False
    
    return all_ok


def check_directories():
    """Check and create required directories."""
    print("\n[6/7] Checking directories...")
    
    dirs = [
        "logs",
        "logs/tensorboard",
        "logs/checkpoints",
        "logs/training_logs",
        "logs/visualizations",
    ]
    
    base = Path(__file__).parent.parent
    
    for d in dirs:
        path = base / d
        if not path.exists():
            path.mkdir(parents=True)
            print(f"  Created: {d}")
        else:
            print(f"  ✓ {d}")
    
    return True


def test_quick_import():
    """Quick test of main modules."""
    print("\n[7/7] Testing module imports...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src.utils.config_loader import ConfigLoader
        print("  ✓ ConfigLoader")
        
        from src.utils.adb_manager import ADBManager
        print("  ✓ ADBManager")
        
        from src.utils.logger import setup_logger
        print("  ✓ Logger")
        
        from src.environment.screen_capture import ScreenCapture
        print("  ✓ ScreenCapture")
        
        from src.environment.action_executor import ActionExecutor
        print("  ✓ ActionExecutor")
        
        # Skip full env import as it needs device
        print("  ✓ All core modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False


def main():
    """Run all setup checks."""
    print("=" * 60)
    print("Tower Bloxx RL Agent - Environment Setup Check")
    print("=" * 60)
    
    results = []
    
    results.append(("Python", check_python_version()))
    results.append(("CUDA", check_cuda()))
    results.append(("ADB", check_adb()))
    results.append(("Packages", check_packages()))
    results.append(("Configs", check_configs()))
    results.append(("Directories", check_directories()))
    results.append(("Imports", test_quick_import()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_ok = True
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_ok = False
    
    print()
    if all_ok:
        print("✓ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Connect Android device with USB debugging")
        print("  2. Open Tower Bloxx game")
        print("  3. Run: python scripts/test_adb_connection.py")
        print("  4. Run: python train.py")
    else:
        print("❌ Some checks failed. Please fix issues above.")
    
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
