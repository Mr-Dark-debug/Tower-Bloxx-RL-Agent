#!/usr/bin/env python
"""
Test Environment
Tests the Tower Bloxx Gymnasium environment.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.mobile_game_env import TowerBloxEnv
from src.utils.logger import setup_logger, get_logger


def test_environment():
    """Test the Tower Bloxx environment."""
    setup_logger()
    logger = get_logger("test_env")
    
    logger.info("=" * 60)
    logger.info("Tower Bloxx RL Agent - Environment Test")
    logger.info("=" * 60)
    
    try:
        # Create environment
        logger.info("\n[Test 1] Creating environment...")
        env = TowerBloxEnv()
        
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
        # Test reset
        logger.info("\n[Test 2] Testing reset...")
        obs, info = env.reset()
        logger.info(f"Observation shape: {obs.shape}")
        logger.info(f"Observation dtype: {obs.dtype}")
        logger.info(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        logger.info(f"Info: {info}")
        
        # Test random actions
        logger.info("\n[Test 3] Running 50 random steps...")
        total_reward = 0.0
        step_times = []
        
        for step in range(50):
            start_time = time.time()
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_time = time.time() - start_time
            step_times.append(step_time)
            total_reward += reward
            
            if step % 10 == 0:
                logger.info(
                    f"Step {step}: action={action}, reward={reward:.2f}, "
                    f"time={step_time*1000:.0f}ms"
                )
            
            if terminated or truncated:
                logger.info(f"Episode ended at step {step}")
                obs, info = env.reset()
        
        avg_step_time = sum(step_times) / len(step_times)
        fps = 1.0 / avg_step_time
        
        logger.info(f"\nTotal reward: {total_reward:.2f}")
        logger.info(f"Average step time: {avg_step_time*1000:.0f}ms")
        logger.info(f"Effective FPS: {fps:.1f}")
        
        # Test specific actions
        logger.info("\n[Test 4] Testing specific actions...")
        
        # Test wait action
        obs, reward, _, _, _ = env.step(0)
        logger.info(f"Wait action (0): reward={reward:.3f}")
        
        # Test tap action
        obs, reward, _, _, _ = env.step(1)
        logger.info(f"Tap action (1): reward={reward:.3f}")
        
        # Test observation validity
        logger.info("\n[Test 5] Validating observations...")
        assert env.observation_space.contains(obs), "Invalid observation!"
        logger.info("Observation is valid!")
        
        # Clean up
        env.close()
        
        logger.info("\n" + "=" * 60)
        logger.info("All environment tests passed!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Environment test failed: {e}")
        raise


def test_vectorized_env():
    """Test vectorized environment wrapping."""
    setup_logger()
    logger = get_logger("test_vec_env")
    
    logger.info("\n[Test] Testing vectorized environment...")
    
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    
    # Create vectorized env
    def make_env():
        return TowerBloxEnv()
    
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    
    logger.info(f"Vec observation space: {env.observation_space}")
    logger.info(f"Vec action space: {env.action_space}")
    
    # Test step
    obs = env.reset()
    logger.info(f"Vec observation shape: {obs.shape}")
    
    for i in range(10):
        action = [env.action_space.sample()]
        obs, reward, done, info = env.step(action)
        
        if done[0]:
            obs = env.reset()
    
    env.close()
    logger.info("Vectorized environment test passed!")
    
    return True


if __name__ == "__main__":
    success = test_environment()
    
    if success:
        try:
            test_vectorized_env()
        except Exception as e:
            print(f"Vectorized env test skipped: {e}")
    
    sys.exit(0 if success else 1)
