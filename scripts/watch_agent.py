#!/usr/bin/env python
"""
Watch Agent Play
Run the environment with visualization to see what the agent sees.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.mobile_game_env import TowerBloxEnv
from src.utils.logger import setup_logger, get_logger


def main():
    """Watch the agent play with visualization."""
    setup_logger()
    logger = get_logger("watch_agent")
    
    logger.info("=" * 60)
    logger.info("Tower Bloxx RL Agent - Watch Mode")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop")
    
    try:
        # Create environment with visualization enabled
        env = TowerBloxEnv(show_visualization=True)
        
        logger.info(f"Environment ready. Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
        episode = 0
        
        while True:
            episode += 1
            logger.info(f"\n--- Starting Episode {episode} ---")
            
            obs, info = env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            step = 0
            
            while not (done or truncated):
                # Random action for now
                action = env.action_space.sample()
                
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                
                # Print status every 5 steps
                if step % 5 == 0:
                    action_name = "TAP" if action == 1 else "WAIT"
                    logger.info(
                        f"Step {step}: {action_name} | "
                        f"Reward: {reward:.2f} | Total: {total_reward:.2f}"
                    )
            
            logger.info(
                f"Episode {episode} ended: "
                f"steps={step}, reward={total_reward:.2f}"
            )
            
            # Small delay between episodes
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    
    finally:
        env.close()
        logger.info("Environment closed")


if __name__ == "__main__":
    main()
