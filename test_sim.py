
from src.simulation.towerblox_sim import TowerBloxSimEnv
import time
import random

def test_sim():
    print("Initializing TowerBlox Simulator...")
    env = TowerBloxSimEnv(render_mode='human')
    obs, _ = env.reset()
    print("Simulator initialized. Running random agent...")

    for i in range(1000):
        # Random action: 0 = wait, 1 = drop
        # Drop infrequently (approx once every 60 frames / 1 sec)
        action = 1 if random.random() < 0.015 else 0
        
        obs, reward, done, _, info = env.step(action)
        
        if action == 1:
            print(f"Drop! Reward: {reward}")

        if done:
            print("Game Over!")
            env.reset()
            
        time.sleep(0.016) # ~60 FPS

    print("Test complete.")
    env.close()

if __name__ == "__main__":
    test_sim()
