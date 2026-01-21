
"""
Tower Bloxx Simulator
A pure Python implementation of the Tower Bloxx game mechanics using Pygame.
This allows for fast, deterministic training without ADB/device latency.
"""

import math
import random
import time
from typing import List, Tuple, Optional, Dict

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TowerBloxSimulator:
    """
    Simulates the physics and mechanics of Tower Bloxx.
    Enforces strict turn-based logic: Spawn -> Swing -> Drop -> Land/Miss -> Delay -> Spawn.
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        self.width = 400
        self.height = 600
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Physics constants
        self.gravity = 0.5
        self.rope_length = 200
        self.swing_speed = 0.04
        self.max_swing_angle = 45 * (math.pi / 180)
        
        # Block properties
        self.block_width = 80
        self.block_height = 80
        self.floor_height = 50
        
        # Game constants
        self.respawn_delay_frames = 40
        
        # Game state
        self.score = 0
        self.coins = 0
        self.combo = 0
        self.lives = 3
        self.game_over = False
        
        # Internal state machine
        # States: 'SPAWNING', 'SWINGING', 'FALLING', 'SETTLING'
        self.state = 'SPAWNING'
        self.state_timer = 0
        
        # Objects
        self.blocks: List[Dict] = []
        self.current_block = None
        self.swing_time = 0.0
        
        # Camera
        self.camera_y = 0.0
        self.target_camera_y = 0.0
        
        # Visual assets
        self.bg_buildings = self._generate_bg_buildings()
        
        # Initialize
        if render_mode == 'human':
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Tower Bloxx Simulator")
            self.clock = pygame.time.Clock()
            
    def _generate_bg_buildings(self):
        """Generate static background buildings."""
        buildings = []
        for i in range(5):
            w = random.randint(60, 120)
            h = random.randint(200, 500)
            x = i * 80 + random.randint(-20, 20)
            c = (random.randint(150, 200), random.randint(150, 200), random.randint(160, 210))
            buildings.append({'rect': (x, 600-h, w, h), 'color': c})
        return buildings
    
    def reset(self):
        """Reset the game state."""
        self.score = 0
        self.coins = 0
        self.combo = 0
        self.lives = 3
        self.game_over = False
        self.blocks = []
        self.camera_y = 0.0
        self.target_camera_y = 0.0
        
        # Add base block
        base_x = self.width // 2
        base_y = self.height - self.floor_height
        self.blocks.append({
            'x': base_x,
            'y': base_y,
            'vx': 0.0,
            'vy': 0.0,
            'width': self.block_width,
            'height': self.block_height,
            'settled': True,
            'color': (200, 200, 200),  # Gray base foundation
            'perfect': True,
            'is_base': True
        })
        
        # Initial wait
        self.state = 'SPAWNING'
        self.state_timer = 20
        self.current_block = None
        
        return self._get_observation()
    
    def _spawn_new_block(self):
        """Spawn a new block on the rope."""
        self.current_block = {
            'x': self.width // 2,
            'y': 150,
            'vx': 0.0,
            'vy': 0.0,
            'width': self.block_width,
            'height': self.block_height,
            'rotation': 0.0,
            'color': (255, 190, 0),  # Distinctive Orange/Yellow
            'rope_angle': 0.0,
            'perfect': False
        }
        
        # Start at the edges of the swing to prevent instant-drop exploits
        # sin(pi/2) = 1 (Max Right), sin(3pi/2) = -1 (Max Left)
        if random.random() > 0.5:
            self.swing_time = math.pi / 2
        else:
            self.swing_time = 3 * math.pi / 2
            
        self.state = 'SWINGING'
    
    def step(self, action: int):
        """
        Advance the simulation by one step.
        Args:
            action: 0 (wait) or 1 (drop)
        """
        # Event Loop
        if self.render_mode == 'human':
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True

        reward = 0.0
        if self.game_over:
            return self._get_observation(), 0, True, {}
            
        # --- STATE MACHINE ---
        
        if self.state == 'SPAWNING':
            self.state_timer -= 1
            if self.state_timer <= 0:
                self._spawn_new_block()
                
        elif self.state == 'SWINGING':
            # Pendulum Physics
            self.swing_time += self.swing_speed
            angle = math.sin(self.swing_time) * self.max_swing_angle
            self.current_block['rope_angle'] = angle
            
            # Anchor is fixed at top center (visualized or offscreen)
            anchor_x = self.width // 2
            anchor_y = -50 + self.camera_y * 0.2 # Slight parallax for anchor
            
            # Position
            self.current_block['x'] = anchor_x + math.sin(angle) * self.rope_length
            self.current_block['y'] = anchor_y + self.rope_length + 50 # Add rope length offset
            
            # Action: DROP
            if action == 1:
                self.state = 'FALLING'
                # Pass momentum
                # v = r * w * cos(wt)
                # This gives it that characteristic "fling" if released at the bottom
                swing_v = (math.cos(self.swing_time) * self.max_swing_angle * self.swing_speed) * 30
                self.current_block['vx'] = swing_v
                self.current_block['vy'] = 0.0
                
        elif self.state == 'FALLING':
            # Gravity
            self.current_block['vy'] += self.gravity
            self.current_block['y'] += self.current_block['vy']
            self.current_block['x'] += self.current_block['vx']
            
            # Collision Check
            top_block = self.blocks[-1]
            landed = False
            
            # Check bounding box Vertical overlap with tolerance
            # If bottom of cur passes top of last
            if (self.current_block['y'] + self.block_height/2) >= (top_block['y'] - self.block_height/2):
                # Verify we are close enough in Y to trigger collision (don't trigger if way below)
                dy_penetration = (self.current_block['y'] + self.block_height/2) - (top_block['y'] - self.block_height/2)
                
                if dy_penetration < 30: # Only collide if we just hit the surface
                    # Check Horizontal overlap
                    dx = abs(self.current_block['x'] - top_block['x'])
                    
                    if dx < self.block_width * 0.8:
                        # HIT
                        reward = self._land_block(dx)
                        landed = True
                    else:
                        # MISS (Will fall past)
                        pass
                        
            # Check if fell off screen
            if not landed and self.current_block['y'] > self.height + self.camera_y + 200:
                self._miss_block()
                reward = -1.0
                
        # Camera easing
        if self.target_camera_y > self.camera_y:
            self.camera_y += (self.target_camera_y - self.camera_y) * 0.1
            
        return self._get_observation(), reward, self.game_over, {
            'score': self.score,
            'lives': self.lives,
            'height': len(self.blocks),
            'combo': self.combo
        }

    def _land_block(self, dx):
        """Process landing logic."""
        top_block = self.blocks[-1]
        
        # Alignment Score (0.0 to 1.0)
        max_offset = self.block_width * 0.8
        alignment = max(0.0, 1.0 - (dx / max_offset))
        
        # Snap y
        self.current_block['y'] = top_block['y'] - self.block_height
        self.current_block['vy'] = 0
        self.current_block['vx'] = 0
        self.current_block['settled'] = True
        
        reward = 1.0 + (alignment ** 2 * 2.0)
        
        if dx < 10: # Perfect threshold
            self.current_block['perfect'] = True
            self.combo += 1
            self.score += 10 * self.combo
            reward += 2.0
            # Flash effect could be added here
        else:
            self.current_block['perfect'] = False
            self.combo = 0
            self.score += 10
            
        self.blocks.append(self.current_block)
        self.current_block = None
        self.target_camera_y += self.block_height
        
        self.state = 'SPAWNING'
        self.state_timer = self.respawn_delay_frames
        
        return reward

    def _miss_block(self):
        self.lives -= 1
        self.combo = 0
        self.current_block = None
        if self.lives <= 0:
            self.game_over = True
        else:
            self.state = 'SPAWNING'
            self.state_timer = self.respawn_delay_frames

    def _get_observation(self):
        # Setup surface
        if not self.window:
            surface = pygame.Surface((self.width, self.height))
        else:
            surface = self.window
            
        # Draw Sky
        surface.fill((180, 230, 255)) 
        
        # Draw Background Buildings (Parallax?)
        # Just static for now relative to camera to simulate distance? 
        # Actually usually they scroll slower.
        bg_offset = self.camera_y * 0.5
        for b in self.bg_buildings:
            r = list(b['rect'])
            r[1] += bg_offset # Parallax
            pygame.draw.rect(surface, b['color'], r)
        
        # Ground
        ground_y = self.height - self.floor_height + int(self.camera_y)
        pygame.draw.rect(surface, (100, 80, 60), (0, ground_y, self.width, 1000))
        pygame.draw.line(surface, (80, 60, 40), (0, ground_y), (self.width, ground_y), 4)

        # Draw Blocks
        for b in self.blocks:
             self._draw_block(surface, b)
             
        # Draw Crane System
        anchor_x = self.width // 2
        anchor_y = -50 + self.camera_y * 0.2
        
        # Crane line always visible
        if self.current_block:
             # Rope goes to block
            block_top = (self.current_block['x'], self.current_block['y'] - self.block_height/2)
            pygame.draw.line(surface, (50, 50, 50), (anchor_x, anchor_y), block_top, 4)
            # Draw Hook
            pygame.draw.rect(surface, (80, 80, 80), (block_top[0]-10, block_top[1]-10, 20, 10))
            
            self._draw_block(surface, self.current_block)
            
        else:
            # Empty Crane waiting
            # Draw rope to center
            hook_y = 100 + self.camera_y * 0.2 # Fixed-ish position
            pygame.draw.line(surface, (50, 50, 50), (anchor_x, anchor_y), (anchor_x, hook_y), 3)
            # Empty Hook
            pygame.draw.rect(surface, (80, 80, 80), (anchor_x-10, hook_y-5, 20, 10))

        # UI Overlay
        if self.window:
            self._draw_ui(surface)
            
        # Return Observation
        if not self.window:
            view = pygame.surfarray.array3d(surface)
            view = view.transpose([1, 0, 2])
            return view
        else:
            view = pygame.surfarray.array3d(self.window)
            view = view.transpose([1, 0, 2])
            return view

    def _draw_block(self, surface, b):
        """Draw a single block with windows detail."""
        x = int(b['x'] - b['width']/2)
        y = int(b['y'] - b['height']/2 + self.camera_y)
        w = int(b['width'])
        h = int(b['height'])
        
        rect = pygame.Rect(x, y, w, h)
        
        # Main body
        color = b.get('color', (200, 200, 200))
        if b.get('perfect'):
            color = (255, 255, 220) # Lighter if perfect
            
        pygame.draw.rect(surface, color, rect)
        
        # 3D edge effect (bottom/right darker)
        pygame.draw.rect(surface, (0, 0, 0), rect, 2)
        
        if b.get('is_base'):
             # Door for base
             pygame.draw.rect(surface, (50, 50, 60), (x + w//2 - 15, y + h - 40, 30, 40))
        else:
            # Windows (2x2 grid)
            win_color = (135, 206, 250)
            win_frame = (50, 50, 50)
            
            margin = 10
            win_w = (w - 3*margin) // 2
            win_h = (h - 3*margin) // 2
            
            # Top Left
            pygame.draw.rect(surface, win_color, (x+margin, y+margin, win_w, win_h))
            pygame.draw.rect(surface, win_frame, (x+margin, y+margin, win_w, win_h), 2)
            
            # Top Right
            pygame.draw.rect(surface, win_color, (x+2*margin+win_w, y+margin, win_w, win_h))
            pygame.draw.rect(surface, win_frame, (x+2*margin+win_w, y+margin, win_w, win_h), 2)
            
            # Bottom Left
            pygame.draw.rect(surface, win_color, (x+margin, y+2*margin+win_h, win_w, win_h))
            pygame.draw.rect(surface, win_frame, (x+margin, y+2*margin+win_h, win_w, win_h), 2)

            # Bottom Right
            pygame.draw.rect(surface, win_color, (x+2*margin+win_w, y+2*margin+win_h, win_w, win_h))
            pygame.draw.rect(surface, win_frame, (x+2*margin+win_w, y+2*margin+win_h, win_w, win_h), 2)
            
            if b.get('perfect') and random.random() < 0.1:
                # Occasional sparkle
                pass

    def _draw_ui(self, surface):
        font = pygame.font.Font(None, 36)
        
        # Score banner
        # pygame.draw.rect(surface, (0,0,0,100), (0,0, self.width, 50))
        text = font.render(f"{self.score}", True, (255, 215, 0))
        surface.blit(text, (self.width - 80, 10))
        
        lives_text = font.render(f"Lives: {self.lives}", True, (255, 50, 50))
        surface.blit(lives_text, (10, 10))
        
        if self.combo > 1:
            c_font = pygame.font.Font(None, 48)
            label = c_font.render(f"x{self.combo}", True, (0, 255, 0))
            surface.blit(label, (self.width//2 - 20, 60))
            
        pygame.display.flip()
        if self.clock:
            self.clock.tick(60)

    def close(self):
        if self.window:
            pygame.quit()

class FrameStacker:
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = None
        
    def reset(self, frame):
        # frame is H, W (2D)
        # We need H, W, N
        self.frames = np.stack([frame] * self.n_frames, axis=-1)
        return self.frames
        
    def add(self, frame):
        # Shift existing frames
        if len(frame.shape) == 3: # H, W, C
            # Not supported for multi-channel stacking in this simple implementation
            # We assume frame is H, W (grayscale)
            pass
        
        # frame is H, W
        # self.frames is H, W, N
        self.frames = np.dstack((self.frames[:, :, 1:], frame))
        return self.frames

class TowerBloxSimEnv(gym.Env):
    """
    Gym wrapper for the simulator.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, render_mode=None, grayscale=True, frame_stack=4):
        self.sim = TowerBloxSimulator(render_mode=render_mode)
        self.grayscale = grayscale
        self.frame_stack = frame_stack
        
        channels = frame_stack if grayscale else 3
        
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(84, 84, channels), 
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(2)
        
        self.stacker = FrameStacker(n_frames=frame_stack) if frame_stack > 1 else None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.sim.reset()
        processed_obs = self._process_obs(obs)
        
        if self.stacker:
            processed_obs = self.stacker.reset(processed_obs)
            
        return processed_obs, {}
        
    def step(self, action):
        obs, reward, done, sub_info = self.sim.step(action)
        processed_obs = self._process_obs(obs)
        
        if self.stacker:
            processed_obs = self.stacker.add(processed_obs)
            
        # Add sim-specific info
        info = sub_info
        info['episode_id'] = -1 # Dummy
        
        return processed_obs, reward, done, False, info
        
    def _process_obs(self, obs):
        import cv2
        # obs is H, W, 3 (RGB)
        
        # Resize first
        resized = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        
        if self.grayscale:
            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            return gray # H, W
            
        return resized # H, W, 3
        
    def render(self):
        return self.sim._get_observation()
        
    def close(self):
        self.sim.close()
