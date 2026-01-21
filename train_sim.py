#!/usr/bin/env python
"""
Tower Bloxx RL Agent - Simulator Training Entry Point

Usage:
    python train_sim.py                    # Train with default settings
    python train_sim.py --timesteps 500000 # Train for specific steps
    python train_sim.py --load model.zip   # Continue training from checkpoint

For full options:
    python train_sim.py --help
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import Trainer
from src.utils.logger import setup_logger, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Tower Bloxx RL Agent (Simulator)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (uses config if not specified)",
    )
    
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to model checkpoint to continue training from",
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="./configs",
        help="Path to configuration directory",
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Path to logging directory",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training",
    )
    
    parser.add_argument(
        "--reset-timesteps",
        action="store_true",
        help="Reset timestep counter when loading checkpoint",
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during training (will slow down training)",
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logger(log_dir=f"{args.log_dir}/training_logs_sim")
    logger = get_logger("train_sim")
    
    logger.info("=" * 60)
    logger.info("Tower Bloxx RL Agent - Simulator Training")
    logger.info("=" * 60)
    
    # Print configuration
    logger.info(f"Config directory: {args.config_dir}")
    logger.info(f"Log directory: {args.log_dir}")
    logger.info(f"Device: {args.device}")
    if args.load:
        logger.info(f"Loading checkpoint: {args.load}")
    if args.timesteps:
        logger.info(f"Training for: {args.timesteps:,} timesteps")
    
    try:
        # Create trainer with simulator enabled
        trainer = Trainer(
            config_dir=args.config_dir,
            device=args.device,
            log_dir=args.log_dir,
            use_simulator=True,
            render_mode="human" if args.render else None,
        )
        
        # Start training
        model = trainer.train(
            total_timesteps=args.timesteps,
            load_path=args.load,
            reset_timesteps=args.reset_timesteps,
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Final model saved to: {args.log_dir}/checkpoints/")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # raise
        
    finally:
        if 'trainer' in locals():
            trainer.close()


if __name__ == "__main__":
    main()
