#!/usr/bin/env python
"""
Tower Bloxx RL Agent - Evaluation Entry Point

Usage:
    python evaluate.py --model model.zip       # Evaluate a model
    python evaluate.py --model model.zip -n 50 # Run 50 episodes
    python evaluate.py --model model.zip --render # Watch agent play

For full options:
    python evaluate.py --help
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.evaluator import Evaluator
from src.evaluation.visualizer import Visualizer
from src.utils.logger import setup_logger, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Tower Bloxx RL Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model file (.zip)",
    )
    
    parser.add_argument(
        "-n", "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes (watch agent play)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        default=None,
        help="Additional models to compare",
    )
    
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also test random baseline",
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots",
    )
    
    return parser.parse_args()


def main():
    """Main evaluation entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logger()
    logger = get_logger("evaluate")
    
    logger.info("=" * 60)
    logger.info("Tower Bloxx RL Agent - Evaluation")
    logger.info("=" * 60)
    
    try:
        # Create evaluator
        evaluator = Evaluator(device=args.device)
        
        # Compare multiple models
        if args.compare:
            all_models = [args.model] + args.compare
            logger.info(f"Comparing {len(all_models)} models")
            
            results = evaluator.compare_models(
                model_paths=all_models,
                n_episodes=args.episodes,
            )
            
            if args.plot:
                visualizer = Visualizer()
                visualizer.compare_models_plot(
                    results,
                    save_path="./logs/visualizations/model_comparison.png"
                )
        
        else:
            # Single model evaluation
            logger.info(f"Model: {args.model}")
            logger.info(f"Episodes: {args.episodes}")
            
            evaluator.load_model(args.model)
            
            results = evaluator.evaluate(
                n_episodes=args.episodes,
                deterministic=True,
                render=args.render,
            )
            
            # Print results
            logger.info("\n" + "=" * 40)
            logger.info("EVALUATION RESULTS")
            logger.info("=" * 40)
            logger.info(f"Episodes:      {results['n_episodes']}")
            logger.info(f"Mean Reward:   {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            logger.info(f"Min Reward:    {results['min_reward']:.2f}")
            logger.info(f"Max Reward:    {results['max_reward']:.2f}")
            logger.info(f"Mean Length:   {results['mean_length']:.1f}")
            logger.info(f"Total Steps:   {results['total_steps']:,}")
            logger.info(f"Action Distribution:")
            for action, count in results['action_distribution'].items():
                logger.info(f"  - {action}: {count}")
            
            # Generate plots
            if args.plot:
                visualizer = Visualizer()
                visualizer.plot_evaluation_results(
                    results,
                    save_path="./logs/visualizations/evaluation_results.png"
                )
        
        # Test random baseline
        if args.baseline:
            logger.info("\n" + "-" * 40)
            baseline_results = evaluator.test_random_baseline(n_episodes=args.episodes)
            
            logger.info("BASELINE VS TRAINED:")
            if 'results' in dir() and isinstance(results, dict):
                improvement = results['mean_reward'] - baseline_results['mean_reward']
                logger.info(f"  Improvement: {improvement:.2f} reward")
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove non-serializable items
            save_results = {k: v for k, v in results.items() if k not in ['rewards', 'lengths']}
            save_results['rewards_summary'] = {
                'all_rewards': results['rewards'],
                'all_lengths': results['lengths'],
            }
            
            with open(output_path, 'w') as f:
                json.dump(save_results, f, indent=2)
            
            logger.info(f"Results saved to: {output_path}")
        
        logger.info("\nEvaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
