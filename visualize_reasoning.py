#!/usr/bin/env python3
"""
Enhanced Visualization Script for Reasoning Method Comparisons

This script provides comprehensive visualization capabilities for comparing
benchmark results across different reasoning methods and models.

Supports nested directory structure:
  output/json/
    ├── direct/
    ├── zero-shot-cot/
    ├── few-shot-cot/
    └── self-consistency/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

# Import visualization functions
from utils.visualization import (
    detect_directory_structure,
    load_results_for_comparison,
    create_timestamp
)

# Import plotting functions
from visualize_results import (
    plot_reasoning_comparison,
    plot_model_comparison_by_reasoning,
    plot_matrix_heatmap,
    plot_improvement_chart
)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize benchmark results across reasoning methods and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Compare reasoning methods for one model
  python visualize_reasoning.py \\
      --benchmark arc \\
      --model "gemma3:27b" \\
      --compare reasoning

  # Compare models for one reasoning method
  python visualize_reasoning.py \\
      --benchmark arc \\
      --reasoning-method zero-shot-cot \\
      --compare models

  # Full matrix heatmap (all combinations)
  python visualize_reasoning.py \\
      --benchmark arc \\
      --compare matrix

  # Show CoT improvements over baseline
  python visualize_reasoning.py \\
      --benchmark arc \\
      --compare improvement

  # Filter specific models and reasoning methods
  python visualize_reasoning.py \\
      --benchmark gpqa \\
      --models "gemma3:27b" "gpt-oss:20b" \\
      --reasoning-methods direct zero-shot-cot \\
      --compare matrix
        """
    )

    # Required arguments
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        help='Benchmark to visualize (arc, gpqa, mmlu, hellaswag, etc.)'
    )

    # Directory arguments
    parser.add_argument(
        '--input-dir',
        type=str,
        default='output/json',
        help='Base directory containing result files (default: output/json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/plots',
        help='Directory to save plots (default: output/plots)'
    )

    # Comparison mode
    parser.add_argument(
        '--compare',
        type=str,
        choices=['reasoning', 'models', 'matrix', 'improvement', 'all'],
        default='all',
        help='Comparison mode (default: all)'
    )

    # Filtering arguments
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Specific model to analyze (required for --compare reasoning)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Specific models to include in comparison'
    )
    parser.add_argument(
        '--reasoning-method',
        type=str,
        default=None,
        help='Specific reasoning method to analyze (required for --compare models)'
    )
    parser.add_argument(
        '--reasoning-methods',
        type=str,
        nargs='+',
        default=None,
        help='Specific reasoning methods to include in comparison'
    )

    # Additional options
    parser.add_argument(
        '--baseline',
        type=str,
        default='direct',
        help='Baseline reasoning method for improvement comparison (default: direct)'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default=None,
        help='Custom filename prefix for plots (default: auto-generated)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.compare == 'reasoning' and not args.model:
        parser.error("--compare reasoning requires --model argument")
    if args.compare == 'models' and not args.reasoning_method:
        parser.error("--compare models requires --reasoning-method argument")

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect structure
    structure = detect_directory_structure(input_dir)
    print(f"Detected directory structure: {structure}")
    print(f"Input directory: {input_dir}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Comparison mode: {args.compare}")
    print("-" * 70)

    # Load results
    print("Loading results...")
    data = load_results_for_comparison(
        input_dir,
        args.benchmark,
        models=args.models,
        reasoning_methods=args.reasoning_methods
    )

    if not data:
        print(f"Error: No results found for benchmark '{args.benchmark}' in {input_dir}")
        sys.exit(1)

    # Print summary
    print(f"\nFound results:")
    for reasoning_method, models_dict in data.items():
        print(f"  {reasoning_method}: {len(models_dict)} model(s) - {list(models_dict.keys())}")
    print()

    # Generate filename prefix
    if args.filename:
        filename_prefix = args.filename
    else:
        timestamp = create_timestamp()
        filename_prefix = f"{args.benchmark}_{timestamp}"

    # Generate plots based on comparison mode
    if args.compare == 'reasoning' or args.compare == 'all':
        if args.model or args.compare == 'all':
            # Get all models if 'all' mode
            all_models = set()
            for models_dict in data.values():
                all_models.update(models_dict.keys())

            models_to_plot = [args.model] if args.model else sorted(all_models)

            for model in models_to_plot:
                output_path = output_dir / f"{filename_prefix}_reasoning_{model.replace(':', '_')}.png"
                print(f"Generating reasoning comparison for {model}...")
                plot_reasoning_comparison(data, model, args.benchmark, output_path)

    if args.compare == 'models' or args.compare == 'all':
        if args.reasoning_method or args.compare == 'all':
            # Get all reasoning methods if 'all' mode
            methods_to_plot = [args.reasoning_method] if args.reasoning_method else sorted(data.keys())

            for method in methods_to_plot:
                output_path = output_dir / f"{filename_prefix}_models_{method}.png"
                print(f"Generating model comparison for {method}...")
                plot_model_comparison_by_reasoning(data, method, args.benchmark, output_path)

    if args.compare == 'matrix' or args.compare == 'all':
        output_path = output_dir / f"{filename_prefix}_matrix.png"
        print(f"Generating matrix heatmap...")
        plot_matrix_heatmap(data, args.benchmark, output_path)

    if args.compare == 'improvement' or args.compare == 'all':
        if args.baseline in data:
            output_path = output_dir / f"{filename_prefix}_improvement.png"
            print(f"Generating improvement chart (baseline: {args.baseline})...")
            plot_improvement_chart(data, args.benchmark, output_path, baseline=args.baseline)
        else:
            print(f"Warning: Baseline '{args.baseline}' not found in results. Skipping improvement chart.")

    print()
    print("=" * 70)
    print("Visualization complete!")
    print(f"Plots saved to: {output_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
