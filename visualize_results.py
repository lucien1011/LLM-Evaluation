#!/usr/bin/env python3
"""
Benchmark Results Visualization Script

Creates comparison plots of benchmark results across multiple models
Supports both MMLU and MMLU-Pro benchmarks
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from utils.visualization import (
    load_result_file,
    extract_model_name,
    extract_overall_accuracy,
    extract_subject_accuracies,
    detect_benchmark_type,
    find_result_files,
    filter_results_by_benchmark,
    filter_results_by_models,
    create_timestamp,
    get_color_palette,
    setup_plot_style
)


def plot_overall_comparison(
    results: List[Tuple[Path, Dict]],
    output_path: Path,
    benchmark_name: str = None
):
    """
    Create a bar chart comparing overall accuracy across models

    Args:
        results: List of (filepath, result_dict) tuples
        output_path: Path to save the plot
        benchmark_name: Name of benchmark for title
    """
    if not results:
        print("No results to plot")
        return

    setup_plot_style()

    # Extract data
    models = []
    accuracies = []

    for filepath, result in results:
        model = extract_model_name(result)
        accuracy = extract_overall_accuracy(result)
        models.append(model)
        accuracies.append(accuracy)

    # Sort by accuracy (descending)
    sorted_data = sorted(zip(models, accuracies), key=lambda x: x[1], reverse=True)
    models, accuracies = zip(*sorted_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = get_color_palette(len(models))
    bars = ax.bar(range(len(models)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Customize plot
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')

    title = f'{benchmark_name.upper() if benchmark_name else "Benchmark"} - Overall Accuracy Comparison'
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add reference lines
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1, label='50% baseline')
    ax.axhline(y=25, color='orange', linestyle='--', alpha=0.3, linewidth=1, label='25% (random MMLU)')

    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved overall comparison plot to: {output_path}")
    plt.close()


def plot_subject_comparison(
    results: List[Tuple[Path, Dict]],
    output_path: Path,
    benchmark_name: str = None,
    top_n_subjects: int = 15
):
    """
    Create a grouped bar chart comparing accuracy across subjects/categories

    Args:
        results: List of (filepath, result_dict) tuples
        output_path: Path to save the plot
        benchmark_name: Name of benchmark for title
        top_n_subjects: Number of top subjects to show
    """
    if not results:
        print("No results to plot")
        return

    setup_plot_style()

    # Extract data
    model_subject_data = {}

    for filepath, result in results:
        model = extract_model_name(result)
        subject_accs = extract_subject_accuracies(result)

        if subject_accs and len(subject_accs) > 1:  # Skip if only overall
            model_subject_data[model] = subject_accs

    if not model_subject_data:
        print("No subject-level data available for plotting")
        return

    # Get all subjects across all models
    all_subjects = set()
    for subject_accs in model_subject_data.values():
        all_subjects.update(subject_accs.keys())

    all_subjects = sorted(all_subjects)

    # If too many subjects, select top N by average accuracy
    if len(all_subjects) > top_n_subjects:
        subject_avg_accs = {}
        for subject in all_subjects:
            accs = [model_subject_data[model].get(subject, 0)
                   for model in model_subject_data.keys()]
            subject_avg_accs[subject] = np.mean(accs)

        top_subjects = sorted(subject_avg_accs.items(),
                             key=lambda x: x[1],
                             reverse=True)[:top_n_subjects]
        subjects = [s[0] for s in top_subjects]
    else:
        subjects = all_subjects

    # Prepare data for plotting
    models = list(model_subject_data.keys())
    n_models = len(models)
    n_subjects = len(subjects)

    # Create plot
    fig, ax = plt.subplots(figsize=(max(12, n_subjects * 0.8), 8))

    bar_width = 0.8 / n_models
    x = np.arange(n_subjects)
    colors = get_color_palette(n_models)

    # Plot bars for each model
    for i, model in enumerate(models):
        subject_accs = model_subject_data[model]
        accuracies = [subject_accs.get(subject, 0) for subject in subjects]

        offset = (i - n_models/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, accuracies, bar_width,
                     label=model, color=colors[i], alpha=0.8,
                     edgecolor='black', linewidth=0.5)

    # Customize plot
    ax.set_xlabel('Subject / Category', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')

    title = f'{benchmark_name.upper() if benchmark_name else "Benchmark"} - Per-Subject Accuracy Comparison'
    if len(all_subjects) > top_n_subjects:
        title += f' (Top {top_n_subjects})'
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved subject comparison plot to: {output_path}")
    plt.close()


def plot_radar_comparison(
    results: List[Tuple[Path, Dict]],
    output_path: Path,
    benchmark_name: str = None,
    max_subjects: int = 12
):
    """
    Create a radar chart comparing models across subjects

    Args:
        results: List of (filepath, result_dict) tuples
        output_path: Path to save the plot
        benchmark_name: Name of benchmark for title
        max_subjects: Maximum number of subjects to show
    """
    if not results:
        print("No results to plot")
        return

    setup_plot_style()

    # Extract data
    model_subject_data = {}

    for filepath, result in results:
        model = extract_model_name(result)
        subject_accs = extract_subject_accuracies(result)

        if subject_accs and len(subject_accs) > 1:
            model_subject_data[model] = subject_accs

    if not model_subject_data:
        print("No subject-level data available for radar plot")
        return

    # Get common subjects
    all_subjects = set()
    for subject_accs in model_subject_data.values():
        all_subjects.update(subject_accs.keys())

    subjects = sorted(all_subjects)[:max_subjects]
    n_subjects = len(subjects)

    if n_subjects < 3:
        print("Need at least 3 subjects for radar plot")
        return

    # Prepare data
    models = list(model_subject_data.keys())
    angles = np.linspace(0, 2 * np.pi, n_subjects, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    colors = get_color_palette(len(models))

    for i, model in enumerate(models):
        subject_accs = model_subject_data[model]
        values = [subject_accs.get(subject, 0) for subject in subjects]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=model,
               color=colors[i], markersize=6)
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # Customize plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(subjects, size=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'])
    ax.grid(True, alpha=0.3)

    title = f'{benchmark_name.upper() if benchmark_name else "Benchmark"} - Radar Comparison'
    plt.title(title, fontweight='bold', fontsize=14, pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved radar comparison plot to: {output_path}")
    plt.close()


def plot_heatmap_comparison(
    results: List[Tuple[Path, Dict]],
    output_path: Path,
    benchmark_name: str = None
):
    """
    Create a heatmap comparing models across subjects

    Args:
        results: List of (filepath, result_dict) tuples
        output_path: Path to save the plot
        benchmark_name: Name of benchmark for title
    """
    if not results:
        print("No results to plot")
        return

    setup_plot_style()

    # Extract data
    model_subject_data = {}

    for filepath, result in results:
        model = extract_model_name(result)
        subject_accs = extract_subject_accuracies(result)

        if subject_accs and len(subject_accs) > 1:
            model_subject_data[model] = subject_accs

    if not model_subject_data:
        print("No subject-level data available for heatmap")
        return

    # Get all subjects
    all_subjects = set()
    for subject_accs in model_subject_data.values():
        all_subjects.update(subject_accs.keys())

    subjects = sorted(all_subjects)
    models = list(model_subject_data.keys())

    # Create matrix
    matrix = np.zeros((len(models), len(subjects)))
    for i, model in enumerate(models):
        for j, subject in enumerate(subjects):
            matrix[i, j] = model_subject_data[model].get(subject, 0)

    # Create plot
    fig, ax = plt.subplots(figsize=(max(12, len(subjects) * 0.6), max(6, len(models) * 0.8)))

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)

    # Set ticks
    ax.set_xticks(np.arange(len(subjects)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.set_yticklabels(models)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontweight='bold')

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(subjects)):
            text = ax.text(j, i, f'{matrix[i, j]:.0f}',
                         ha="center", va="center", color="black", fontsize=8)

    # Customize plot
    title = f'{benchmark_name.upper() if benchmark_name else "Benchmark"} - Accuracy Heatmap'
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap comparison to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize benchmark results across models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all MMLU results in output/json/
  python visualize_results.py --benchmark mmlu --input-dir output/json

  # Visualize MMLU-Pro with specific models
  python visualize_results.py --benchmark mmlu-pro --input-dir output/json \\
      --models "gemma3:4b" "gemma3:27b"

  # Save to custom output directory
  python visualize_results.py --benchmark mmlu --input-dir output/json \\
      --output-dir plots/

  # Generate all plot types
  python visualize_results.py --benchmark mmlu --input-dir output/json \\
      --plot-types overall subject radar heatmap
        """
    )

    parser.add_argument(
        '--benchmark',
        type=str,
        default=None,
        choices=['mmlu', 'mmlu-pro', None],
        help='Benchmark type to visualize (default: auto-detect from files)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='output/json',
        help='Directory containing result JSON files (default: output/json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/plots',
        help='Directory to save plots (default: output/plots)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Specific models to include (default: all models found)'
    )
    parser.add_argument(
        '--plot-types',
        type=str,
        nargs='+',
        default=['overall', 'subject'],
        choices=['overall', 'subject', 'radar', 'heatmap'],
        help='Types of plots to generate (default: overall subject)'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default='benchmark_comparison',
        help='Base filename for plots (default: benchmark_comparison)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find result files
    print(f"Searching for result files in: {args.input_dir}")
    result_files = find_result_files(args.input_dir)

    if not result_files:
        print(f"No JSON files found in {args.input_dir}")
        return

    print(f"Found {len(result_files)} result file(s)")

    # Filter by benchmark type
    results = filter_results_by_benchmark(result_files, args.benchmark)

    if not results:
        print(f"No results matching benchmark type: {args.benchmark}")
        return

    print(f"Loaded {len(results)} result(s) for benchmark: {args.benchmark or 'auto'}")

    # Detect benchmark if not specified
    if args.benchmark is None and results:
        args.benchmark = detect_benchmark_type(results[0][1])
        print(f"Auto-detected benchmark type: {args.benchmark}")

    # Filter by models if specified
    if args.models:
        results = filter_results_by_models(results, args.models)
        print(f"Filtered to {len(results)} result(s) for specified models")

    if not results:
        print("No results after filtering")
        return

    # Print summary
    print("\nResults to visualize:")
    for filepath, result in results:
        model = extract_model_name(result)
        accuracy = extract_overall_accuracy(result)
        print(f"  - {model}: {accuracy:.2f}% (from {filepath.name})")

    # Generate timestamp
    timestamp = create_timestamp()

    # Generate plots
    print(f"\nGenerating plots...")

    if 'overall' in args.plot_types:
        output_path = output_dir / f"{args.filename}_overall_{timestamp}.png"
        plot_overall_comparison(results, output_path, args.benchmark)

    if 'subject' in args.plot_types:
        output_path = output_dir / f"{args.filename}_subjects_{timestamp}.png"
        plot_subject_comparison(results, output_path, args.benchmark)

    if 'radar' in args.plot_types:
        output_path = output_dir / f"{args.filename}_radar_{timestamp}.png"
        plot_radar_comparison(results, output_path, args.benchmark)

    if 'heatmap' in args.plot_types:
        output_path = output_dir / f"{args.filename}_heatmap_{timestamp}.png"
        plot_heatmap_comparison(results, output_path, args.benchmark)

    print(f"\nDone! All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
