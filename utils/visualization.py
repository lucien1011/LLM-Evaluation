"""
Visualization utilities for benchmark results
Supports MMLU and MMLU-Pro result visualization
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


def load_result_file(filepath: str) -> Dict:
    """
    Load a benchmark result JSON file

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary with benchmark results
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_model_name(result: Dict) -> str:
    """
    Extract model name from result

    Args:
        result: Result dictionary

    Returns:
        Model name string
    """
    return result.get('model', 'Unknown')


def extract_overall_accuracy(result: Dict) -> float:
    """
    Extract overall accuracy from result

    Args:
        result: Result dictionary

    Returns:
        Accuracy as percentage
    """
    return result.get('overall_accuracy', 0.0)


def extract_subject_accuracies(result: Dict) -> Dict[str, float]:
    """
    Extract per-subject/category accuracies

    Args:
        result: Result dictionary

    Returns:
        Dictionary mapping subject/category name to accuracy
    """
    subject_accuracies = {}

    # Check if it's MMLU-Pro format (has dataset_stats)
    if 'dataset_stats' in result:
        # MMLU-Pro format - categories are in dataset_stats
        # But we don't have per-category results in this format
        # For now, just return overall
        return {"Overall": result.get('overall_accuracy', 0.0)}

    # MMLU format with subject_results
    if 'subject_results' in result:
        for subject_result in result['subject_results']:
            if 'error' not in subject_result:
                subject = subject_result.get('subject', 'Unknown')
                accuracy = subject_result.get('accuracy', 0.0)
                subject_accuracies[subject] = accuracy

    return subject_accuracies


def detect_benchmark_type(result: Dict) -> str:
    """
    Detect benchmark type from result file

    Args:
        result: Result dictionary

    Returns:
        Benchmark type: 'mmlu', 'mmlu-pro', 'truthfulqa', 'arc', 'hellaswag', 'gpqa', 'gsm8k', 'humaneval'
    """
    # Check for explicit benchmark field
    if 'benchmark' in result:
        return result['benchmark'].lower()

    # Check for format field (TruthfulQA)
    if 'format' in result:
        return 'truthfulqa'

    # Check for subset field (GPQA)
    if 'subset' in result and result['subset'].startswith('gpqa'):
        return 'gpqa'

    # Check for difficulty field (ARC)
    if 'difficulty' in result and result['difficulty'].startswith('ARC'):
        return 'arc'

    # Check for pass_at_1 field (HumanEval)
    if 'pass_at_1' in result:
        return 'humaneval'

    # Check for dataset_stats with categories (MMLU-Pro)
    if 'dataset_stats' in result and 'categories' in result.get('dataset_stats', {}):
        return 'mmlu-pro'

    # Check for subject_results (MMLU)
    if 'subject_results' in result:
        return 'mmlu'

    # Check result count to distinguish (GSM8K has ~1000-8500, HellaSwag has ~70000)
    total = result.get('total_questions', 0)
    if total > 50000:
        return 'hellaswag'
    elif total > 5000:
        return 'gsm8k'

    # Default to mmlu
    return 'mmlu'


def find_result_files(directory: str, pattern: str = "*.json") -> List[Path]:
    """
    Find all result JSON files in directory

    Args:
        directory: Directory to search
        pattern: File pattern (default: *.json)

    Returns:
        List of Path objects
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    return list(dir_path.glob(pattern))


def filter_results_by_benchmark(
    result_files: List[Path],
    benchmark_name: str = None
) -> List[Tuple[Path, Dict]]:
    """
    Filter result files by benchmark type

    Args:
        result_files: List of result file paths
        benchmark_name: 'mmlu', 'mmlu-pro', or None for all

    Returns:
        List of tuples (filepath, result_dict)
    """
    filtered = []

    for filepath in result_files:
        try:
            result = load_result_file(filepath)
            benchmark_type = detect_benchmark_type(result)

            if benchmark_name is None or benchmark_type == benchmark_name:
                filtered.append((filepath, result))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return filtered


def filter_results_by_models(
    results: List[Tuple[Path, Dict]],
    model_names: List[str] = None
) -> List[Tuple[Path, Dict]]:
    """
    Filter results by model names

    Args:
        results: List of (filepath, result_dict) tuples
        model_names: List of model names to include (None for all)

    Returns:
        Filtered list of (filepath, result_dict) tuples
    """
    if model_names is None:
        return results

    filtered = []
    for filepath, result in results:
        model_name = extract_model_name(result)
        if model_name in model_names:
            filtered.append((filepath, result))

    return filtered


def create_timestamp() -> str:
    """
    Create timestamp string for filename

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_color_palette(n_colors: int) -> List[str]:
    """
    Get a color palette for plotting

    Args:
        n_colors: Number of colors needed

    Returns:
        List of color hex codes
    """
    # Use a nice color palette
    base_colors = [
        '#3498db',  # Blue
        '#e74c3c',  # Red
        '#2ecc71',  # Green
        '#f39c12',  # Orange
        '#9b59b6',  # Purple
        '#1abc9c',  # Turquoise
        '#e67e22',  # Carrot
        '#95a5a6',  # Gray
        '#34495e',  # Dark blue
        '#16a085',  # Green sea
    ]

    if n_colors <= len(base_colors):
        return base_colors[:n_colors]

    # If we need more colors, use matplotlib's colormap
    cmap = plt.cm.get_cmap('tab20')
    return [plt.cm.colors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n_colors)]


def setup_plot_style():
    """Setup matplotlib plot style for better-looking plots"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
