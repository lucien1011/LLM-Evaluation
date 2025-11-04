"""
Visualization utilities for benchmark results
Supports MMLU and MMLU-Pro result visualization
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

    # Check for GSM8K (has reasoning_strategy and ~1319 questions in test set)
    # GSM8K files have reasoning_strategy field and total_questions in range 1000-2000
    if 'reasoning_strategy' in result:
        total = result.get('total_questions', 0)
        if 1000 <= total <= 2000:
            return 'gsm8k'

    # Check result count to distinguish remaining benchmarks
    total = result.get('total_questions', 0)
    if total > 50000:
        return 'hellaswag'
    elif total > 5000:
        return 'gsm8k'  # GSM8K train set (7473 questions)

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


# ============================================================================
# Enhanced Discovery Functions for Nested Structure (Reasoning Methods)
# ============================================================================

def detect_directory_structure(base_dir: Path) -> str:
    """
    Detect if directory structure is nested (by reasoning method) or flat

    Args:
        base_dir: Base directory to check

    Returns:
        'nested' if has subdirectories like direct/, zero-shot-cot/
        'flat' if JSON files are directly in base_dir
    """
    if not base_dir.exists():
        return 'flat'

    # Check for JSON files directly in base_dir
    json_files = list(base_dir.glob('*.json'))
    if json_files:
        return 'flat'

    # Check for subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return 'flat'

    # Check if subdirectories have names matching reasoning methods
    reasoning_dirs = {'direct', 'zero-shot-cot', 'few-shot-cot', 'self-consistency'}
    found_reasoning_dirs = {d.name for d in subdirs}

    if reasoning_dirs & found_reasoning_dirs:  # Intersection exists
        return 'nested'

    return 'flat'


def discover_nested_results(base_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover results in nested structure

    Args:
        base_dir: Base directory containing reasoning method subdirectories

    Returns:
        {
            'reasoning_method': [list of all result JSON files]
        }

    Example:
        {
            'direct': [Path('output/json/direct/arc_gemma3_27b.json'), ...],
            'zero-shot-cot': [Path('output/json/zero-shot-cot/arc_gemma3_27b.json'), ...]
        }
    """
    results = {}

    for reasoning_dir in base_dir.iterdir():
        if reasoning_dir.is_dir():
            reasoning_method = reasoning_dir.name
            json_files = list(reasoning_dir.glob('*.json'))
            if json_files:
                results[reasoning_method] = json_files

    return results


def load_results_for_comparison(
    base_dir: Path,
    benchmark: str,
    models: Optional[List[str]] = None,
    reasoning_methods: Optional[List[str]] = None
) -> Dict[str, Dict[str, Dict]]:
    """
    Load results for comparison across reasoning methods and models

    Args:
        base_dir: Base directory (either flat or with reasoning method subdirs)
        benchmark: Benchmark name to filter (e.g., 'arc', 'gpqa', 'mmlu')
        models: Optional list of specific models to include
        reasoning_methods: Optional list of specific reasoning methods to include

    Returns:
        {
            'reasoning_method': {
                'model_name': result_dict
            }
        }

    Example:
        {
            'direct': {
                'gemma3:27b': {...},
                'gemma3:4b': {...}
            },
            'zero-shot-cot': {
                'gemma3:27b': {...},
                'gemma3:4b': {...}
            }
        }
    """
    structure = detect_directory_structure(base_dir)

    if structure == 'nested':
        nested_results = discover_nested_results(base_dir)

        # Filter by reasoning methods if specified
        if reasoning_methods:
            nested_results = {
                k: v for k, v in nested_results.items()
                if k in reasoning_methods
            }

        # Load and filter results
        comparison_data = {}
        for reasoning_method, files in nested_results.items():
            comparison_data[reasoning_method] = {}

            for filepath in files:
                try:
                    result = load_result_file(str(filepath))
                    result_benchmark = detect_benchmark_type(result)

                    # Filter by benchmark
                    if result_benchmark != benchmark:
                        continue

                    model = extract_model_name(result)

                    # Filter by models if specified
                    if models and model not in models:
                        continue

                    comparison_data[reasoning_method][model] = result

                except Exception as e:
                    print(f"Error loading {filepath}: {e}")

        # Remove empty reasoning methods
        comparison_data = {k: v for k, v in comparison_data.items() if v}

        return comparison_data

    else:  # flat structure
        files = list(base_dir.glob('*.json'))

        # Skip batch_summary.json
        files = [f for f in files if f.name != 'batch_summary.json']

        comparison_data = {}

        for filepath in files:
            try:
                result = load_result_file(str(filepath))
                result_benchmark = detect_benchmark_type(result)

                if result_benchmark != benchmark:
                    continue

                model = extract_model_name(result)

                if models and model not in models:
                    continue

                # Extract reasoning method from filename or file content
                reasoning_method = None

                # Try to extract from filename first (e.g., arc_gemma3_1b_direct.json -> direct)
                filename = filepath.stem  # Get filename without extension
                for method in ['direct', 'zero-shot-cot', 'few-shot-cot', 'self-consistency']:
                    if method in filename:
                        reasoning_method = method
                        break

                # If not found in filename, try from result content
                if reasoning_method is None and 'reasoning_strategy' in result:
                    reasoning_method = result.get('reasoning_strategy', 'direct').lower()
                    # Clean up strategy name (e.g., "ZeroShotCoTStrategy" -> "zero-shot-cot")
                    if 'zeroshotcot' in reasoning_method.replace('-', ''):
                        reasoning_method = 'zero-shot-cot'
                    elif 'fewshotcot' in reasoning_method.replace('-', ''):
                        reasoning_method = 'few-shot-cot'
                    elif 'selfconsistency' in reasoning_method.replace('-', ''):
                        reasoning_method = 'self-consistency'
                    elif 'direct' in reasoning_method:
                        reasoning_method = 'direct'

                # Default to 'direct' if still not found
                if reasoning_method is None:
                    reasoning_method = 'direct'

                # Filter by reasoning_methods if specified
                if reasoning_methods and reasoning_method not in reasoning_methods:
                    continue

                # Add to comparison data structure
                if reasoning_method not in comparison_data:
                    comparison_data[reasoning_method] = {}

                comparison_data[reasoning_method][model] = result

            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        return comparison_data
