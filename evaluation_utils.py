"""
General evaluation utilities for benchmarking
Can be reused across different benchmark datasets
"""

import json
import time
from typing import Dict, List, Callable, Any
from pathlib import Path
from tqdm import tqdm


def evaluate_sample(
    sample: Dict,
    evaluator_fn: Callable[[Dict], tuple]
) -> Dict:
    """
    Evaluate a single sample using provided evaluator function

    Args:
        sample: Sample data dictionary
        evaluator_fn: Function that takes sample and returns (is_correct, predicted, correct)

    Returns:
        Dictionary with evaluation results
    """
    is_correct, predicted, correct = evaluator_fn(sample)

    return {
        "is_correct": is_correct,
        "predicted": predicted,
        "correct": correct
    }


def evaluate_dataset(
    dataset: Any,
    evaluator_fn: Callable[[Dict], tuple],
    max_samples: int = None,
    delay: float = 0.1,
    description: str = "Evaluating"
) -> Dict:
    """
    Evaluate an entire dataset

    Args:
        dataset: Dataset to evaluate (iterable)
        evaluator_fn: Function that evaluates each sample
        max_samples: Maximum number of samples to evaluate (None for all)
        delay: Delay in seconds between samples
        description: Description for progress bar

    Returns:
        Dictionary with evaluation statistics
    """
    if max_samples:
        # Handle both list and dataset objects
        if hasattr(dataset, 'select'):
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        else:
            dataset = list(dataset)[:max_samples]

    correct = 0
    total = 0
    results = []

    for sample in tqdm(dataset, desc=description):
        is_correct, predicted, correct_answer = evaluator_fn(sample)

        correct += int(is_correct)
        total += 1

        results.append({
            "is_correct": is_correct,
            "predicted": predicted,
            "correct": correct_answer
        })

        if delay > 0:
            time.sleep(delay)

    accuracy = (correct / total * 100) if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }


def calculate_overall_stats(subject_results: List[Dict]) -> Dict:
    """
    Calculate overall statistics from multiple subject results

    Args:
        subject_results: List of per-subject result dictionaries

    Returns:
        Dictionary with overall statistics
    """
    # Filter out results with errors
    valid_results = [r for r in subject_results if "error" not in r]

    total_correct = sum(r.get("correct", 0) for r in valid_results)
    total_questions = sum(r.get("total", 0) for r in valid_results)
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0

    return {
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "num_subjects": len(valid_results)
    }


def save_results(results: Dict, output_path: str) -> Path:
    """
    Save evaluation results to JSON file

    Args:
        results: Results dictionary to save
        output_path: Path to output file

    Returns:
        Path object of saved file
    """
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    return output_path


def print_results_summary(results: Dict, show_subjects: bool = True):
    """
    Print a formatted summary of benchmark results

    Args:
        results: Results dictionary
        show_subjects: Whether to show per-subject breakdown
    """
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Model: {results.get('model', 'N/A')}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Correct: {results['total_correct']}/{results['total_questions']}")
    print(f"Subjects Evaluated: {results['num_subjects']}")

    if 'elapsed_time_seconds' in results:
        print(f"Time Elapsed: {results['elapsed_time_seconds']:.2f} seconds")

    if show_subjects and 'subject_results' in results:
        print("\nPer-Subject Accuracy:")
        print("-" * 50)
        for subject_result in results['subject_results']:
            if "error" not in subject_result:
                subject_name = subject_result.get('subject', 'Unknown')
                accuracy = subject_result.get('accuracy', 0)
                print(f"{subject_name:<40} {accuracy:>6.2f}%")
