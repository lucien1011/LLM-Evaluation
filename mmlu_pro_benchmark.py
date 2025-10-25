#!/usr/bin/env python3
"""
MMLU-Pro Benchmark Script for Ollama Models (Modular Version)

MMLU-Pro is a more challenging version of MMLU with:
- 12,000+ questions (more complex than original MMLU)
- 10 answer choices (A-J) instead of 4 (A-D)
- 14 categories instead of 57 subjects
- Chain-of-thought explanations included

Evaluates language models on the MMLU-Pro benchmark using Ollama
"""

import argparse
import time
from typing import Dict, List, Tuple
from pathlib import Path

# Import our modular utilities
from ollama_utils import check_ollama_connection, query_ollama
from prompt_utils import format_multiple_choice_prompt, extract_letter_answer
from evaluation_utils import (
    evaluate_dataset,
    calculate_overall_stats,
    save_results,
    print_results_summary
)
from mmlu_pro_utils import (
    load_mmlu_pro_dataset,
    filter_by_category,
    parse_mmlu_pro_sample,
    get_sample_category,
    MMLU_PRO_CATEGORIES,
    get_dataset_stats
)


def create_mmlu_pro_evaluator(model_name: str, ollama_url: str):
    """
    Create an evaluator function for MMLU-Pro samples

    Args:
        model_name: Name of the Ollama model
        ollama_url: URL of the Ollama API

    Returns:
        Function that evaluates a single MMLU-Pro sample
    """
    # Valid choices for MMLU-Pro (A through J)
    valid_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    def evaluate_mmlu_pro_sample(sample: Dict) -> Tuple[bool, str, str]:
        """Evaluate a single MMLU-Pro sample"""
        # Parse the MMLU-Pro sample
        question, choices, correct_letter = parse_mmlu_pro_sample(sample)

        # Custom instruction for 10 choices
        instruction = (
            "Answer the following multiple choice question by selecting the correct option.\n"
            "The options are labeled A through J (10 choices).\n"
            "Only respond with the letter of your answer (A, B, C, D, E, F, G, H, I, or J), nothing else."
        )

        # Format the prompt
        prompt = format_multiple_choice_prompt(question, choices, instruction)

        # Query the model
        response = query_ollama(
            prompt=prompt,
            model_name=model_name,
            ollama_url=ollama_url,
            temperature=0.0,
            max_tokens=20  # Slightly more tokens since explanation might be longer
        )

        # Extract the predicted answer (with valid choices A-J)
        predicted_letter = extract_letter_answer(response, valid_choices)

        # Check if correct
        is_correct = predicted_letter == correct_letter

        return is_correct, predicted_letter, correct_letter

    return evaluate_mmlu_pro_sample


def benchmark_mmlu_pro_category(
    category: str,
    model_name: str,
    ollama_url: str,
    split: str = "test",
    max_samples: int = None
) -> Dict:
    """
    Benchmark a specific MMLU-Pro category

    Args:
        category: MMLU-Pro category name
        model_name: Name of the Ollama model
        ollama_url: URL of the Ollama API
        split: Dataset split ('test' or 'validation')
        max_samples: Maximum number of samples to evaluate (None for all)

    Returns:
        Dictionary with evaluation results
    """
    print(f"\nEvaluating category: {category}")

    # Load the full dataset
    full_dataset = load_mmlu_pro_dataset(split)
    if full_dataset is None:
        return {"subject": category, "error": "Failed to load dataset"}

    # Filter by category
    dataset = filter_by_category(full_dataset, category)

    if len(dataset) == 0:
        print(f"Warning: No samples found for category '{category}'")
        return {"subject": category, "error": "No samples found"}

    print(f"Found {len(dataset)} samples in category '{category}'")

    # Create evaluator function
    evaluator_fn = create_mmlu_pro_evaluator(model_name, ollama_url)

    # Evaluate the dataset
    results = evaluate_dataset(
        dataset=dataset,
        evaluator_fn=evaluator_fn,
        max_samples=max_samples,
        delay=0.1,
        description=f"Processing {category}"
    )

    # Add category name to results (using 'subject' key for compatibility)
    results["subject"] = category
    results["category"] = category

    return results


def benchmark_mmlu_pro_all(
    model_name: str,
    ollama_url: str,
    categories: List[str] = None,
    split: str = "test",
    max_samples_per_category: int = None
) -> Dict:
    """
    Benchmark multiple MMLU-Pro categories or entire dataset

    Args:
        model_name: Name of the Ollama model
        ollama_url: URL of the Ollama API
        categories: List of categories to benchmark (None for all)
        split: Dataset split to use ('test' or 'validation')
        max_samples_per_category: Max samples per category (None for all)

    Returns:
        Dictionary with overall results
    """
    start_time = time.time()

    # If no specific categories, evaluate entire dataset without filtering
    if categories is None:
        print("\nEvaluating entire MMLU-Pro dataset (all categories)")

        # Load full dataset
        dataset = load_mmlu_pro_dataset(split)
        if dataset is None:
            return {
                "model": model_name,
                "error": "Failed to load dataset"
            }

        # Show dataset statistics
        stats = get_dataset_stats(dataset)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Categories: {stats['num_categories']}")
        for cat, count in sorted(stats['categories'].items()):
            print(f"  - {cat}: {count} samples")

        # Create evaluator
        evaluator_fn = create_mmlu_pro_evaluator(model_name, ollama_url)

        # Evaluate entire dataset
        results = evaluate_dataset(
            dataset=dataset,
            evaluator_fn=evaluator_fn,
            max_samples=max_samples_per_category,
            delay=0.1,
            description="Processing MMLU-Pro"
        )

        elapsed_time = time.time() - start_time

        return {
            "model": model_name,
            "overall_accuracy": results["accuracy"],
            "total_correct": results["correct"],
            "total_questions": results["total"],
            "num_subjects": stats["num_categories"],
            "elapsed_time_seconds": elapsed_time,
            "dataset_stats": stats,
            "all_results": results["results"]
        }

    # Otherwise, evaluate by category
    subject_results = []

    for category in categories:
        result = benchmark_mmlu_pro_category(
            category=category,
            model_name=model_name,
            ollama_url=ollama_url,
            split=split,
            max_samples=max_samples_per_category
        )
        subject_results.append(result)

    # Calculate overall statistics
    overall_stats = calculate_overall_stats(subject_results)
    elapsed_time = time.time() - start_time

    return {
        "model": model_name,
        "overall_accuracy": overall_stats["overall_accuracy"],
        "total_correct": overall_stats["total_correct"],
        "total_questions": overall_stats["total_questions"],
        "num_subjects": overall_stats["num_subjects"],
        "elapsed_time_seconds": elapsed_time,
        "subject_results": subject_results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama models on MMLU-Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available categories:
  {', '.join(MMLU_PRO_CATEGORIES)}

Examples:
  # Quick test with 5 samples total
  python mmlu_pro_benchmark.py --model gemma3:latest --max-samples 5

  # Test specific categories
  python mmlu_pro_benchmark.py --model gemma3:latest --categories math physics

  # Full benchmark
  python mmlu_pro_benchmark.py --model gemma3:latest
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:latest",
        help="Ollama model name (default: gemma3:latest)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Specific categories to benchmark (default: entire dataset)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples total or per category (default: all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Dataset split to use (default: test)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mmlu_pro_results.json",
        help="Output file for results (default: mmlu_pro_results.json)"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available categories and exit"
    )

    args = parser.parse_args()

    # Handle list categories
    if args.list_categories:
        print("MMLU-Pro Categories:")
        for i, cat in enumerate(MMLU_PRO_CATEGORIES, 1):
            print(f"  {i:2d}. {cat}")
        return

    print(f"Starting MMLU-Pro Benchmark")
    print(f"Model: {args.model}")
    print(f"Ollama URL: {args.url}")
    print(f"Split: {args.split}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    if args.categories:
        print(f"Categories: {', '.join(args.categories)}")
    print("-" * 50)

    # Check if Ollama is running
    if not check_ollama_connection(args.url):
        print(f"Error: Cannot connect to Ollama at {args.url}")
        print(f"Please ensure Ollama is running and accessible.")
        return

    print("Ollama API is accessible")

    # Run benchmark
    results = benchmark_mmlu_pro_all(
        model_name=args.model,
        ollama_url=args.url,
        categories=args.categories,
        split=args.split,
        max_samples_per_category=args.max_samples
    )

    # Save results
    output_path = save_results(results, args.output)

    # Print summary
    print_results_summary(results, show_subjects=True)
    print(f"\nResults saved to: {output_path.absolute()}")

    # Additional MMLU-Pro specific info
    if "dataset_stats" in results:
        print("\nDataset Statistics:")
        stats = results["dataset_stats"]
        print(f"Total samples evaluated: {stats['total_samples']}")


if __name__ == "__main__":
    main()
