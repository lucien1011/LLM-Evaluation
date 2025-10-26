#!/usr/bin/env python3
"""
TruthfulQA Benchmark Script for Ollama Models

TruthfulQA measures whether language models are truthful when answering questions.
- 817 questions across 38 categories
- Tests if models avoid false beliefs and misconceptions
- Two formats: MC1 (single correct) and MC2 (multiple correct)

Evaluates language models on the TruthfulQA benchmark using Ollama
"""

import argparse
import time
from typing import Dict, List, Tuple
from pathlib import Path

# Import our modular utilities
from utils.ollama import check_ollama_connection, query_ollama
from utils.prompts import format_multiple_choice_prompt, extract_letter_answer
from utils.evaluation import (
    evaluate_dataset,
    save_results,
    print_results_summary
)
from utils.truthfulqa import (
    load_truthfulqa_dataset,
    parse_truthfulqa_mc1,
    parse_truthfulqa_mc2,
    get_sample_category,
    filter_by_category,
    get_dataset_stats,
    format_truthfulqa_instruction,
    TRUTHFULQA_CATEGORIES
)


def create_truthfulqa_mc1_evaluator(model_name: str, ollama_url: str):
    """
    Create an evaluator function for TruthfulQA MC1 (single correct answer)

    Args:
        model_name: Name of the Ollama model
        ollama_url: URL of the Ollama API

    Returns:
        Function that evaluates a single TruthfulQA MC1 sample
    """
    def evaluate_truthfulqa_mc1_sample(sample: Dict) -> Tuple[bool, str, str]:
        """Evaluate a single TruthfulQA MC1 sample"""
        # Parse the sample
        question, choices, correct_letter = parse_truthfulqa_mc1(sample)

        # Get custom instruction for truthfulness
        instruction = format_truthfulqa_instruction("mc1")

        # Format the prompt
        prompt = format_multiple_choice_prompt(question, choices, instruction)

        # Query the model
        response = query_ollama(
            prompt=prompt,
            model_name=model_name,
            ollama_url=ollama_url,
            temperature=0.0,
            max_tokens=10
        )

        # Extract the predicted answer
        # Determine valid choices based on number of options
        num_choices = len(choices)
        valid_choices = [chr(65 + i) for i in range(num_choices)]  # A, B, C, D, ...

        predicted_letter = extract_letter_answer(response, valid_choices)

        # Check if correct
        is_correct = predicted_letter == correct_letter

        return is_correct, predicted_letter, correct_letter

    return evaluate_truthfulqa_mc1_sample


def create_truthfulqa_mc2_evaluator(model_name: str, ollama_url: str):
    """
    Create an evaluator function for TruthfulQA MC2 (multiple correct answers)

    Args:
        model_name: Name of the Ollama model
        ollama_url: URL of the Ollama API

    Returns:
        Function that evaluates a single TruthfulQA MC2 sample
    """
    def evaluate_truthfulqa_mc2_sample(sample: Dict) -> Tuple[bool, str, str]:
        """Evaluate a single TruthfulQA MC2 sample"""
        # Parse the sample
        question, choices, correct_letters = parse_truthfulqa_mc2(sample)

        # Get custom instruction
        instruction = format_truthfulqa_instruction("mc2")

        # Format the prompt
        prompt = format_multiple_choice_prompt(question, choices, instruction)

        # Query the model
        response = query_ollama(
            prompt=prompt,
            model_name=model_name,
            ollama_url=ollama_url,
            temperature=0.0,
            max_tokens=20  # More tokens for multiple answers
        )

        # Extract predicted answers (could be multiple)
        num_choices = len(choices)
        valid_choices = [chr(65 + i) for i in range(num_choices)]

        # Extract all letter answers from response
        predicted_letters = []
        for char in response.upper():
            if char in valid_choices and char not in predicted_letters:
                predicted_letters.append(char)

        # For MC2, we consider it correct if they got at least one correct answer
        # and didn't select any incorrect ones (strict evaluation)
        predicted_set = set(predicted_letters)
        correct_set = set(correct_letters)

        is_correct = len(predicted_set) > 0 and predicted_set.issubset(correct_set)

        predicted_str = ','.join(predicted_letters) if predicted_letters else ''
        correct_str = ','.join(correct_letters)

        return is_correct, predicted_str, correct_str

    return evaluate_truthfulqa_mc2_sample


def benchmark_truthfulqa(
    model_name: str,
    ollama_url: str,
    format_type: str = "mc1",
    categories: List[str] = None,
    max_samples: int = None
) -> Dict:
    """
    Benchmark TruthfulQA

    Args:
        model_name: Name of the Ollama model
        ollama_url: URL of the Ollama API
        format_type: 'mc1' (single correct) or 'mc2' (multiple correct)
        categories: List of categories to filter (None for all)
        max_samples: Maximum number of samples to evaluate (None for all)

    Returns:
        Dictionary with evaluation results
    """
    print(f"\nEvaluating TruthfulQA ({format_type.upper()} format)")

    # Load the dataset
    dataset = load_truthfulqa_dataset()
    if dataset is None:
        return {"error": "Failed to load dataset"}

    # Filter by categories if specified
    if categories:
        print(f"Filtering by categories: {', '.join(categories)}")
        filtered_samples = []
        for sample in dataset:
            if sample.get('category') in categories:
                filtered_samples.append(sample)
        dataset = filtered_samples
        print(f"Filtered to {len(dataset)} samples")

    if len(dataset) == 0:
        return {"error": "No samples found after filtering"}

    # Show dataset statistics
    if not categories:
        stats = get_dataset_stats(dataset)
        print(f"Total questions: {stats['total_samples']}")
        print(f"Categories: {stats['num_categories']}")

    # Create evaluator based on format type
    if format_type == "mc1":
        evaluator_fn = create_truthfulqa_mc1_evaluator(model_name, ollama_url)
    else:
        evaluator_fn = create_truthfulqa_mc2_evaluator(model_name, ollama_url)

    # Evaluate
    start_time = time.time()

    results = evaluate_dataset(
        dataset=dataset,
        evaluator_fn=evaluator_fn,
        max_samples=max_samples,
        delay=0.1,
        description=f"Processing TruthfulQA ({format_type})"
    )

    elapsed_time = time.time() - start_time

    # Add metadata
    results["model"] = model_name
    results["format"] = format_type
    results["elapsed_time_seconds"] = elapsed_time
    results["overall_accuracy"] = results["accuracy"]
    results["total_correct"] = results["correct"]
    results["total_questions"] = results["total"]
    results["num_subjects"] = 1  # For compatibility with visualization

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama models on TruthfulQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with MC1 format (single correct answer)
  python truthfulqa_benchmark.py --model gemma3:latest --max-samples 10

  # Full benchmark with MC1
  python truthfulqa_benchmark.py --model gemma3:latest

  # Test MC2 format (multiple correct answers)
  python truthfulqa_benchmark.py --model gemma3:latest --format mc2 --max-samples 10

  # Filter by specific categories
  python truthfulqa_benchmark.py --model gemma3:latest --categories Health Law

  # Custom output
  python truthfulqa_benchmark.py --model gemma3:latest --output my_results.json
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
        "--format",
        type=str,
        default="mc1",
        choices=["mc1", "mc2"],
        help="Question format: mc1 (single correct) or mc2 (multiple correct) (default: mc1)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Specific categories to test (default: all)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (default: all 817)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="truthfulqa_results.json",
        help="Output file for results (default: truthfulqa_results.json)"
    )

    args = parser.parse_args()

    print(f"Starting TruthfulQA Benchmark")
    print(f"Model: {args.model}")
    print(f"Format: {args.format.upper()}")
    print(f"Ollama URL: {args.url}")
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
    results = benchmark_truthfulqa(
        model_name=args.model,
        ollama_url=args.url,
        format_type=args.format,
        categories=args.categories,
        max_samples=args.max_samples
    )

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    # Save results
    output_path = save_results(results, args.output)

    # Print summary
    print("\n" + "=" * 50)
    print("TRUTHFULQA RESULTS")
    print("=" * 50)
    print(f"Model: {results['model']}")
    print(f"Format: {results['format'].upper()}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Correct: {results['total_correct']}/{results['total_questions']}")
    print(f"Time Elapsed: {results['elapsed_time_seconds']:.2f} seconds")
    print(f"\nResults saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
