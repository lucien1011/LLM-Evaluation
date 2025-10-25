#!/usr/bin/env python3
"""
MMLU Benchmark Script for Ollama Models (Modular Version)
Evaluates language models on the Massive Multitask Language Understanding benchmark
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
    calculate_overall_stats,
    save_results,
    print_results_summary
)
from utils.mmlu import (
    load_mmlu_subject,
    parse_mmlu_sample,
    DEFAULT_MMLU_SUBJECTS
)


def create_mmlu_evaluator(model_name: str, ollama_url: str):
    """
    Create an evaluator function for MMLU samples

    Args:
        model_name: Name of the Ollama model
        ollama_url: URL of the Ollama API

    Returns:
        Function that evaluates a single MMLU sample
    """
    def evaluate_mmlu_sample(sample: Dict) -> Tuple[bool, str, str]:
        """Evaluate a single MMLU sample"""
        # Parse the MMLU sample
        question, choices, correct_letter = parse_mmlu_sample(sample)

        # Format the prompt
        prompt = format_multiple_choice_prompt(question, choices)

        # Query the model
        response = query_ollama(
            prompt=prompt,
            model_name=model_name,
            ollama_url=ollama_url,
            temperature=0.0,
            max_tokens=10
        )

        # Extract the predicted answer
        predicted_letter = extract_letter_answer(response)

        # Check if correct
        is_correct = predicted_letter == correct_letter

        return is_correct, predicted_letter, correct_letter

    return evaluate_mmlu_sample


def benchmark_mmlu_subject(
    subject: str,
    model_name: str,
    ollama_url: str,
    split: str = "test",
    max_samples: int = None
) -> Dict:
    """
    Benchmark a specific MMLU subject

    Args:
        subject: MMLU subject name
        model_name: Name of the Ollama model
        ollama_url: URL of the Ollama API
        split: Dataset split ('test', 'validation', 'dev')
        max_samples: Maximum number of samples to evaluate (None for all)

    Returns:
        Dictionary with evaluation results
    """
    print(f"\nEvaluating subject: {subject}")

    # Load the dataset
    dataset = load_mmlu_subject(subject, split)
    if dataset is None:
        return {"subject": subject, "error": "Failed to load dataset"}

    # Create evaluator function
    evaluator_fn = create_mmlu_evaluator(model_name, ollama_url)

    # Evaluate the dataset
    results = evaluate_dataset(
        dataset=dataset,
        evaluator_fn=evaluator_fn,
        max_samples=max_samples,
        delay=0.1,
        description=f"Processing {subject}"
    )

    # Add subject name to results
    results["subject"] = subject

    return results


def benchmark_mmlu_all(
    model_name: str,
    ollama_url: str,
    subjects: List[str] = None,
    split: str = "test",
    max_samples_per_subject: int = None
) -> Dict:
    """
    Benchmark multiple MMLU subjects

    Args:
        model_name: Name of the Ollama model
        ollama_url: URL of the Ollama API
        subjects: List of subjects to benchmark (None for default set)
        split: Dataset split to use
        max_samples_per_subject: Max samples per subject (None for all)

    Returns:
        Dictionary with overall results
    """
    if subjects is None:
        subjects = DEFAULT_MMLU_SUBJECTS

    start_time = time.time()
    subject_results = []

    for subject in subjects:
        result = benchmark_mmlu_subject(
            subject=subject,
            model_name=model_name,
            ollama_url=ollama_url,
            split=split,
            max_samples=max_samples_per_subject
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
    parser = argparse.ArgumentParser(description="Benchmark Ollama models on MMLU")
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
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help="Specific subjects to benchmark (default: common set)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per subject (default: all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation", "dev"],
        help="Dataset split to use (default: test)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mmlu_results.json",
        help="Output file for results (default: mmlu_results.json)"
    )

    args = parser.parse_args()

    print(f"Starting MMLU Benchmark")
    print(f"Model: {args.model}")
    print(f"Ollama URL: {args.url}")
    print(f"Split: {args.split}")
    if args.max_samples:
        print(f"Max samples per subject: {args.max_samples}")
    print("-" * 50)

    # Check if Ollama is running
    if not check_ollama_connection(args.url):
        print(f"Error: Cannot connect to Ollama at {args.url}")
        print(f"Please ensure Ollama is running and accessible.")
        return

    print("Ollama API is accessible")

    # Run benchmark
    results = benchmark_mmlu_all(
        model_name=args.model,
        ollama_url=args.url,
        subjects=args.subjects,
        split=args.split,
        max_samples_per_subject=args.max_samples
    )

    # Save results
    output_path = save_results(results, args.output)

    # Print summary
    print_results_summary(results, show_subjects=True)
    print(f"\nResults saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
