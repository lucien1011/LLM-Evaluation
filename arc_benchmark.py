#!/usr/bin/env python3
"""
ARC (AI2 Reasoning Challenge) Benchmark Script for Ollama Models

ARC tests scientific reasoning and common sense with grade-school science questions.
- ~7,800 questions
- Two difficulty levels: Easy and Challenge
- 4 answer choices (A-D)
"""

import argparse
import time
from typing import Dict, Tuple
from pathlib import Path

from utils.ollama import check_ollama_connection, query_ollama
from utils.evaluation import evaluate_dataset, save_results, print_results_summary
from utils.arc import load_arc_dataset, parse_arc_sample, get_dataset_stats
from utils.reasoning import ReasoningStrategy, DirectStrategy, create_strategy
from utils.cot_examples import get_cot_examples


def create_arc_evaluator(model_name: str, ollama_url: str, strategy: ReasoningStrategy):
    """Create an evaluator function for ARC samples with reasoning strategy"""
    def evaluate_arc_sample(sample: Dict) -> Tuple[bool, str, str]:
        question, choices, correct_letter = parse_arc_sample(sample)

        # Use strategy to format prompt with default instruction
        # (includes \boxed{} notation for proper answer extraction)
        prompt = strategy.format_prompt(question, choices)

        # Use longer timeout for CoT reasoning
        timeout = 180 if strategy.get_max_tokens() > 100 else 60

        # Handle self-consistency (multiple samples)
        if strategy.requires_multiple_samples():
            responses = []
            for _ in range(strategy.get_num_samples()):
                response = query_ollama(
                    prompt, model_name, ollama_url,
                    temperature=strategy.get_temperature(),
                    max_tokens=strategy.get_max_tokens(),
                    timeout=timeout
                )
                responses.append(response)
            # Extract answer via voting
            predicted_letter = strategy.extract_answer_from_multiple(responses, ['A', 'B', 'C', 'D'])
        else:
            # Single sample
            response = query_ollama(
                prompt, model_name, ollama_url,
                temperature=strategy.get_temperature(),
                max_tokens=strategy.get_max_tokens(),
                timeout=timeout
            )
            predicted_letter = strategy.extract_answer(response, ['A', 'B', 'C', 'D'])

        return predicted_letter == correct_letter, predicted_letter, correct_letter

    return evaluate_arc_sample


def benchmark_arc(model_name: str, ollama_url: str, difficulty: str = "ARC-Challenge",
                  split: str = "test", max_samples: int = None, strategy: ReasoningStrategy = None) -> Dict:
    """Benchmark ARC"""
    if strategy is None:
        strategy = DirectStrategy()

    print(f"\nEvaluating ARC ({difficulty})")
    print(f"Reasoning strategy: {strategy.__class__.__name__}")

    dataset = load_arc_dataset(split, difficulty)
    if dataset is None:
        return {"error": "Failed to load dataset"}

    stats = get_dataset_stats(dataset)
    print(f"Total questions: {stats['total_samples']}")

    evaluator_fn = create_arc_evaluator(model_name, ollama_url, strategy)
    start_time = time.time()

    results = evaluate_dataset(dataset, evaluator_fn, max_samples, delay=0.1,
                               description=f"Processing ARC ({difficulty})")

    results.update({
        "model": model_name,
        "difficulty": difficulty,
        "reasoning_strategy": strategy.__class__.__name__,
        "elapsed_time_seconds": time.time() - start_time,
        "overall_accuracy": results["accuracy"],
        "total_correct": results["correct"],
        "total_questions": results["total"],
        "num_subjects": 1
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama models on ARC")
    parser.add_argument("--model", type=str, default="gemma3:latest")
    parser.add_argument("--url", type=str, default="http://localhost:11434")
    parser.add_argument("--difficulty", type=str, default="ARC-Challenge",
                       choices=["ARC-Easy", "ARC-Challenge"])
    parser.add_argument("--split", type=str, default="test",
                       choices=["test", "validation", "train"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="arc_results.json")

    # Reasoning strategy arguments
    parser.add_argument("--reasoning", type=str, default="direct",
                       choices=["direct", "zero-shot-cot", "few-shot-cot", "self-consistency"],
                       help="Reasoning strategy to use")
    parser.add_argument("--temperature", type=float, default=0.5,
                       help="Temperature for model generation")
    parser.add_argument("--cot-examples", type=int, default=3,
                       help="Number of examples for few-shot CoT")
    parser.add_argument("--sc-samples", type=int, default=5,
                       help="Number of samples for self-consistency")

    args = parser.parse_args()

    print(f"Starting ARC Benchmark")
    print(f"Model: {args.model}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Reasoning: {args.reasoning}")
    print("-" * 50)

    if not check_ollama_connection(args.url):
        print(f"Error: Cannot connect to Ollama at {args.url}")
        return

    # Create reasoning strategy
    if args.reasoning == "few-shot-cot":
        examples = get_cot_examples('arc', n=args.cot_examples)
        strategy = create_strategy('few-shot-cot', temperature=args.temperature, examples=examples)
    elif args.reasoning == "self-consistency":
        strategy = create_strategy('self-consistency', base_strategy='zero-shot-cot', temperature=args.temperature, n_samples=args.sc_samples)
    else:
        strategy = create_strategy(args.reasoning, temperature=args.temperature, )

    results = benchmark_arc(args.model, args.url, args.difficulty, args.split, args.max_samples, strategy)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    output_path = save_results(results, args.output)
    print_results_summary(results, show_subjects=False)
    print(f"\nResults saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
