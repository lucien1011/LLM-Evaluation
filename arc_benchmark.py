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
from utils.prompts import format_multiple_choice_prompt, extract_letter_answer
from utils.evaluation import evaluate_dataset, save_results, print_results_summary
from utils.arc import load_arc_dataset, parse_arc_sample, get_dataset_stats


def create_arc_evaluator(model_name: str, ollama_url: str):
    """Create an evaluator function for ARC samples"""
    def evaluate_arc_sample(sample: Dict) -> Tuple[bool, str, str]:
        question, choices, correct_letter = parse_arc_sample(sample)

        instruction = (
            "Answer the following science question by selecting the correct option.\n"
            "Use scientific reasoning and common sense.\n"
            "Only respond with the letter of your answer (A, B, C, or D), nothing else."
        )

        prompt = format_multiple_choice_prompt(question, choices, instruction)
        response = query_ollama(prompt, model_name, ollama_url, temperature=0.0, max_tokens=10)
        predicted_letter = extract_letter_answer(response, ['A', 'B', 'C', 'D'])

        return predicted_letter == correct_letter, predicted_letter, correct_letter

    return evaluate_arc_sample


def benchmark_arc(model_name: str, ollama_url: str, difficulty: str = "ARC-Challenge",
                  split: str = "test", max_samples: int = None) -> Dict:
    """Benchmark ARC"""
    print(f"\nEvaluating ARC ({difficulty})")

    dataset = load_arc_dataset(split, difficulty)
    if dataset is None:
        return {"error": "Failed to load dataset"}

    stats = get_dataset_stats(dataset)
    print(f"Total questions: {stats['total_samples']}")

    evaluator_fn = create_arc_evaluator(model_name, ollama_url)
    start_time = time.time()

    results = evaluate_dataset(dataset, evaluator_fn, max_samples, delay=0.1,
                               description=f"Processing ARC ({difficulty})")

    results.update({
        "model": model_name,
        "difficulty": difficulty,
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

    args = parser.parse_args()

    print(f"Starting ARC Benchmark")
    print(f"Model: {args.model}")
    print(f"Difficulty: {args.difficulty}")
    print("-" * 50)

    if not check_ollama_connection(args.url):
        print(f"Error: Cannot connect to Ollama at {args.url}")
        return

    results = benchmark_arc(args.model, args.url, args.difficulty, args.split, args.max_samples)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    output_path = save_results(results, args.output)
    print_results_summary(results, show_subjects=False)
    print(f"\nResults saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
