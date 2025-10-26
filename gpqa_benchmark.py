#!/usr/bin/env python3
"""
GPQA (Graduate-Level Google-Proof Q&A) Benchmark Script for Ollama Models

GPQA tests expert-level knowledge with graduate-level science questions.
- 448 questions (main subset)
- Multiple choice with 4 options
- Expert-verified, challenging questions
"""

import argparse
import time
from typing import Dict, Tuple

from utils.ollama import check_ollama_connection, query_ollama
from utils.prompts import format_multiple_choice_prompt, extract_letter_answer
from utils.evaluation import evaluate_dataset, save_results, print_results_summary
from utils.gpqa import load_gpqa_dataset, parse_gpqa_sample, get_dataset_stats, format_gpqa_instruction


def create_gpqa_evaluator(model_name: str, ollama_url: str):
    """Create an evaluator function for GPQA samples"""
    def evaluate_gpqa_sample(sample: Dict) -> Tuple[bool, str, str]:
        question, choices, correct_letter = parse_gpqa_sample(sample)

        instruction = format_gpqa_instruction()

        prompt = format_multiple_choice_prompt(question, choices, instruction)
        response = query_ollama(prompt, model_name, ollama_url, temperature=0.0, max_tokens=10)
        predicted_letter = extract_letter_answer(response, ['A', 'B', 'C', 'D'])

        return predicted_letter == correct_letter, predicted_letter, correct_letter

    return evaluate_gpqa_sample


def benchmark_gpqa(model_name: str, ollama_url: str, subset: str = "gpqa_main",
                   split: str = "train", max_samples: int = None) -> Dict:
    """Benchmark GPQA"""
    print(f"\nEvaluating GPQA ({subset})")

    dataset = load_gpqa_dataset(split, subset)
    if dataset is None:
        return {"error": "Failed to load dataset"}

    stats = get_dataset_stats(dataset)
    print(f"Total questions: {stats['total_samples']}")

    evaluator_fn = create_gpqa_evaluator(model_name, ollama_url)
    start_time = time.time()

    results = evaluate_dataset(dataset, evaluator_fn, max_samples, delay=0.1,
                               description=f"Processing GPQA ({subset})")

    results.update({
        "model": model_name,
        "subset": subset,
        "elapsed_time_seconds": time.time() - start_time,
        "overall_accuracy": results["accuracy"],
        "total_correct": results["correct"],
        "total_questions": results["total"],
        "num_subjects": 1
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama models on GPQA")
    parser.add_argument("--model", type=str, default="gemma3:latest")
    parser.add_argument("--url", type=str, default="http://localhost:11434")
    parser.add_argument("--subset", type=str, default="gpqa_main",
                       choices=["gpqa_main", "gpqa_diamond", "gpqa_extended"])
    parser.add_argument("--split", type=str, default="train",
                       choices=["train"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="gpqa_results.json")

    args = parser.parse_args()

    print(f"Starting GPQA Benchmark")
    print(f"Model: {args.model}")
    print(f"Subset: {args.subset}")
    print("-" * 50)

    if not check_ollama_connection(args.url):
        print(f"Error: Cannot connect to Ollama at {args.url}")
        return

    results = benchmark_gpqa(args.model, args.url, args.subset, args.split, args.max_samples)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    output_path = save_results(results, args.output)
    print_results_summary(results, show_subjects=False)
    print(f"\nResults saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
