#!/usr/bin/env python3
"""
GSM8K (Grade School Math 8K) Benchmark Script for Ollama Models

GSM8K tests mathematical reasoning with grade school math word problems.
- 8,500 questions (7,500 train, 1,000 test)
- Free-form numerical answers
- Multi-step reasoning required
"""

import argparse
import time
from typing import Dict, Tuple

from utils.ollama import check_ollama_connection, query_ollama
from utils.evaluation import evaluate_dataset, save_results, print_results_summary
from utils.gsm8k import (
    load_gsm8k_dataset,
    parse_gsm8k_sample,
    extract_numerical_answer,
    compare_numerical_answers,
    get_dataset_stats,
    format_gsm8k_instruction
)


def create_gsm8k_evaluator(model_name: str, ollama_url: str):
    """Create an evaluator function for GSM8K samples"""
    def evaluate_gsm8k_sample(sample: Dict) -> Tuple[bool, str, str]:
        question, correct_answer = parse_gsm8k_sample(sample)

        instruction = format_gsm8k_instruction()
        prompt = f"{instruction}\n\n{question}"

        # Allow more tokens for math reasoning
        response = query_ollama(prompt, model_name, ollama_url, temperature=0.0, max_tokens=512)

        predicted_answer = extract_numerical_answer(response)

        # Compare numerical answers
        is_correct = compare_numerical_answers(predicted_answer, correct_answer)

        # Return string representations for logging
        predicted_str = str(predicted_answer) if predicted_answer is not None else "NO_ANSWER"
        correct_str = str(correct_answer)

        return is_correct, predicted_str, correct_str

    return evaluate_gsm8k_sample


def benchmark_gsm8k(model_name: str, ollama_url: str, split: str = "test",
                    max_samples: int = None) -> Dict:
    """Benchmark GSM8K"""
    print(f"\nEvaluating GSM8K")

    dataset = load_gsm8k_dataset(split)
    if dataset is None:
        return {"error": "Failed to load dataset"}

    stats = get_dataset_stats(dataset)
    print(f"Total questions: {stats['total_samples']}")

    evaluator_fn = create_gsm8k_evaluator(model_name, ollama_url)
    start_time = time.time()

    results = evaluate_dataset(dataset, evaluator_fn, max_samples, delay=0.1,
                               description="Processing GSM8K")

    results.update({
        "model": model_name,
        "elapsed_time_seconds": time.time() - start_time,
        "overall_accuracy": results["accuracy"],
        "total_correct": results["correct"],
        "total_questions": results["total"],
        "num_subjects": 1
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama models on GSM8K")
    parser.add_argument("--model", type=str, default="gemma3:latest")
    parser.add_argument("--url", type=str, default="http://localhost:11434")
    parser.add_argument("--split", type=str, default="test",
                       choices=["test", "train"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="gsm8k_results.json")

    args = parser.parse_args()

    print(f"Starting GSM8K Benchmark")
    print(f"Model: {args.model}")
    print("-" * 50)

    if not check_ollama_connection(args.url):
        print(f"Error: Cannot connect to Ollama at {args.url}")
        return

    results = benchmark_gsm8k(args.model, args.url, args.split, args.max_samples)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    output_path = save_results(results, args.output)
    print_results_summary(results, show_subjects=False)
    print(f"\nResults saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
