#!/usr/bin/env python3
"""
HumanEval Benchmark Script for Ollama Models

HumanEval tests code generation capabilities with Python programming problems.
- 164 programming problems
- Function completion tasks
- Automatic test case evaluation
"""

import argparse
import time
from typing import Dict, Tuple

from utils.ollama import check_ollama_connection, query_ollama
from utils.evaluation import evaluate_dataset, save_results, print_results_summary
from utils.humaneval import (
    load_humaneval_dataset,
    parse_humaneval_sample,
    extract_code_from_response,
    execute_code_safely,
    get_dataset_stats,
    format_humaneval_instruction
)


def create_humaneval_evaluator(model_name: str, ollama_url: str):
    """Create an evaluator function for HumanEval samples"""
    def evaluate_humaneval_sample(sample: Dict) -> Tuple[bool, str, str]:
        task_id, prompt, test_code, entry_point = parse_humaneval_sample(sample)

        instruction = format_humaneval_instruction()
        full_prompt = f"{instruction}\n\n{prompt}"

        # Allow more tokens for code generation
        response = query_ollama(full_prompt, model_name, ollama_url, temperature=0.0, max_tokens=1024)

        # Extract code from response
        generated_code = extract_code_from_response(response, entry_point)

        if generated_code is None:
            return False, "NO_CODE", "PASS"

        # Combine prompt (function signature) with generated code
        # If generated code includes the signature, use it as is
        # Otherwise, combine them
        if f"def {entry_point}" in generated_code:
            complete_code = generated_code
        else:
            complete_code = prompt + "\n" + generated_code

        # Execute code with test cases
        success, error_msg = execute_code_safely(complete_code, test_code, timeout_seconds=5)

        # Return results
        predicted = "PASS" if success else f"FAIL: {error_msg}"
        correct = "PASS"

        return success, predicted, correct

    return evaluate_humaneval_sample


def benchmark_humaneval(model_name: str, ollama_url: str, split: str = "test",
                        max_samples: int = None) -> Dict:
    """Benchmark HumanEval"""
    print(f"\nEvaluating HumanEval")

    dataset = load_humaneval_dataset(split)
    if dataset is None:
        return {"error": "Failed to load dataset"}

    stats = get_dataset_stats(dataset)
    print(f"Total questions: {stats['total_samples']}")

    evaluator_fn = create_humaneval_evaluator(model_name, ollama_url)
    start_time = time.time()

    results = evaluate_dataset(dataset, evaluator_fn, max_samples, delay=0.1,
                               description="Processing HumanEval")

    results.update({
        "model": model_name,
        "elapsed_time_seconds": time.time() - start_time,
        "overall_accuracy": results["accuracy"],
        "total_correct": results["correct"],
        "total_questions": results["total"],
        "num_subjects": 1,
        "pass_at_1": results["accuracy"]  # Pass@1 metric
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama models on HumanEval")
    parser.add_argument("--model", type=str, default="gemma3:latest")
    parser.add_argument("--url", type=str, default="http://localhost:11434")
    parser.add_argument("--split", type=str, default="test",
                       choices=["test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="humaneval_results.json")

    args = parser.parse_args()

    print(f"Starting HumanEval Benchmark")
    print(f"Model: {args.model}")
    print("-" * 50)

    if not check_ollama_connection(args.url):
        print(f"Error: Cannot connect to Ollama at {args.url}")
        return

    results = benchmark_humaneval(args.model, args.url, args.split, args.max_samples)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    output_path = save_results(results, args.output)
    print_results_summary(results, show_subjects=False)
    print(f"\nPass@1: {results['pass_at_1']:.2%}")
    print(f"\nResults saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
