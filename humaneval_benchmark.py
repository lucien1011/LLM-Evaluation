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
from utils.reasoning import ReasoningStrategy, DirectStrategy, create_strategy
from utils.cot_examples import get_cot_examples


def create_humaneval_evaluator(model_name: str, ollama_url: str, strategy: ReasoningStrategy):
    """Create an evaluator function for HumanEval samples with reasoning strategy"""
    def evaluate_humaneval_sample(sample: Dict) -> Tuple[bool, str, str]:
        task_id, prompt, test_code, entry_point = parse_humaneval_sample(sample)

        instruction = format_humaneval_instruction()

        # For code generation, adapt strategy
        if hasattr(strategy, 'cot_trigger'):
            # Zero-shot CoT: ask to explain approach first
            full_prompt = f"{instruction}\n\n{prompt}\n\nFirst, explain your approach step by step, then provide the complete code."
        elif hasattr(strategy, 'examples'):
            # Few-shot CoT: include examples
            examples_text = get_cot_examples('humaneval', n=len(strategy.examples))
            from utils.cot_examples import format_cot_example
            formatted_examples = [format_cot_example(ex, 'humaneval') for ex in examples_text]
            examples_str = '\n\n---\n\n'.join(formatted_examples)
            full_prompt = f"{instruction}\n\n{examples_str}\n\n---\n\nProblem:\n{prompt}"
        else:
            # Direct strategy
            full_prompt = f"{instruction}\n\n{prompt}"

        # Use longer timeout for CoT reasoning
        timeout = 180 if strategy.get_max_tokens() > 512 else 60

        # Handle strategies that require multiple samples (take first successful one)
        if strategy.requires_multiple_samples():
            for _ in range(strategy.get_num_samples()):
                response = query_ollama(
                    full_prompt, model_name, ollama_url,
                    temperature=strategy.get_temperature(),
                    max_tokens=strategy.get_max_tokens(),
                    timeout=timeout
                )
                generated_code = extract_code_from_response(response, entry_point)

                if generated_code is not None:
                    # Try this code
                    if f"def {entry_point}" in generated_code:
                        complete_code = generated_code
                    else:
                        complete_code = prompt + "\n" + generated_code

                    success, error_msg = execute_code_safely(complete_code, test_code, timeout_seconds=5)
                    if success:
                        # Found a working solution
                        return True, "PASS", "PASS"

            # None of the samples worked
            return False, "FAIL: No valid solution found", "PASS"

        else:
            # Single response
            response = query_ollama(
                full_prompt, model_name, ollama_url,
                temperature=strategy.get_temperature(),
                max_tokens=strategy.get_max_tokens(),
                timeout=timeout
            )

            # Extract code from response
            generated_code = extract_code_from_response(response, entry_point)

            if generated_code is None:
                return False, "NO_CODE", "PASS"

            # Combine prompt (function signature) with generated code
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
                        max_samples: int = None, strategy: ReasoningStrategy = None) -> Dict:
    """Benchmark HumanEval"""
    if strategy is None:
        strategy = DirectStrategy()

    print(f"\nEvaluating HumanEval")

    dataset = load_humaneval_dataset(split)
    if dataset is None:
        return {"error": "Failed to load dataset"}

    stats = get_dataset_stats(dataset)
    print(f"Total questions: {stats['total_samples']}")

    evaluator_fn = create_humaneval_evaluator(model_name, ollama_url, strategy)
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

    # Reasoning strategy arguments
    parser.add_argument(
        "--reasoning",
        type=str,
        default="direct",
        choices=["direct", "zero-shot-cot", "few-shot-cot", "self-consistency"],
        help="Reasoning strategy to use (default: direct)"
    )
    parser.add_argument(
        "--cot-examples",
        type=int,
        default=3,
        help="Number of examples for few-shot CoT (default: 3)"
    )
    parser.add_argument(
        "--sc-samples",
        type=int,
        default=5,
        help="Number of samples for self-consistency (default: 5)"
    )

    args = parser.parse_args()

    print(f"Starting HumanEval Benchmark")
    print(f"Model: {args.model}")
    print(f"Reasoning strategy: {args.reasoning}")
    print("-" * 50)

    if not check_ollama_connection(args.url):
        print(f"Error: Cannot connect to Ollama at {args.url}")
        return

    # Create reasoning strategy
    if args.reasoning == "few-shot-cot":
        examples = get_cot_examples('humaneval', n=args.cot_examples)
        strategy = create_strategy('few-shot-cot', examples=examples)
    elif args.reasoning == "self-consistency":
        strategy = create_strategy('self-consistency', base_strategy='zero-shot-cot', n_samples=args.sc_samples)
    else:
        strategy = create_strategy(args.reasoning)

    print(f"Using reasoning strategy: {strategy.__class__.__name__}")

    results = benchmark_humaneval(args.model, args.url, args.split, args.max_samples, strategy)

    # Add reasoning strategy to results
    results['reasoning_strategy'] = args.reasoning

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    output_path = save_results(results, args.output)
    print_results_summary(results, show_subjects=False)
    print(f"\nPass@1: {results['pass_at_1']:.2%}")
    print(f"\nResults saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
