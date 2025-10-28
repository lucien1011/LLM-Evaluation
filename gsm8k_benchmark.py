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
from utils.reasoning import ReasoningStrategy, DirectStrategy, create_strategy
from utils.cot_examples import get_cot_examples


def create_gsm8k_evaluator(model_name: str, ollama_url: str, strategy: ReasoningStrategy):
    """Create an evaluator function for GSM8K samples with reasoning strategy"""
    def evaluate_gsm8k_sample(sample: Dict) -> Tuple[bool, str, str]:
        question, correct_answer = parse_gsm8k_sample(sample)

        instruction = format_gsm8k_instruction()

        # For math problems, we adapt the strategy formatting
        # GSM8K doesn't use multiple choice, so we need special handling
        if hasattr(strategy, 'cot_trigger'):
            # Zero-shot CoT: add trigger to instruction
            prompt = f"{instruction}\n\n{question}\n\n{strategy.cot_trigger}"
        elif hasattr(strategy, 'examples'):
            # Few-shot CoT: format with examples
            examples_text = get_cot_examples('gsm8k', n=len(strategy.examples))
            from utils.cot_examples import format_cot_example
            formatted_examples = [format_cot_example(ex, 'gsm8k') for ex in examples_text]
            examples_str = '\n\n---\n\n'.join(formatted_examples)
            prompt = f"{instruction}\n\n{examples_str}\n\n---\n\nQuestion: {question}"
        else:
            # Direct strategy
            prompt = f"{instruction}\n\n{question}"

        # Use longer timeout for CoT reasoning
        timeout = 180 if strategy.get_max_tokens() > 100 else 60

        # Handle strategies that require multiple samples
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

            # Extract numerical answers from all responses
            predicted_answers = [extract_numerical_answer(r) for r in responses]
            # Use majority voting
            from collections import Counter
            answer_counts = Counter(str(a) for a in predicted_answers if a is not None)
            if answer_counts:
                most_common_str = answer_counts.most_common(1)[0][0]
                try:
                    predicted_answer = float(most_common_str)
                except:
                    predicted_answer = None
            else:
                predicted_answer = None
        else:
            # Single response
            response = query_ollama(
                prompt, model_name, ollama_url,
                temperature=strategy.get_temperature(),
                max_tokens=strategy.get_max_tokens(),
                timeout=timeout
            )
            predicted_answer = extract_numerical_answer(response)

        # Compare numerical answers
        is_correct = compare_numerical_answers(predicted_answer, correct_answer)

        # Return string representations for logging
        predicted_str = str(predicted_answer) if predicted_answer is not None else "NO_ANSWER"
        correct_str = str(correct_answer)

        return is_correct, predicted_str, correct_str

    return evaluate_gsm8k_sample


def benchmark_gsm8k(model_name: str, ollama_url: str, split: str = "test",
                    max_samples: int = None, strategy: ReasoningStrategy = None) -> Dict:
    """Benchmark GSM8K"""
    if strategy is None:
        strategy = DirectStrategy()

    print(f"\nEvaluating GSM8K")

    dataset = load_gsm8k_dataset(split)
    if dataset is None:
        return {"error": "Failed to load dataset"}

    stats = get_dataset_stats(dataset)
    print(f"Total questions: {stats['total_samples']}")

    evaluator_fn = create_gsm8k_evaluator(model_name, ollama_url, strategy)
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

    print(f"Starting GSM8K Benchmark")
    print(f"Model: {args.model}")
    print(f"Reasoning strategy: {args.reasoning}")
    print("-" * 50)

    if not check_ollama_connection(args.url):
        print(f"Error: Cannot connect to Ollama at {args.url}")
        return

    # Create reasoning strategy
    if args.reasoning == "few-shot-cot":
        examples = get_cot_examples('gsm8k', n=args.cot_examples)
        strategy = create_strategy('few-shot-cot', examples=examples)
    elif args.reasoning == "self-consistency":
        strategy = create_strategy('self-consistency', base_strategy='zero-shot-cot', n_samples=args.sc_samples)
    else:
        strategy = create_strategy(args.reasoning)

    print(f"Using reasoning strategy: {strategy.__class__.__name__}")

    results = benchmark_gsm8k(args.model, args.url, args.split, args.max_samples, strategy)

    # Add reasoning strategy to results
    results['reasoning_strategy'] = args.reasoning

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    output_path = save_results(results, args.output)
    print_results_summary(results, show_subjects=False)
    print(f"\nResults saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
