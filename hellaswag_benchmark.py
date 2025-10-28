#!/usr/bin/env python3
"""
HellaSwag Benchmark Script for Ollama Models

HellaSwag tests commonsense reasoning about everyday physical and social situations.
- ~70,000 questions
- Sentence completion format
- 4 answer choices
"""

import argparse
import time
from typing import Dict, Tuple

from utils.ollama import check_ollama_connection, query_ollama
from utils.evaluation import evaluate_dataset, save_results, print_results_summary
from utils.hellaswag import load_hellaswag_dataset, parse_hellaswag_sample
from utils.reasoning import ReasoningStrategy, DirectStrategy, create_strategy
from utils.cot_examples import get_cot_examples


def create_hellaswag_evaluator(model_name: str, ollama_url: str, strategy: ReasoningStrategy):
    """Create an evaluator function for HellaSwag samples with reasoning strategy"""
    def evaluate_hellaswag_sample(sample: Dict) -> Tuple[bool, str, str]:
        question, choices, correct_letter = parse_hellaswag_sample(sample)

        instruction = (
            "Complete the following scenario by selecting the most plausible continuation.\n"
            "Use your commonsense reasoning about everyday situations."
        )

        prompt = strategy.format_prompt(question, choices, instruction)

        # Use longer timeout for CoT reasoning
        timeout = 180 if strategy.get_max_tokens() > 100 else 60

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
            predicted_letter = strategy.extract_answer_from_multiple(responses, ['A', 'B', 'C', 'D'])
        else:
            response = query_ollama(
                prompt, model_name, ollama_url,
                temperature=strategy.get_temperature(),
                max_tokens=strategy.get_max_tokens(),
                timeout=timeout
            )
            predicted_letter = strategy.extract_answer(response, ['A', 'B', 'C', 'D'])

        return predicted_letter == correct_letter, predicted_letter, correct_letter

    return evaluate_hellaswag_sample


def benchmark_hellaswag(model_name: str, ollama_url: str, split: str = "validation",
                        max_samples: int = None, strategy: ReasoningStrategy = None) -> Dict:
    """Benchmark HellaSwag"""
    if strategy is None:
        strategy = DirectStrategy()

    print(f"\nEvaluating HellaSwag")
    print(f"Reasoning strategy: {strategy.__class__.__name__}")

    dataset = load_hellaswag_dataset(split)
    if dataset is None:
        return {"error": "Failed to load dataset"}

    print(f"Total questions: {len(dataset)}")

    evaluator_fn = create_hellaswag_evaluator(model_name, ollama_url, strategy)
    start_time = time.time()

    results = evaluate_dataset(dataset, evaluator_fn, max_samples, delay=0.1,
                               description="Processing HellaSwag")

    results.update({
        "model": model_name,
        "reasoning_strategy": strategy.__class__.__name__,
        "elapsed_time_seconds": time.time() - start_time,
        "overall_accuracy": results["accuracy"],
        "total_correct": results["correct"],
        "total_questions": results["total"],
        "num_subjects": 1
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama models on HellaSwag")
    parser.add_argument("--model", type=str, default="gemma3:latest")
    parser.add_argument("--url", type=str, default="http://localhost:11434")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "train"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="hellaswag_results.json")

    # Reasoning strategy arguments
    parser.add_argument("--reasoning", type=str, default="direct",
                       choices=["direct", "zero-shot-cot", "few-shot-cot", "self-consistency"],
                       help="Reasoning strategy to use")
    parser.add_argument("--cot-examples", type=int, default=3,
                       help="Number of examples for few-shot CoT")
    parser.add_argument("--sc-samples", type=int, default=5,
                       help="Number of samples for self-consistency")

    args = parser.parse_args()

    print(f"Starting HellaSwag Benchmark")
    print(f"Model: {args.model}")
    print(f"Reasoning: {args.reasoning}")
    print("-" * 50)

    if not check_ollama_connection(args.url):
        print(f"Error: Cannot connect to Ollama at {args.url}")
        return

    # Create reasoning strategy
    if args.reasoning == "few-shot-cot":
        examples = get_cot_examples('hellaswag', n=args.cot_examples)
        strategy = create_strategy('few-shot-cot', examples=examples)
    elif args.reasoning == "self-consistency":
        strategy = create_strategy('self-consistency', base_strategy='zero-shot-cot', n_samples=args.sc_samples)
    else:
        strategy = create_strategy(args.reasoning)

    results = benchmark_hellaswag(args.model, args.url, args.split, args.max_samples, strategy)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    output_path = save_results(results, args.output)
    print_results_summary(results, show_subjects=False)
    print(f"\nResults saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
