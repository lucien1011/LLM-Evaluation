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
from utils.prompts import format_multiple_choice_prompt, extract_letter_answer
from utils.evaluation import evaluate_dataset, save_results, print_results_summary
from utils.hellaswag import load_hellaswag_dataset, parse_hellaswag_sample


def create_hellaswag_evaluator(model_name: str, ollama_url: str):
    """Create an evaluator function for HellaSwag samples"""
    def evaluate_hellaswag_sample(sample: Dict) -> Tuple[bool, str, str]:
        question, choices, correct_letter = parse_hellaswag_sample(sample)

        instruction = (
            "Complete the following scenario by selecting the most plausible continuation.\n"
            "Use your commonsense reasoning about everyday situations.\n"
            "Only respond with the letter of your answer (A, B, C, or D), nothing else."
        )

        prompt = format_multiple_choice_prompt(question, choices, instruction)
        response = query_ollama(prompt, model_name, ollama_url, temperature=0.0, max_tokens=10)
        predicted_letter = extract_letter_answer(response, ['A', 'B', 'C', 'D'])

        return predicted_letter == correct_letter, predicted_letter, correct_letter

    return evaluate_hellaswag_sample


def benchmark_hellaswag(model_name: str, ollama_url: str, split: str = "validation",
                        max_samples: int = None) -> Dict:
    """Benchmark HellaSwag"""
    print(f"\nEvaluating HellaSwag")

    dataset = load_hellaswag_dataset(split)
    if dataset is None:
        return {"error": "Failed to load dataset"}

    print(f"Total questions: {len(dataset)}")

    evaluator_fn = create_hellaswag_evaluator(model_name, ollama_url)
    start_time = time.time()

    results = evaluate_dataset(dataset, evaluator_fn, max_samples, delay=0.1,
                               description="Processing HellaSwag")

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
    parser = argparse.ArgumentParser(description="Benchmark Ollama models on HellaSwag")
    parser.add_argument("--model", type=str, default="gemma3:latest")
    parser.add_argument("--url", type=str, default="http://localhost:11434")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "train"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="hellaswag_results.json")

    args = parser.parse_args()

    print(f"Starting HellaSwag Benchmark")
    print(f"Model: {args.model}")
    print("-" * 50)

    if not check_ollama_connection(args.url):
        print(f"Error: Cannot connect to Ollama at {args.url}")
        return

    results = benchmark_hellaswag(args.model, args.url, args.split, args.max_samples)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    output_path = save_results(results, args.output)
    print_results_summary(results, show_subjects=False)
    print(f"\nResults saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
