#!/usr/bin/env python3
"""
Debug Model Comparison Script

Compare how different models respond to the same question using different reasoning strategies.
Useful for debugging why one model performs better than another.

Usage:
    python debug_model_comparison.py --benchmark arc --question-index 0 \\
        --models gemma:1b gemma:4b --reasoning direct zero-shot-cot

    python debug_model_comparison.py --benchmark mmlu --subject high_school_physics \\
        --question-index 5 --models gemma:1b gemma:4b gemma:7b \\
        --reasoning direct zero-shot-cot few-shot-cot
"""

import argparse
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from utils.ollama import check_ollama_connection, query_ollama
from utils.reasoning import create_strategy
from utils.cot_examples import get_cot_examples


def load_question_from_benchmark(benchmark: str, question_index: int, subject: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a specific question from a benchmark dataset

    Args:
        benchmark: Benchmark name (arc, mmlu, gpqa, hellaswag, truthfulqa, mmlu-pro)
        question_index: Index of the question to load
        subject: Subject name (required for MMLU)

    Returns:
        Dictionary with question, choices, correct_answer, and metadata
    """
    question_data = {}

    if benchmark == 'arc':
        from utils.arc import load_arc_dataset, parse_arc_sample
        dataset = load_arc_dataset('test', 'ARC-Challenge')  # Fixed: split first, then difficulty
        if dataset is None or question_index >= len(dataset):
            raise ValueError(f"Question index {question_index} out of range")

        sample = dataset[question_index]
        question, choices, correct_letter = parse_arc_sample(sample)

        question_data = {
            'question': question,
            'choices': choices,
            'correct_answer': correct_letter,
            'benchmark': 'ARC',
            'subject': 'ARC-Challenge',
            'index': question_index
        }

    elif benchmark == 'mmlu':
        if not subject:
            raise ValueError("MMLU requires --subject argument")

        from utils.mmlu import load_mmlu_subject, parse_mmlu_sample
        dataset = load_mmlu_subject(subject, 'test')
        if dataset is None or question_index >= len(dataset):
            raise ValueError(f"Question index {question_index} out of range")

        sample = dataset[question_index]
        question, choices, correct_letter = parse_mmlu_sample(sample)

        question_data = {
            'question': question,
            'choices': choices,
            'correct_answer': correct_letter,
            'benchmark': 'MMLU',
            'subject': subject,
            'index': question_index
        }

    elif benchmark == 'gpqa':
        from utils.gpqa import load_gpqa_dataset, parse_gpqa_sample
        dataset = load_gpqa_dataset('train')  # GPQA only has 'train' split
        if dataset is None or question_index >= len(dataset):
            raise ValueError(f"Question index {question_index} out of range")

        sample = dataset[question_index]
        question, choices, correct_letter = parse_gpqa_sample(sample)

        question_data = {
            'question': question,
            'choices': choices,
            'correct_answer': correct_letter,
            'benchmark': 'GPQA',
            'subject': 'gpqa',
            'index': question_index
        }

    elif benchmark == 'hellaswag':
        from utils.hellaswag import load_hellaswag_dataset, parse_hellaswag_sample
        dataset = load_hellaswag_dataset('validation')  # HellaSwag uses 'validation' split
        if dataset is None or question_index >= len(dataset):
            raise ValueError(f"Question index {question_index} out of range")

        sample = dataset[question_index]
        context, choices, correct_letter = parse_hellaswag_sample(sample)

        question_data = {
            'question': context,
            'choices': choices,
            'correct_answer': correct_letter,
            'benchmark': 'HellaSwag',
            'subject': 'hellaswag',
            'index': question_index
        }

    elif benchmark == 'truthfulqa':
        from utils.truthfulqa import load_truthfulqa_dataset, parse_truthfulqa_mc1
        dataset = load_truthfulqa_dataset()
        if dataset is None or question_index >= len(dataset):
            raise ValueError(f"Question index {question_index} out of range")

        sample = dataset[question_index]
        question, choices, correct_letter = parse_truthfulqa_mc1(sample)

        question_data = {
            'question': question,
            'choices': choices,
            'correct_answer': correct_letter,
            'benchmark': 'TruthfulQA',
            'subject': 'truthfulqa',
            'index': question_index
        }

    elif benchmark == 'mmlu-pro':
        from utils.mmlu_pro import load_mmlu_pro_dataset, parse_mmlu_pro_sample
        dataset = load_mmlu_pro_dataset('test')
        if dataset is None or question_index >= len(dataset):
            raise ValueError(f"Question index {question_index} out of range")

        sample = dataset[question_index]
        question, choices, correct_letter = parse_mmlu_pro_sample(sample)

        question_data = {
            'question': question,
            'choices': choices,
            'correct_answer': correct_letter,
            'benchmark': 'MMLU-Pro',
            'subject': 'mmlu-pro',
            'index': question_index
        }

    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    return question_data


def query_model_with_strategy(
    model_name: str,
    question: str,
    choices: List[str],
    reasoning_method: str,
    ollama_url: str,
    max_tokens: int = 32000,
    benchmark: str = 'arc'
) -> Dict[str, Any]:
    """
    Query a model with a specific reasoning strategy

    Returns:
        Dictionary with prompt, response, extracted_answer, and metadata
    """
    # Create strategy
    if reasoning_method == 'few-shot-cot':
        examples = get_cot_examples(benchmark.lower(), n=3)
        strategy = create_strategy('few-shot-cot', examples=examples, max_tokens=max_tokens)
    elif reasoning_method == 'self-consistency':
        strategy = create_strategy('self-consistency', base_strategy='zero-shot-cot',
                                   n_samples=5, max_tokens=max_tokens)
    else:
        strategy = create_strategy(reasoning_method, max_tokens=max_tokens)

    # Format prompt
    prompt = strategy.format_prompt(question, choices)

    # Determine valid choices based on number of options
    valid_choices = [chr(65 + i) for i in range(len(choices))]  # A, B, C, D, ...

    # Query model
    try:
        if strategy.requires_multiple_samples():
            # Self-consistency: multiple samples
            responses = []
            for i in range(strategy.get_num_samples()):
                response = query_ollama(
                    prompt=prompt,
                    model_name=model_name,
                    ollama_url=ollama_url,
                    temperature=strategy.get_temperature(),
                    max_tokens=strategy.get_max_tokens(),
                    timeout=180
                )
                responses.append(response)

            # Extract answer from multiple responses
            extracted_answer = strategy.extract_answer_from_multiple(responses, valid_choices)
            full_response = "\n\n---SAMPLE SEPARATOR---\n\n".join(
                [f"Sample {i+1}:\n{r}" for i, r in enumerate(responses)]
            )
        else:
            # Single response
            response = query_ollama(
                prompt=prompt,
                model_name=model_name,
                ollama_url=ollama_url,
                temperature=strategy.get_temperature(),
                max_tokens=strategy.get_max_tokens(),
                timeout=180
            )
            extracted_answer = strategy.extract_answer(response, valid_choices)
            full_response = response

        return {
            'prompt': prompt,
            'response': full_response,
            'extracted_answer': extracted_answer,
            'max_tokens': strategy.get_max_tokens(),
            'temperature': strategy.get_temperature(),
            'success': True,
            'error': None
        }

    except Exception as e:
        return {
            'prompt': prompt,
            'response': None,
            'extracted_answer': None,
            'max_tokens': strategy.get_max_tokens(),
            'temperature': strategy.get_temperature(),
            'success': False,
            'error': str(e)
        }


def print_question_info(question_data: Dict[str, Any]):
    """Print formatted question information"""
    print("\n" + "=" * 80)
    print("QUESTION INFORMATION")
    print("=" * 80)
    print(f"Benchmark: {question_data['benchmark']}")
    print(f"Subject: {question_data['subject']}")
    print(f"Question Index: {question_data['index']}")
    print(f"Correct Answer: {question_data['correct_answer']}")
    print()
    print(f"Question: {question_data['question']}")
    print()
    print("Choices:")
    for choice in question_data['choices']:
        print(f"  {choice}")
    print("=" * 80)


def print_model_response(
    model_name: str,
    reasoning_method: str,
    result: Dict[str, Any],
    correct_answer: str
):
    """Print formatted model response"""
    print("\n" + "-" * 80)
    print(f"MODEL: {model_name} | REASONING: {reasoning_method}")
    print("-" * 80)

    if not result['success']:
        print(f"❌ ERROR: {result['error']}")
        return

    # Check if correct
    is_correct = result['extracted_answer'] == correct_answer
    status_icon = "✓" if is_correct else "✗"

    print(f"Extracted Answer: {result['extracted_answer']} {status_icon}")
    print(f"Correct Answer: {correct_answer}")
    print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
    print(f"Max Tokens: {result['max_tokens']}")
    print(f"Temperature: {result['temperature']}")
    print()
    print("PROMPT:")
    print("-" * 40)
    print(result['prompt'])
    print()
    print("RESPONSE:")
    print("-" * 40)
    print(result['response'])
    print("-" * 80)


def generate_comparison_table(
    models: List[str],
    reasoning_methods: List[str],
    results: Dict[str, Dict[str, Dict]],
    correct_answer: str
):
    """Generate a comparison table showing all results"""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    # Header
    header = f"{'Model':<20} | "
    for method in reasoning_methods:
        header += f"{method:<25} | "
    print(header)
    print("-" * 80)

    # Rows
    for model in models:
        row = f"{model:<20} | "
        for method in reasoning_methods:
            result = results[model][method]
            if result['success']:
                answer = result['extracted_answer'] or 'None'
                is_correct = answer == correct_answer
                status = "✓" if is_correct else "✗"
                row += f"{answer} {status:<22} | "
            else:
                row += f"{'ERROR':<25} | "
        print(row)

    print("-" * 80)
    print(f"Correct Answer: {correct_answer}")
    print("=" * 80)


def generate_analysis(
    models: List[str],
    reasoning_methods: List[str],
    results: Dict[str, Dict[str, Dict]],
    correct_answer: str
):
    """Generate analysis of the results"""
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Count correct answers per model
    print("\nCorrect Answers per Model:")
    for model in models:
        correct_count = 0
        total_count = 0
        for method in reasoning_methods:
            result = results[model][method]
            if result['success']:
                total_count += 1
                if result['extracted_answer'] == correct_answer:
                    correct_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"  {model}: {correct_count}/{total_count} ({accuracy:.1%})")

    # Count correct answers per reasoning method
    print("\nCorrect Answers per Reasoning Method:")
    for method in reasoning_methods:
        correct_count = 0
        total_count = 0
        for model in models:
            result = results[model][method]
            if result['success']:
                total_count += 1
                if result['extracted_answer'] == correct_answer:
                    correct_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"  {method}: {correct_count}/{total_count} ({accuracy:.1%})")

    # Identify best and worst performers
    print("\nObservations:")

    # Find models that got it right vs wrong
    all_correct = []
    all_wrong = []
    mixed = []

    for model in models:
        correct_count = sum(
            1 for method in reasoning_methods
            if results[model][method]['success'] and
               results[model][method]['extracted_answer'] == correct_answer
        )
        total_count = sum(
            1 for method in reasoning_methods
            if results[model][method]['success']
        )

        if correct_count == total_count and total_count > 0:
            all_correct.append(model)
        elif correct_count == 0:
            all_wrong.append(model)
        else:
            mixed.append((model, correct_count, total_count))

    if all_correct:
        print(f"  ✓ Always correct: {', '.join(all_correct)}")
    if all_wrong:
        print(f"  ✗ Always wrong: {', '.join(all_wrong)}")
    if mixed:
        for model, correct, total in mixed:
            print(f"  ± Mixed results: {model} ({correct}/{total} correct)")

    # Check if reasoning helps
    for model in models:
        direct_result = results[model].get('direct', {})
        cot_result = results[model].get('zero-shot-cot', {})

        if direct_result.get('success') and cot_result.get('success'):
            direct_correct = direct_result['extracted_answer'] == correct_answer
            cot_correct = cot_result['extracted_answer'] == correct_answer

            if not direct_correct and cot_correct:
                print(f"  → {model}: CoT helped! (wrong → correct)")
            elif direct_correct and not cot_correct:
                print(f"  → {model}: CoT hurt! (correct → wrong)")

    print("=" * 80)


def save_results_to_file(
    question_data: Dict[str, Any],
    models: List[str],
    reasoning_methods: List[str],
    results: Dict[str, Dict[str, Dict]],
    output_path: str
):
    """Save detailed results to JSON file"""
    output = {
        'timestamp': datetime.now().isoformat(),
        'question': question_data,
        'models': models,
        'reasoning_methods': reasoning_methods,
        'results': results
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug model comparison on a specific question",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two Gemma models on ARC question
  python debug_model_comparison.py --benchmark arc --question-index 0 \\
      --models gemma:1b gemma:4b --reasoning direct zero-shot-cot

  # Compare multiple models with multiple reasoning methods
  python debug_model_comparison.py --benchmark mmlu --subject high_school_physics \\
      --question-index 5 --models gemma:1b gemma:4b gemma:7b \\
      --reasoning direct zero-shot-cot few-shot-cot

  # Compare on GPQA with self-consistency
  python debug_model_comparison.py --benchmark gpqa --question-index 10 \\
      --models gemma:4b gemma:7b --reasoning zero-shot-cot self-consistency

  # Save results to file
  python debug_model_comparison.py --benchmark arc --question-index 0 \\
      --models gemma:1b gemma:4b --reasoning direct zero-shot-cot \\
      --output debug_results.json
        """
    )

    # Question selection
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        choices=['arc', 'mmlu', 'gpqa', 'hellaswag', 'truthfulqa', 'mmlu-pro'],
        help='Benchmark to load question from'
    )
    parser.add_argument(
        '--question-index',
        type=int,
        required=True,
        help='Index of the question in the dataset (0-based)'
    )
    parser.add_argument(
        '--subject',
        type=str,
        default=None,
        help='Subject name (required for MMLU, e.g., high_school_physics)'
    )

    # Model and reasoning selection
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='List of model names to compare (e.g., gemma:1b gemma:4b)'
    )
    parser.add_argument(
        '--reasoning',
        type=str,
        nargs='+',
        default=['direct'],
        choices=['direct', 'zero-shot-cot', 'few-shot-cot', 'self-consistency'],
        help='Reasoning methods to test (default: direct)'
    )

    # Configuration
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:11434',
        help='Ollama API URL (default: http://localhost:11434)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=32000,
        help='Maximum tokens for model response (default: 32000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save detailed results to JSON file (optional)'
    )

    args = parser.parse_args()

    # Check Ollama connection
    print("Checking Ollama connection...")
    if not check_ollama_connection(args.url):
        print(f"❌ Error: Cannot connect to Ollama at {args.url}")
        print("Please ensure Ollama is running.")
        sys.exit(1)
    print("✓ Ollama is accessible\n")

    # Load question
    print(f"Loading question from {args.benchmark}...")
    try:
        question_data = load_question_from_benchmark(
            args.benchmark,
            args.question_index,
            args.subject
        )
    except Exception as e:
        print(f"❌ Error loading question: {e}")
        sys.exit(1)

    # Print question info
    print_question_info(question_data)

    # Query all models with all reasoning methods
    print(f"\nQuerying {len(args.models)} model(s) with {len(args.reasoning)} reasoning method(s)...")
    print("This may take a few minutes...\n")

    results = {}
    total_queries = len(args.models) * len(args.reasoning)
    current_query = 0

    for model in args.models:
        results[model] = {}
        for reasoning_method in args.reasoning:
            current_query += 1
            print(f"[{current_query}/{total_queries}] Querying {model} with {reasoning_method}...")

            result = query_model_with_strategy(
                model_name=model,
                question=question_data['question'],
                choices=question_data['choices'],
                reasoning_method=reasoning_method,
                ollama_url=args.url,
                max_tokens=args.max_tokens,
                benchmark=args.benchmark
            )

            results[model][reasoning_method] = result

            # Print brief status
            if result['success']:
                is_correct = result['extracted_answer'] == question_data['correct_answer']
                status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
                print(f"  → {status} (answered: {result['extracted_answer']})")
            else:
                print(f"  → ❌ ERROR: {result['error']}")

    # Print detailed results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for model in args.models:
        for reasoning_method in args.reasoning:
            print_model_response(
                model,
                reasoning_method,
                results[model][reasoning_method],
                question_data['correct_answer']
            )

    # Generate comparison table
    generate_comparison_table(
        args.models,
        args.reasoning,
        results,
        question_data['correct_answer']
    )

    # Generate analysis
    generate_analysis(
        args.models,
        args.reasoning,
        results,
        question_data['correct_answer']
    )

    # Save to file if requested
    if args.output:
        save_results_to_file(
            question_data,
            args.models,
            args.reasoning,
            results,
            args.output
        )

    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
