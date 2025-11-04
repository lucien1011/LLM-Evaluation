#!/usr/bin/env python3
"""
Debug Model Comparison Script for GSM8K (Math Problems)

Compare how different models solve the same math problem using different reasoning strategies.
Useful for debugging why one model performs better than another on mathematical reasoning.

Usage:
    python debug_model_comparison.py --question-index 0 \\
        --models gemma3:1b gemma3:4b --reasoning direct zero-shot-cot

    python debug_model_comparison.py --question-index 10 \\
        --models gemma3:1b gemma3:4b gemma3:27b \\
        --reasoning direct zero-shot-cot self-consistency
"""

import argparse
import sys
from typing import List, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from utils.ollama import check_ollama_connection, query_ollama
from utils.gsm8k import (
    load_gsm8k_dataset,
    parse_gsm8k_sample,
    extract_numerical_answer,
    compare_numerical_answers,
    format_gsm8k_instruction
)
from utils.reasoning import create_strategy
from utils.cot_examples import get_cot_examples

# Import reusable functions from MC debug script
from mc_debug_model_comparison import (
    save_results_to_file,
    generate_analysis,
    generate_comparison_table
)


def load_gsm8k_question(question_index: int, split: str = 'test') -> Dict[str, Any]:
    """
    Load a specific question from GSM8K dataset

    Args:
        question_index: Index of the question to load
        split: Dataset split ('test' or 'train')

    Returns:
        Dictionary with question, correct_answer, and metadata
    """
    dataset = load_gsm8k_dataset(split)
    if dataset is None or question_index >= len(dataset):
        raise ValueError(f"Question index {question_index} out of range (dataset has {len(dataset) if dataset else 0} questions)")

    sample = dataset[question_index]
    question, correct_answer = parse_gsm8k_sample(sample)

    return {
        'question': question,
        'correct_answer': correct_answer,
        'benchmark': 'GSM8K',
        'subject': 'math',
        'index': question_index,
        'split': split
    }


def query_model_with_strategy(
    model_name: str,
    question: str,
    reasoning_method: str,
    ollama_url: str,
    max_tokens: int = 32000,
    temperature: float = 0.5
) -> Dict[str, Any]:
    """
    Query a model with a specific reasoning strategy for GSM8K

    Returns:
        Dictionary with prompt, response, extracted_answer, and metadata
    """
    instruction = format_gsm8k_instruction()

    # Create strategy
    if reasoning_method == 'few-shot-cot':
        examples = get_cot_examples('gsm8k', n=3)
        strategy = create_strategy('few-shot-cot', examples=examples, max_tokens=max_tokens, temperature=temperature)
    elif reasoning_method == 'self-consistency':
        strategy = create_strategy('self-consistency', base_strategy='zero-shot-cot',
                                   n_samples=5, max_tokens=max_tokens, temperature=temperature)
    else:
        strategy = create_strategy(reasoning_method, max_tokens=max_tokens, temperature=temperature)

    # Format prompt based on strategy
    if hasattr(strategy, 'cot_trigger'):
        # Zero-shot CoT: add trigger to instruction
        prompt = f"{instruction}\n\n{question}\n\n{strategy.cot_trigger}"
    elif hasattr(strategy, 'examples'):
        # Few-shot CoT: format with examples
        from utils.cot_examples import format_cot_example
        formatted_examples = [format_cot_example(ex, 'gsm8k') for ex in strategy.examples]
        examples_str = '\n\n---\n\n'.join(formatted_examples)
        prompt = f"{instruction}\n\n{examples_str}\n\n---\n\nQuestion: {question}"
    else:
        # Direct strategy
        prompt = f"{instruction}\n\n{question}"

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

            # Extract numerical answers from all responses
            predicted_answers = [extract_numerical_answer(r) for r in responses]
            # Use majority voting
            from collections import Counter
            answer_counts = Counter(str(a) for a in predicted_answers if a is not None)
            if answer_counts:
                most_common_str = answer_counts.most_common(1)[0][0]
                try:
                    extracted_answer = float(most_common_str)
                except:
                    extracted_answer = None
            else:
                extracted_answer = None

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
            extracted_answer = extract_numerical_answer(response)
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
            'max_tokens': strategy.get_max_tokens() if strategy else max_tokens,
            'temperature': strategy.get_temperature() if strategy else temperature,
            'success': False,
            'error': str(e)
        }


def print_question_info(question_data: Dict[str, Any]):
    """Print formatted question information for GSM8K"""
    print("\n" + "=" * 80)
    print("QUESTION INFORMATION")
    print("=" * 80)
    print(f"Benchmark: {question_data['benchmark']}")
    print(f"Subject: {question_data['subject']}")
    print(f"Split: {question_data['split']}")
    print(f"Question Index: {question_data['index']}")
    print(f"Correct Answer: {question_data['correct_answer']}")
    print()
    print(f"Question:")
    print(f"{question_data['question']}")
    print("=" * 80)


def print_model_response(
    model_name: str,
    reasoning_method: str,
    result: Dict[str, Any],
    correct_answer: float
):
    """Print formatted model response for GSM8K"""
    print("\n" + "-" * 80)
    print(f"MODEL: {model_name} | REASONING: {reasoning_method}")
    print("-" * 80)

    if not result['success']:
        print(f"❌ ERROR: {result['error']}")
        return

    # Check if correct using numerical comparison
    is_correct = compare_numerical_answers(result['extracted_answer'], correct_answer)
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
    # Truncate very long responses for readability
    response = result['response']
    if len(response) > 2000:
        print(response[:2000])
        print("\n... [truncated] ...")
    else:
        print(response)
    print("-" * 80)


def generate_comparison_table_gsm8k(
    models: List[str],
    reasoning_methods: List[str],
    results: Dict[str, Dict[str, Dict]],
    correct_answer: float
):
    """Generate a comparison table showing all results for GSM8K"""
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
                answer = result['extracted_answer']
                if answer is not None:
                    answer_str = str(answer)
                    is_correct = compare_numerical_answers(answer, correct_answer)
                    status = "✓" if is_correct else "✗"
                    row += f"{answer_str} {status:<{24-len(answer_str)}} | "
                else:
                    row += f"{'None ✗':<25} | "
            else:
                row += f"{'ERROR':<25} | "
        print(row)

    print("-" * 80)
    print(f"Correct Answer: {correct_answer}")
    print("=" * 80)


def generate_analysis_gsm8k(
    models: List[str],
    reasoning_methods: List[str],
    results: Dict[str, Dict[str, Dict]],
    correct_answer: float
):
    """Generate analysis of the results for GSM8K"""
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
                if compare_numerical_answers(result['extracted_answer'], correct_answer):
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
                if compare_numerical_answers(result['extracted_answer'], correct_answer):
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
               compare_numerical_answers(results[model][method]['extracted_answer'], correct_answer)
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
            direct_correct = compare_numerical_answers(direct_result['extracted_answer'], correct_answer)
            cot_correct = compare_numerical_answers(cot_result['extracted_answer'], correct_answer)

            if not direct_correct and cot_correct:
                print(f"  → {model}: CoT helped! (wrong → correct)")
            elif direct_correct and not cot_correct:
                print(f"  → {model}: CoT hurt! (correct → wrong)")

    # Show numerical differences for incorrect answers
    print("\nNumerical Differences (for incorrect answers):")
    for model in models:
        for method in reasoning_methods:
            result = results[model][method]
            if result['success'] and result['extracted_answer'] is not None:
                if not compare_numerical_answers(result['extracted_answer'], correct_answer):
                    diff = result['extracted_answer'] - correct_answer
                    percent_error = abs(diff / correct_answer * 100) if correct_answer != 0 else float('inf')
                    print(f"  {model} ({method}): {result['extracted_answer']} (off by {diff:+.2f}, {percent_error:.1f}% error)")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Debug model comparison on GSM8K math problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models on GSM8K question
  python debug_model_comparison.py --question-index 0 \\
      --models gemma3:1b gemma3:4b --reasoning direct zero-shot-cot

  # Compare multiple models with multiple reasoning methods
  python debug_model_comparison.py --question-index 10 \\
      --models gemma3:1b gemma3:4b gemma3:27b \\
      --reasoning direct zero-shot-cot self-consistency

  # Use training set question
  python debug_model_comparison.py --question-index 0 --split train \\
      --models gemma3:1b gemma3:4b --reasoning direct zero-shot-cot

  # Save results to file
  python debug_model_comparison.py --question-index 0 \\
      --models gemma3:1b gemma3:4b --reasoning direct zero-shot-cot \\
      --output debug_gsm8k_results.json
        """
    )

    # Question selection
    parser.add_argument(
        '--question-index',
        type=int,
        required=True,
        help='Index of the question in the dataset (0-based)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['test', 'train'],
        help='Dataset split (default: test)'
    )

    # Model and reasoning selection
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='List of model names to compare (e.g., gemma3:1b gemma3:4b)'
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
        '--temperature',
        type=float,
        default=0.5,
        help='Temperature for model response (default: 0.5)'
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
    print(f"Loading question from GSM8K ({args.split} split)...")
    try:
        question_data = load_gsm8k_question(args.question_index, args.split)
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
                reasoning_method=reasoning_method,
                ollama_url=args.url,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )

            results[model][reasoning_method] = result

            # Print brief status
            if result['success']:
                is_correct = compare_numerical_answers(result['extracted_answer'], question_data['correct_answer'])
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
    generate_comparison_table_gsm8k(
        args.models,
        args.reasoning,
        results,
        question_data['correct_answer']
    )

    # Generate analysis
    generate_analysis_gsm8k(
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
