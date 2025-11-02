#!/usr/bin/env python3
"""
Batch Benchmark Evaluation Script

Run multiple benchmarks across multiple models with different reasoning strategies.
Results are automatically saved and organized.

Usage:
    # Run with command-line arguments
    python run_benchmarks.py --models gemma3:1b gemma3:4b --reasoning direct zero-shot-cot --benchmarks arc mmlu gpqa

    # Run with config file
    python run_benchmarks.py --config experiments/config.json

    # Run specific MMLU subjects
    python run_benchmarks.py --models gemma3:1b --reasoning direct --benchmarks mmlu --mmlu-subjects high_school_physics biology

    # Run with custom output directory
    python run_benchmarks.py --models gemma3:1b --benchmarks arc --output-dir results/experiment_1
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import subprocess
import time

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_benchmark_script(benchmark: str) -> Optional[str]:
    """Get the script name for a benchmark"""
    benchmark_scripts = {
        'arc': 'arc_benchmark.py',
        'mmlu': 'mmlu_benchmark.py',
        'gpqa': 'gpqa_benchmark.py',
        'hellaswag': 'hellaswag_benchmark.py',
        'truthfulqa': 'truthfulqa_benchmark.py',
        'mmlu-pro': 'mmlu_pro_benchmark.py',
        'gsm8k': 'gsm8k_benchmark.py',
        'humaneval': 'humaneval_benchmark.py'
    }
    return benchmark_scripts.get(benchmark)


def get_benchmark_args(
    benchmark: str,
    model: str,
    reasoning: str,
    ollama_url: str,
    max_samples: Optional[int] = None,
    mmlu_subject: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_tokens: int = 32000
) -> List[str]:
    """Construct command-line arguments for a benchmark"""

    script = get_benchmark_script(benchmark)
    if not script:
        return []

    args = ['python', script, '--model', model, '--reasoning', reasoning]

    # Add Ollama URL
    args.extend(['--url', ollama_url])

    # Note: max_tokens is configured internally in reasoning strategies, not passed as argument

    # Add max samples if specified
    if max_samples:
        args.extend(['--max-samples', str(max_samples)])

    # Add MMLU-specific subject
    if benchmark == 'mmlu' and mmlu_subject:
        args.extend(['--subject', mmlu_subject])

    # Add output file path if output_dir specified
    if output_dir:
        # Create output filename: {benchmark}_{model}_{reasoning}.csv
        # Sanitize model name (replace : with _)
        safe_model = model.replace(':', '_')
        filename = f"{benchmark}_{safe_model}_{reasoning}.json"
        if benchmark == 'mmlu' and mmlu_subject:
            filename = f"mmlu_{mmlu_subject}_{safe_model}_{reasoning}.json"

        output_path = Path(output_dir) / filename
        args.extend(['--output', str(output_path)])

    return args


def run_single_benchmark(
    benchmark: str,
    model: str,
    reasoning_config: Dict[str, Any],  # Changed from str to dict
    ollama_url: str,
    max_samples: Optional[int] = None,
    mmlu_subject: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_tokens: int = 32000
) -> Dict[str, Any]:
    """Run a single benchmark evaluation

    Args:
        reasoning_config: Dict with 'name', 'temperature', 'max_tokens', etc.
    """

    # Extract reasoning method name and settings
    reasoning_name = reasoning_config.get('name', 'direct')
    method_temp = reasoning_config.get('temperature')
    method_max_tokens = reasoning_config.get('max_tokens', max_tokens)

    # Construct command
    args = get_benchmark_args(
        benchmark, model, reasoning_name, ollama_url,
        max_samples, mmlu_subject, output_dir, method_max_tokens
    )

    if not args:
        return {
            'success': False,
            'error': f'Unknown benchmark: {benchmark}',
            'benchmark': benchmark,
            'model': model,
            'reasoning': reasoning_name
        }

    # Create benchmark identifier
    benchmark_id = benchmark
    if benchmark == 'mmlu' and mmlu_subject:
        benchmark_id = f"{benchmark}_{mmlu_subject}"

    # Display with temperature info if specified
    temp_info = f" (T={method_temp})" if method_temp is not None else ""
    print_info(f"Running: {benchmark_id} | {model} | {reasoning_name}{temp_info}")
    print(f"  Command: {' '.join(args)}")
    print()  # Add blank line before progress output

    start_time = time.time()

    try:
        # Run the benchmark with real-time output streaming
        # Pass output directly to terminal to preserve progress bars
        process = subprocess.Popen(
            args,
            stdout=None,  # Inherit stdout (goes directly to terminal)
            stderr=None,  # Inherit stderr (goes directly to terminal)
        )

        # Wait for process to complete with timeout
        try:
            return_code = process.wait(timeout=36000)  # 1 hour timeout
        except subprocess.TimeoutExpired:
            process.kill()
            elapsed_time = time.time() - start_time
            print_error("Timeout (1 hour)")
            return {
                'success': False,
                'benchmark': benchmark_id,
                'model': model,
                'reasoning': reasoning_name,
                'reasoning_config': reasoning_config,
                'elapsed_time': elapsed_time,
                'error': 'Timeout after 1 hour'
            }
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise

        elapsed_time = time.time() - start_time
        print()  # Add blank line after output

        if return_code == 0:
            print_success(f"Completed in {elapsed_time:.1f}s")
            return {
                'success': True,
                'benchmark': benchmark_id,
                'model': model,
                'reasoning': reasoning_name,
                'reasoning_config': reasoning_config,
                'elapsed_time': elapsed_time,
                'stdout': '',  # Not captured when streaming to terminal
                'stderr': ''
            }
        else:
            print_error(f"Failed with exit code {return_code}")
            return {
                'success': False,
                'benchmark': benchmark_id,
                'model': model,
                'reasoning': reasoning_name,
                'reasoning_config': reasoning_config,
                'elapsed_time': elapsed_time,
                'error': f'Process exited with code {return_code}',
                'exit_code': return_code
            }

    except Exception as e:
        print_error(f"Exception: {str(e)}")
        return {
            'success': False,
            'benchmark': benchmark_id,
            'model': model,
            'reasoning': reasoning_name,
            'reasoning_config': reasoning_config,
            'error': str(e)
        }


def run_batch_benchmarks(
    models: List[str],
    reasoning_methods: List[str],
    benchmarks: List[str],
    ollama_url: str = 'http://localhost:11434',
    max_samples: Optional[int] = None,
    mmlu_subjects: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    max_tokens: int = 32000,
    skip_existing: bool = False
) -> Dict[str, Any]:
    """Run multiple benchmarks across multiple models and reasoning methods"""

    print_header("BATCH BENCHMARK EVALUATION")

    # Print configuration
    print_info("Configuration:")
    print(f"  Models: {', '.join(models)}")
    # Extract method names for display (handle both str and dict formats)
    method_names = [m if isinstance(m, str) else m.get('name', 'unknown') for m in reasoning_methods]
    print(f"  Reasoning methods: {', '.join(method_names)}")
    print(f"  Benchmarks: {', '.join(benchmarks)}")
    if mmlu_subjects:
        print(f"  MMLU subjects: {', '.join(mmlu_subjects)}")
    if max_samples:
        print(f"  Max samples per benchmark: {max_samples}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Output directory: {output_dir or 'default'}")
    print()

    # Create output directory
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Calculate total runs
    total_runs = len(models) * len(reasoning_methods) * len(benchmarks)
    if 'mmlu' in benchmarks and mmlu_subjects:
        # MMLU with specific subjects counts as multiple benchmarks
        total_runs = (total_runs - len(models) * len(reasoning_methods) +
                     len(models) * len(reasoning_methods) * len(mmlu_subjects))

    print_info(f"Total benchmark runs: {total_runs}")
    print()

    # Track results
    results = []
    successful = 0
    failed = 0
    current_run = 0

    # Run all combinations
    for model in models:
        for reasoning in reasoning_methods:
            for benchmark in benchmarks:
                # Handle MMLU subjects
                if benchmark == 'mmlu' and mmlu_subjects:
                    for subject in mmlu_subjects:
                        current_run += 1
                        print(f"\n[{current_run}/{total_runs}] ", end='')

                        result = run_single_benchmark(
                            benchmark, model, reasoning, ollama_url,
                            max_samples, subject, output_dir, max_tokens
                        )
                        results.append(result)

                        if result['success']:
                            successful += 1
                        else:
                            failed += 1
                else:
                    current_run += 1
                    print(f"\n[{current_run}/{total_runs}] ", end='')

                    result = run_single_benchmark(
                        benchmark, model, reasoning, ollama_url,
                        max_samples, None, output_dir, max_tokens
                    )
                    results.append(result)

                    if result['success']:
                        successful += 1
                    else:
                        failed += 1

    # Summary
    print_header("BATCH EVALUATION SUMMARY")
    print(f"Total runs: {total_runs}")
    print_success(f"Successful: {successful}")
    if failed > 0:
        print_error(f"Failed: {failed}")
    print()

    # Calculate total time
    total_time = sum(r.get('elapsed_time', 0) for r in results)
    print(f"Total elapsed time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print()

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'models': models,
            'reasoning_methods': reasoning_methods,
            'benchmarks': benchmarks,
            'mmlu_subjects': mmlu_subjects,
            'max_samples': max_samples,
            'max_tokens': max_tokens,
            'ollama_url': ollama_url
        },
        'results': results,
        'summary': {
            'total_runs': total_runs,
            'successful': successful,
            'failed': failed,
            'total_time': total_time
        }
    }

    # Save to file
    if output_dir:
        summary_path = Path(output_dir) / 'batch_summary.json'
    else:
        summary_path = Path('results') / f'batch_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print_success(f"Summary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run batch benchmark evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with multiple models and benchmarks
  python run_benchmarks.py --models gemma3:1b gemma3:4b --reasoning direct zero-shot-cot --benchmarks arc gpqa

  # Run specific MMLU subjects
  python run_benchmarks.py --models gemma3:1b --reasoning direct --benchmarks mmlu --mmlu-subjects physics biology

  # Limit samples for quick testing
  python run_benchmarks.py --models gemma3:1b --reasoning direct --benchmarks arc --max-samples 100

  # Use config file
  python run_benchmarks.py --config experiments/quick_test.json

  # Custom output directory
  python run_benchmarks.py --models gemma3:1b --benchmarks arc --output-dir results/test_run_1
        """
    )

    # Configuration options
    parser.add_argument(
        '--config',
        type=str,
        help='JSON configuration file (overrides other arguments)'
    )

    # Model and reasoning selection
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='List of model names (e.g., gemma3:1b gemma3:4b)'
    )
    parser.add_argument(
        '--reasoning',
        type=str,
        nargs='+',
        default=['direct'],
        choices=['direct', 'zero-shot-cot', 'few-shot-cot', 'self-consistency'],
        help='Reasoning methods to test (default: direct)'
    )

    # Benchmark selection
    parser.add_argument(
        '--benchmarks',
        type=str,
        nargs='+',
        choices=['arc', 'mmlu', 'gpqa', 'hellaswag', 'truthfulqa', 'mmlu-pro', 'gsm8k', 'humaneval'],
        help='Benchmarks to run'
    )
    parser.add_argument(
        '--mmlu-subjects',
        type=str,
        nargs='+',
        help='MMLU subjects to test (only used if mmlu is in benchmarks)'
    )

    # Configuration
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:11434',
        help='Ollama API URL (default: http://localhost:11434)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per benchmark (default: all)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=32000,
        help='Maximum tokens for model response (default: 32000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: results/)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip benchmarks that already have results'
    )

    args = parser.parse_args()

    # Load from config file if specified
    if args.config:
        print_info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        models = config.get('models', [])
        reasoning_methods_raw = config.get('reasoning_methods', ['direct'])

        # Handle both formats: simple list of strings OR list of dicts with settings
        reasoning_methods = []
        for method in reasoning_methods_raw:
            if isinstance(method, str):
                # Simple format: just method name
                reasoning_methods.append({
                    'name': method,
                    'temperature': None,
                    'max_tokens': config.get('max_tokens', 32000)
                })
            elif isinstance(method, dict):
                # Advanced format: dict with settings
                reasoning_methods.append(method)
            else:
                raise ValueError(f"Invalid reasoning_methods format: {method}")

        benchmarks = config.get('benchmarks', [])
        mmlu_subjects = config.get('mmlu_subjects', None)
        ollama_url = config.get('ollama_url', 'http://localhost:11434')
        max_samples = config.get('max_samples', None)
        max_tokens = config.get('max_tokens', 32000)
        output_dir = config.get('output_dir', None)
        skip_existing = config.get('skip_existing', False)
    else:
        # Use command-line arguments
        if not args.models or not args.benchmarks:
            parser.error("--models and --benchmarks are required (or use --config)")

        models = args.models
        # Convert simple string list to dict format for consistency
        reasoning_methods = [
            {
                'name': method,
                'temperature': None,
                'max_tokens': args.max_tokens
            }
            for method in args.reasoning
        ]
        benchmarks = args.benchmarks
        mmlu_subjects = args.mmlu_subjects
        ollama_url = args.url
        max_samples = args.max_samples
        max_tokens = args.max_tokens
        output_dir = args.output_dir
        skip_existing = args.skip_existing

    # Run batch evaluation
    try:
        summary = run_batch_benchmarks(
            models=models,
            reasoning_methods=reasoning_methods,
            benchmarks=benchmarks,
            ollama_url=ollama_url,
            max_samples=max_samples,
            mmlu_subjects=mmlu_subjects,
            output_dir=output_dir,
            max_tokens=max_tokens,
            skip_existing=skip_existing
        )

        # Exit with appropriate code
        if summary['summary']['failed'] > 0:
            print_warning("Some benchmarks failed. Check the summary for details.")
            sys.exit(1)
        else:
            print_success("All benchmarks completed successfully!")
            sys.exit(0)

    except KeyboardInterrupt:
        print_error("\nBatch evaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
