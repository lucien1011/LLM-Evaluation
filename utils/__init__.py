"""
Utility modules for LLM evaluation toolkit

This package contains reusable utilities for:
- Ollama API interaction
- Prompt formatting
- Evaluation framework
- Dataset handling (MMLU, MMLU-Pro)
- Visualization
"""

from .ollama import check_ollama_connection, query_ollama, get_available_models
from .prompts import format_multiple_choice_prompt, extract_letter_answer, index_to_letter
from .evaluation import evaluate_dataset, save_results, print_results_summary
from .visualization import load_result_file, create_timestamp, get_color_palette

__all__ = [
    # Ollama
    'check_ollama_connection',
    'query_ollama',
    'get_available_models',
    # Prompts
    'format_multiple_choice_prompt',
    'extract_letter_answer',
    'index_to_letter',
    # Evaluation
    'evaluate_dataset',
    'save_results',
    'print_results_summary',
    # Visualization
    'load_result_file',
    'create_timestamp',
    'get_color_palette',
]
