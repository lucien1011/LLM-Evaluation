"""
GSM8K (Grade School Math 8K) specific utility functions

GSM8K is a dataset of grade school math word problems.
- 8,500 questions (7,500 train, 1,000 test)
- Free-form numerical answers
- Requires multi-step reasoning
"""

from datasets import load_dataset
from typing import Dict, Tuple
import re


def load_gsm8k_dataset(split: str = "test"):
    """
    Load GSM8K dataset

    Args:
        split: Dataset split ('train' or 'test')

    Returns:
        Dataset object or None if error
    """
    try:
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        return dataset
    except Exception as e:
        print(f"Error loading GSM8K dataset: {e}")
        return None


def parse_gsm8k_sample(sample: Dict) -> Tuple[str, str]:
    """
    Parse GSM8K sample

    Args:
        sample: GSM8K sample dictionary with fields:
            - question: str (the math word problem)
            - answer: str (solution with final answer after ####)

    Returns:
        Tuple of (question, correct_answer_number)
    """
    question = sample['question']

    # Answer format: "Step 1...\nStep 2...\n#### 42"
    # Extract the number after ####
    answer_text = sample['answer']

    # Find the number after ####
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
    if match:
        # Remove commas from numbers like "1,000"
        correct_answer = match.group(1).replace(',', '')
    else:
        # Fallback: just take the last number in the answer
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_text)
        if numbers:
            correct_answer = numbers[-1].replace(',', '')
        else:
            correct_answer = "0"

    return question, correct_answer


def extract_numerical_answer(response: str) -> str:
    """
    Extract numerical answer from model response

    Handles various formats:
    - "The answer is 42"
    - "42"
    - "Therefore, the result is 3.14"
    - "1,234"

    Args:
        response: Model response text

    Returns:
        Extracted number as string, or None if no number found
    """
    # Clean up response
    response = response.strip()

    # Try to find "answer is X" or "result is X" patterns
    patterns = [
        r'answer\s+is\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'result\s+is\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'equals?\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',  # GSM8K format
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '')

    # If no pattern match, try to extract the last number in the response
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', response)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def compare_numerical_answers(predicted: str, correct: str) -> bool:
    """
    Compare two numerical answers

    Args:
        predicted: Predicted answer (as string)
        correct: Correct answer (as string)

    Returns:
        True if answers match (allowing small floating point errors)
    """
    if predicted is None:
        return False

    try:
        # Try to convert to float for comparison
        pred_num = float(predicted)
        correct_num = float(correct)

        # Allow small floating point errors
        return abs(pred_num - correct_num) < 1e-6
    except (ValueError, TypeError):
        # If conversion fails, do string comparison
        return predicted.strip() == correct.strip()


def get_dataset_stats(dataset) -> Dict:
    """
    Get statistics about the dataset

    Args:
        dataset: GSM8K dataset

    Returns:
        Dictionary with statistics
    """
    return {
        "total_samples": len(dataset),
    }


def format_gsm8k_instruction() -> str:
    """
    Get custom instruction for GSM8K

    Returns:
        Instruction string
    """
    return (
        "Solve the following grade school math problem step by step.\n"
        "Show your reasoning, then provide the final numerical answer.\n"
        "Format your final answer as: #### [number]"
    )
