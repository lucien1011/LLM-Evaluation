"""
GPQA (Graduate-Level Google-Proof Q&A) specific utility functions

GPQA is a challenging benchmark with graduate-level questions in science.
- 448 questions across 3 subsets: main, diamond, extended
- Multiple choice with 4 options
- Expert-verified, Google-proof questions
"""

from datasets import load_dataset
from typing import Dict, List, Tuple


def load_gpqa_dataset(split: str = "train", subset: str = "gpqa_main"):
    """
    Load GPQA dataset

    Args:
        split: Dataset split ('train' is the main evaluation set)
        subset: One of 'gpqa_main', 'gpqa_diamond', 'gpqa_extended'
            - gpqa_main: 448 questions (default)
            - gpqa_diamond: 198 highest quality questions
            - gpqa_extended: 546 questions (includes all)

    Returns:
        Dataset object or None if error
    """
    try:
        dataset = load_dataset("Idavidrein/gpqa", subset, split=split)
        return dataset
    except Exception as e:
        print(f"Error loading GPQA dataset: {e}")
        return None


def parse_gpqa_sample(sample: Dict) -> Tuple[str, List[str], str]:
    """
    Parse GPQA sample

    Args:
        sample: GPQA sample dictionary with fields:
            - Question: str
            - Correct Answer: str (one of the choice texts)
            - Incorrect Answer 1: str
            - Incorrect Answer 2: str
            - Incorrect Answer 3: str

    Returns:
        Tuple of (question, choices, correct_answer_letter)
    """
    question = sample['Question']

    # Collect all choices
    correct_answer = sample['Correct Answer']
    incorrect_answers = [
        sample['Incorrect Answer 1'],
        sample['Incorrect Answer 2'],
        sample['Incorrect Answer 3']
    ]

    # Combine and shuffle to avoid position bias
    # Put correct answer first, then incorrect ones
    choices = [correct_answer] + incorrect_answers

    # The correct answer is at index 0 (we put it first)
    from .prompts import index_to_letter
    correct_letter = index_to_letter(0)

    return question, choices, correct_letter


def get_dataset_stats(dataset) -> Dict:
    """
    Get statistics about the dataset

    Args:
        dataset: GPQA dataset

    Returns:
        Dictionary with statistics
    """
    return {
        "total_samples": len(dataset),
    }


def format_gpqa_instruction() -> str:
    """
    Get custom instruction for GPQA

    Returns:
        Instruction string
    """
    return (
        "Answer the following graduate-level question by selecting the correct option.\n"
        "These questions require expert-level knowledge in science.\n"
        "Only respond with the letter of your answer (A, B, C, or D), nothing else."
    )
