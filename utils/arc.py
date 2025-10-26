"""
ARC (AI2 Reasoning Challenge) utility functions

ARC is a dataset of science questions for testing reasoning ability.
- ~7,800 genuine grade-school science questions
- Two difficulty levels: Easy and Challenge
- 4 answer choices (A-D)
- Tests scientific reasoning and common sense
"""

from datasets import load_dataset
from typing import Dict, List, Tuple


def load_arc_dataset(split: str = "test", difficulty: str = "ARC-Challenge"):
    """
    Load ARC dataset

    Args:
        split: Dataset split ('test', 'validation', or 'train')
        difficulty: 'ARC-Easy' or 'ARC-Challenge' (default)

    Returns:
        Dataset object or None if error
    """
    try:
        dataset = load_dataset("allenai/ai2_arc", difficulty, split=split)
        return dataset
    except Exception as e:
        print(f"Error loading ARC dataset: {e}")
        return None


def parse_arc_sample(sample: Dict) -> Tuple[str, List[str], str]:
    """
    Parse ARC sample to extract question, choices, and answer

    Args:
        sample: ARC sample dictionary with fields:
            - question: str
            - choices: dict with 'text' and 'label' lists
            - answerKey: str ('A', 'B', 'C', 'D', or '1', '2', '3', '4')

    Returns:
        Tuple of (question, choices, correct_answer_letter)
    """
    question = sample['question']
    choices = sample['choices']['text']
    answer_key = sample['answerKey']

    # Convert numeric answer keys (1,2,3,4) to letters (A,B,C,D)
    if answer_key.isdigit():
        from .prompts import index_to_letter
        correct_letter = index_to_letter(int(answer_key) - 1)
    else:
        correct_letter = answer_key.upper()

    return question, choices, correct_letter


def get_dataset_stats(dataset) -> Dict:
    """
    Get statistics about the dataset

    Args:
        dataset: ARC dataset

    Returns:
        Dictionary with statistics
    """
    return {
        "total_samples": len(dataset),
    }
