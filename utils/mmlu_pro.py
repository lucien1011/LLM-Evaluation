"""
MMLU-Pro specific utility functions

MMLU-Pro is a more challenging version with:
- 12,000+ questions (vs 14,000 in MMLU)
- 10 answer choices (A-J) instead of 4 (A-D)
- More complex questions across 14 categories
- Chain-of-thought explanations included
"""

from datasets import load_dataset
from typing import Dict, List, Tuple


# MMLU-Pro categories (14 total)
MMLU_PRO_CATEGORIES = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "philosophy",
    "physics",
    "psychology",
    "other"
]


def load_mmlu_pro_dataset(split: str = "test"):
    """
    Load MMLU-Pro dataset

    Args:
        split: Dataset split ('test' or 'validation')

    Returns:
        Dataset object or None if error
    """
    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
        return dataset
    except Exception as e:
        print(f"Error loading MMLU-Pro dataset: {e}")
        return None


def filter_by_category(dataset, category: str):
    """
    Filter dataset by category

    Args:
        dataset: MMLU-Pro dataset
        category: Category name (from MMLU_PRO_CATEGORIES)

    Returns:
        Filtered dataset
    """
    if category not in MMLU_PRO_CATEGORIES:
        print(f"Warning: '{category}' not in known categories")

    return dataset.filter(lambda x: x['category'] == category)


def parse_mmlu_pro_sample(sample: Dict) -> Tuple[str, List[str], str]:
    """
    Parse MMLU-Pro sample to extract question, choices, and answer

    Args:
        sample: MMLU-Pro sample dictionary with fields:
            - question: str
            - options: List[str] (10 choices)
            - answer: str ('A' through 'J')
            - answer_index: int (0-9)
            - category: str

    Returns:
        Tuple of (question, choices, correct_answer_letter)
    """
    question = sample['question']
    choices = sample['options']  # List of 10 options
    correct_answer = sample['answer']  # Letter A-J

    # Ensure it's uppercase
    correct_letter = correct_answer.upper()

    return question, choices, correct_letter


def get_sample_category(sample: Dict) -> str:
    """
    Get the category of a sample

    Args:
        sample: MMLU-Pro sample dictionary

    Returns:
        Category name
    """
    return sample.get('category', 'unknown')


def get_cot_explanation(sample: Dict) -> str:
    """
    Get the chain-of-thought explanation for a sample

    Args:
        sample: MMLU-Pro sample dictionary

    Returns:
        Chain-of-thought explanation text
    """
    return sample.get('cot_content', '')


def answer_index_to_letter(index: int) -> str:
    """
    Convert answer index (0-9) to letter (A-J)

    Args:
        index: Zero-based index (0-9)

    Returns:
        Corresponding letter (A-J)
    """
    if 0 <= index <= 9:
        return chr(65 + index)  # 65 is ASCII for 'A'
    else:
        raise ValueError(f"Index must be 0-9, got {index}")


def answer_letter_to_index(letter: str) -> int:
    """
    Convert answer letter (A-J) to index (0-9)

    Args:
        letter: Letter A-J

    Returns:
        Zero-based index (0-9)
    """
    letter = letter.upper()
    if 'A' <= letter <= 'J':
        return ord(letter) - 65
    else:
        raise ValueError(f"Letter must be A-J, got {letter}")


def get_dataset_stats(dataset) -> Dict:
    """
    Get statistics about the dataset

    Args:
        dataset: MMLU-Pro dataset

    Returns:
        Dictionary with statistics
    """
    categories = {}
    for sample in dataset:
        cat = sample['category']
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_samples": len(dataset),
        "categories": categories,
        "num_categories": len(categories)
    }
