"""
HellaSwag utility functions

HellaSwag tests commonsense reasoning about everyday situations.
- ~70,000 questions
- Sentence completion format with 4 choices
- Tests physical and social commonsense
"""

from datasets import load_dataset
from typing import Dict, List, Tuple


def load_hellaswag_dataset(split: str = "validation"):
    """
    Load HellaSwag dataset

    Args:
        split: Dataset split ('validation' or 'train')

    Returns:
        Dataset object or None if error
    """
    try:
        dataset = load_dataset("Rowan/hellaswag", split=split)
        return dataset
    except Exception as e:
        print(f"Error loading HellaSwag dataset: {e}")
        return None


def parse_hellaswag_sample(sample: Dict) -> Tuple[str, List[str], str]:
    """
    Parse HellaSwag sample

    Args:
        sample: HellaSwag sample dictionary with fields:
            - ctx: context/prompt string
            - endings: list of 4 possible continuations
            - label: correct answer index (0-3)

    Returns:
        Tuple of (question, choices, correct_answer_letter)
    """
    context = sample['ctx']
    endings = sample['endings']
    label = int(sample['label'])

    # Create the question from context
    question = f"What happens next?\n\nContext: {context}"

    # The endings are the choices
    choices = endings

    # Convert label to letter
    from .prompts import index_to_letter
    correct_letter = index_to_letter(label)

    return question, choices, correct_letter
