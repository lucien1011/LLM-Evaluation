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
        subset: One of 'gpqa_main', 'gpqa_diamond', 'gpqa_extended', 'gpqa_experts'
            - gpqa_main: 448 questions (default)
            - gpqa_diamond: 198 highest quality questions
            - gpqa_extended: 546 questions (includes all)
            - gpqa_experts: Expert-validated subset

    Returns:
        Dataset object or None if error
    """
    try:
        # Use Wanfq/gpqa as it's publicly accessible
        # Original dataset (Idavidrein/gpqa) is gated to prevent data contamination
        dataset = load_dataset("Wanfq/gpqa", subset, split=split)
        return dataset
    except Exception as e:
        print(f"Error loading GPQA dataset: {e}")
        return None


def parse_gpqa_sample(sample: Dict) -> Tuple[str, List[str], str]:
    """
    Parse GPQA sample

    Args:
        sample: GPQA sample dictionary with fields:
            - Question: str (post-revision, preferred)
            - Correct Answer: str (one of the choice texts)
            - Incorrect Answer 1: str
            - Incorrect Answer 2: str
            - Incorrect Answer 3: str

    Returns:
        Tuple of (question, choices, correct_answer_letter)
    """
    import random
    import hashlib

    # Use post-revision question (better quality)
    question = sample.get('Question', sample.get('Pre-Revision Question', ''))

    # Collect all choices (use post-revision answers)
    correct_answer = sample.get('Correct Answer', sample.get('Pre-Revision Correct Answer', ''))
    incorrect_answers = [
        sample.get('Incorrect Answer 1', sample.get('Pre-Revision Incorrect Answer 1', '')),
        sample.get('Incorrect Answer 2', sample.get('Pre-Revision Incorrect Answer 2', '')),
        sample.get('Incorrect Answer 3', sample.get('Pre-Revision Incorrect Answer 3', ''))
    ]

    # Combine all choices
    all_choices = [correct_answer] + incorrect_answers

    # Shuffle deterministically based on question hash
    # This ensures: 1) Same question always has same order, 2) Different questions have different orders
    question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
    seed = int(question_hash[:8], 16)  # Use first 8 hex chars as seed
    rng = random.Random(seed)
    rng.shuffle(all_choices)

    # Find the index of the correct answer after shuffling
    correct_index = all_choices.index(correct_answer)

    # Convert index to letter
    from .prompts import index_to_letter
    correct_letter = index_to_letter(correct_index)

    return question, all_choices, correct_letter


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
