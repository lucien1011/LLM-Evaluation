"""
Utility functions for formatting prompts and extracting answers
Can be reused for different multiple-choice benchmarks
"""

from typing import List


def format_multiple_choice_prompt(
    question: str,
    choices: List[str],
    instruction: str = None
) -> str:
    """
    Format a multiple choice question into a prompt

    Args:
        question: The question text
        choices: List of answer choices
        instruction: Optional custom instruction (uses default if None)

    Returns:
        Formatted prompt string
    """
    if instruction is None:
        instruction = (
            "Answer the following multiple choice question by selecting the correct option (A, B, C, or D).\n"
            "Only respond with the letter of your answer (A, B, C, or D), nothing else."
        )

    formatted_choices = "\n".join([
        f"{chr(65+i)}. {choice}"
        for i, choice in enumerate(choices)
    ])

    prompt = f"""{instruction}

Question: {question}

{formatted_choices}

Answer:"""

    return prompt


def extract_letter_answer(response: str, valid_choices: List[str] = None) -> str:
    """
    Extract a letter answer from model response

    Args:
        response: Raw model response
        valid_choices: List of valid choices (default: ['A', 'B', 'C', 'D'])

    Returns:
        Single letter answer or empty string if not found
    """
    if valid_choices is None:
        valid_choices = ['A', 'B', 'C', 'D']

    # Clean and uppercase the response
    response = response.strip().upper()

    # Look for valid choices in the response
    for char in response:
        if char in valid_choices:
            return char

    return ""


def index_to_letter(index: int) -> str:
    """
    Convert answer index to letter (0->A, 1->B, etc.)

    Args:
        index: Zero-based index

    Returns:
        Corresponding letter
    """
    return chr(65 + index)


def letter_to_index(letter: str) -> int:
    """
    Convert answer letter to index (A->0, B->1, etc.)

    Args:
        letter: Letter (A-Z)

    Returns:
        Zero-based index
    """
    return ord(letter.upper()) - 65
