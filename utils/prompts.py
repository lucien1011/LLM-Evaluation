"""
Utility functions for formatting prompts and extracting answers
Can be reused for different multiple-choice benchmarks
"""

import re
from typing import List, Optional


def format_multiple_choice_prompt(
    question: str,
    choices: List[str],
    instruction: str = None,
    use_boxed: bool = True
) -> str:
    """
    Format a multiple choice question into a prompt

    Args:
        question: The question text
        choices: List of answer choices
        instruction: Optional custom instruction (uses default if None)
        use_boxed: If True, instruct to use \\boxed{} notation (default: True)

    Returns:
        Formatted prompt string
    """
    if instruction is None:
        if use_boxed:
            instruction = (
                "Answer the following multiple choice question by selecting the correct option.\n"
                "Provide your final answer in the format: \\boxed{X} where X is the letter of your choice."
            )
        else:
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


def extract_boxed_answer(response: str, valid_choices: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract answer from \\boxed{} notation in model response.
    Falls back to extract_letter_answer if no boxed notation found.

    Handles various formats:
    - \\boxed{A}
    - \\boxed{B.}
    - \\boxed{C}
    - $\\boxed{D}$
    - \\text{boxed}{E}

    Args:
        response: Raw model response
        valid_choices: List of valid choices (default: ['A', 'B', 'C', 'D'])

    Returns:
        Single letter answer or None if not found
    """
    if valid_choices is None:
        valid_choices = ['A', 'B', 'C', 'D']

    # Try multiple regex patterns for boxed notation
    patterns = [
        r'\\boxed\{([^}]+)\}',           # \boxed{A}
        r'\$\\boxed\{([^}]+)\}\$',       # $\boxed{A}$
        r'\\text\{boxed\}\{([^}]+)\}',   # \text{boxed}{A}
        r'boxed\{([^}]+)\}',             # boxed{A} (without backslash)
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            # Get the last match (most likely the final answer)
            content = matches[-1].strip().upper()

            # Extract just the letter from the content
            for char in content:
                if char in valid_choices:
                    return char

    # Fallback to regular letter extraction
    return extract_letter_answer(response, valid_choices)


def extract_letter_answer(response: str, valid_choices: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract a letter answer from model response (fallback method)

    Args:
        response: Raw model response
        valid_choices: List of valid choices (default: ['A', 'B', 'C', 'D'])

    Returns:
        Single letter answer or None if not found
    """
    if valid_choices is None:
        valid_choices = ['A', 'B', 'C', 'D']

    # Clean the response
    response = response.strip()

    # Try to find patterns indicating final answer (case insensitive)
    # Use \b for word boundaries to avoid partial matches
    final_answer_patterns = [
        r'final answer[:\s]+\b([A-Z])\b',       # "Final Answer: A" or "Final answer A"
        r'answer[:\s]+\b([A-Z])\b\s*$',         # "Answer: A" at end
        r'therefore[,:\s]+\b([A-Z])\b',         # "Therefore, A"
        r'conclusion[:\s]+\b([A-Z])\b',         # "Conclusion: A"
        r'the answer is[:\s]+\b([A-Z])\b',      # "The answer is A"
        r'^\s*([A-Z])\s*$',                     # Single letter on its own line
        r'\b([A-Z])\s*$',                       # Single letter at the very end
    ]

    for pattern in final_answer_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            # Get the last match and check if it's valid
            answer = matches[-1].strip().upper()
            if answer in valid_choices:
                return answer

    # NO "last resort" guessing - if we can't find a clear answer pattern, return None
    # This is important for accurate benchmarking - we shouldn't guess!
    return None


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
