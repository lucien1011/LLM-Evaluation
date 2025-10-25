"""
MMLU-specific utility functions
"""

from datasets import load_dataset
from typing import Dict, List


# Default MMLU subjects covering various domains
DEFAULT_MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
    "moral_scenarios",
    "philosophy",
    "professional_medicine",
    "world_religions"
]


def load_mmlu_subject(subject: str, split: str = "test"):
    """
    Load a specific MMLU subject dataset

    Args:
        subject: MMLU subject name
        split: Dataset split ('test', 'validation', 'dev')

    Returns:
        Dataset object or None if error
    """
    try:
        dataset = load_dataset("cais/mmlu", subject, split=split)
        return dataset
    except Exception as e:
        print(f"Error loading dataset for {subject}: {e}")
        return None


def get_all_mmlu_subjects() -> List[str]:
    """
    Get list of all available MMLU subjects

    Returns:
        List of subject names
    """
    # This is the complete list of 57 MMLU subjects
    return [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes",
        "moral_scenarios", "nutrition", "philosophy",
        "prehistory", "professional_accounting", "professional_law",
        "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology",
        "us_foreign_policy", "virology", "world_religions"
    ]


def parse_mmlu_sample(sample: Dict) -> tuple:
    """
    Parse MMLU sample to extract question, choices, and answer

    Args:
        sample: MMLU sample dictionary

    Returns:
        Tuple of (question, choices, correct_answer_letter)
    """
    question = sample['question']
    choices = sample['choices']
    correct_answer = sample['answer']

    # Convert answer index to letter (0->A, 1->B, 2->C, 3->D)
    if isinstance(correct_answer, int):
        from .prompts import index_to_letter
        correct_letter = index_to_letter(correct_answer)
    else:
        correct_letter = correct_answer.upper()

    return question, choices, correct_letter
