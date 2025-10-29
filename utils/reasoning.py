"""
Reasoning strategies for LLM evaluation

This module provides a framework for different reasoning approaches:
- Direct: Answer directly without reasoning (current default)
- Zero-Shot CoT: "Let's think step by step"
- Few-Shot CoT: With example reasoning chains
- Self-Consistency: Multiple samples with majority voting
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
import re
from collections import Counter


class ReasoningStrategy(ABC):
    """Base class for reasoning strategies"""

    def __init__(self, max_tokens: Optional[int] = None):
        """
        Initialize reasoning strategy

        Args:
            max_tokens: Maximum tokens for model response (default: 32000)
        """
        self.max_tokens = max_tokens if max_tokens is not None else 32000

    @abstractmethod
    def format_prompt(self, question: str, choices: List[str],
                     instruction: str = None) -> str:
        """
        Format the prompt according to this reasoning strategy

        Args:
            question: The question text
            choices: List of answer choices
            instruction: Optional instruction prefix

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def extract_answer(self, response: str, valid_choices: List[str]) -> Optional[str]:
        """
        Extract the final answer from model response

        Args:
            response: Model's text response
            valid_choices: List of valid answer choices (e.g., ['A', 'B', 'C', 'D'])

        Returns:
            Extracted answer letter or None if extraction failed
        """
        pass

    def get_max_tokens(self) -> int:
        """Return configured max_tokens for this strategy"""
        return self.max_tokens

    def get_temperature(self) -> float:
        """Return recommended temperature for this strategy"""
        return 0.0

    def requires_multiple_samples(self) -> bool:
        """Whether this strategy requires multiple model calls"""
        return False

    def get_num_samples(self) -> int:
        """Number of samples needed (for self-consistency)"""
        return 1


class DirectStrategy(ReasoningStrategy):
    """
    Direct answer without reasoning (current default approach)

    Prompts the model to answer immediately with just the letter.
    """

    def __init__(self, max_tokens: Optional[int] = None):
        """
        Args:
            max_tokens: Maximum tokens for model response (default: 32000)
        """
        super().__init__(max_tokens)

    def format_prompt(self, question: str, choices: List[str],
                     instruction: str = None) -> str:
        from .prompts import format_multiple_choice_prompt
        return format_multiple_choice_prompt(question, choices, instruction, use_boxed=True)

    def extract_answer(self, response: str, valid_choices: List[str]) -> Optional[str]:
        from .prompts import extract_boxed_answer
        return extract_boxed_answer(response, valid_choices)

    def get_temperature(self) -> float:
        return 0.0


class ZeroShotCoTStrategy(ReasoningStrategy):
    """
    Zero-shot Chain-of-Thought reasoning

    Adds "Let's think step by step" prompt to encourage reasoning before answering.
    Based on: Kojima et al. "Large Language Models are Zero-Shot Reasoners" (2022)
    """

    def __init__(self, cot_trigger: str = "Let's think step by step:", max_tokens: Optional[int] = None):
        """
        Args:
            cot_trigger: The phrase that triggers CoT reasoning
            max_tokens: Maximum tokens for model response (default: 32000)
        """
        super().__init__(max_tokens)
        self.cot_trigger = cot_trigger

    def format_prompt(self, question: str, choices: List[str],
                     instruction: str = None) -> str:
        from .prompts import format_multiple_choice_prompt

        # Modify instruction to encourage reasoning with boxed answer
        if instruction is None:
            instruction = (
                "Think through this problem step by step.\n"
                "After your reasoning, provide your final answer in the format: \\boxed{X} where X is the letter of your choice."
            )
        else:
            # Ensure boxed notation instruction is included
            if "boxed" not in instruction.lower():
                instruction = instruction + "\nProvide your final answer in the format: \\boxed{X} where X is the letter of your choice."

        base_prompt = format_multiple_choice_prompt(question, choices, instruction, use_boxed=True)
        return f"{base_prompt}\n\n{self.cot_trigger}"

    def extract_answer(self, response: str, valid_choices: List[str]) -> Optional[str]:
        """
        Extract answer from CoT response
        Prioritizes boxed notation, then falls back to pattern matching
        """
        from .prompts import extract_boxed_answer
        return extract_boxed_answer(response, valid_choices)

    def get_temperature(self) -> float:
        return 0.0


class FewShotCoTStrategy(ReasoningStrategy):
    """
    Few-shot Chain-of-Thought reasoning

    Provides example questions with reasoning chains to guide the model.
    Based on: Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in LLMs" (2022)
    """

    def __init__(self, examples: List[Dict[str, str]], max_tokens: Optional[int] = None):
        """
        Args:
            examples: List of dicts with keys:
                - question: str
                - choices: List[str]
                - reasoning: str (the step-by-step reasoning)
                - answer: str (the final answer letter)
            max_tokens: Maximum tokens for model response (default: 32000)
        """
        super().__init__(max_tokens)
        self.examples = examples

    def format_prompt(self, question: str, choices: List[str],
                     instruction: str = None) -> str:
        from .prompts import format_multiple_choice_prompt

        prompt_parts = []

        # Add instruction if provided
        if instruction:
            # Modify to encourage reasoning
            instruction = re.sub(
                r'only respond with.*?\.',
                'Think through the problem step by step, then provide your answer.',
                instruction,
                flags=re.IGNORECASE
            )
            prompt_parts.append(instruction)
            prompt_parts.append("")

        # Add few-shot examples
        for i, ex in enumerate(self.examples, 1):
            # Format example question
            ex_prompt = format_multiple_choice_prompt(
                ex['question'],
                ex['choices'],
                instruction=None
            )
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(ex_prompt)
            prompt_parts.append(f"\nReasoning: {ex['reasoning']}")
            prompt_parts.append(f"Answer: {ex['answer']}")
            prompt_parts.append("")

        # Add actual question
        prompt_parts.append("Now answer this question:")
        actual_prompt = format_multiple_choice_prompt(question, choices, instruction=None)
        prompt_parts.append(actual_prompt)
        prompt_parts.append("\nReasoning:")

        return "\n".join(prompt_parts)

    def extract_answer(self, response: str, valid_choices: List[str]) -> Optional[str]:
        # Use same extraction as ZeroShotCoT
        return ZeroShotCoTStrategy().extract_answer(response, valid_choices)

    def get_temperature(self) -> float:
        return 0.0


class SelfConsistencyStrategy(ReasoningStrategy):
    """
    Self-consistency with Chain-of-Thought

    Samples multiple reasoning paths and selects the most consistent answer via majority vote.
    Based on: Wang et al. "Self-Consistency Improves Chain of Thought Reasoning" (2022)
    """

    def __init__(self, base_strategy: ReasoningStrategy, n_samples: int = 5):
        """
        Args:
            base_strategy: Underlying CoT strategy (typically ZeroShotCoT or FewShotCoT)
            n_samples: Number of reasoning paths to sample
        """
        self.base_strategy = base_strategy
        self.n_samples = n_samples

    def format_prompt(self, question: str, choices: List[str],
                     instruction: str = None) -> str:
        # Use the base strategy's prompt formatting
        return self.base_strategy.format_prompt(question, choices, instruction)

    def extract_answer(self, response: str, valid_choices: List[str]) -> Optional[str]:
        """
        For self-consistency, this should not be called directly.
        Use extract_answer_from_multiple() instead.
        """
        return self.base_strategy.extract_answer(response, valid_choices)

    def extract_answer_from_multiple(self, responses: List[str],
                                     valid_choices: List[str]) -> Optional[str]:
        """
        Extract answers from multiple responses and perform majority voting

        Args:
            responses: List of model responses (one per sample)
            valid_choices: Valid answer choices

        Returns:
            Most common answer via majority vote, or None if no valid answers
        """
        answers = []
        for response in responses:
            answer = self.base_strategy.extract_answer(response, valid_choices)
            if answer:
                answers.append(answer)

        if not answers:
            return None

        # Majority voting
        vote_counts = Counter(answers)
        most_common = vote_counts.most_common(1)[0]
        return most_common[0]

    def get_voting_details(self, responses: List[str],
                          valid_choices: List[str]) -> Dict[str, int]:
        """
        Get detailed voting breakdown

        Returns:
            Dictionary mapping answer -> vote count
        """
        answers = []
        for response in responses:
            answer = self.base_strategy.extract_answer(response, valid_choices)
            if answer:
                answers.append(answer)

        return dict(Counter(answers))

    def get_max_tokens(self) -> int:
        return self.base_strategy.get_max_tokens()

    def get_temperature(self) -> float:
        # Use higher temperature for diversity in reasoning paths
        return 0.7

    def requires_multiple_samples(self) -> bool:
        return True

    def get_num_samples(self) -> int:
        return self.n_samples


def create_strategy(strategy_name: str, **kwargs) -> ReasoningStrategy:
    """
    Factory function to create reasoning strategies

    Args:
        strategy_name: One of 'direct', 'zero-shot-cot', 'few-shot-cot', 'self-consistency'
        **kwargs: Strategy-specific arguments
            - max_tokens: Maximum tokens for model response (default: 32000)
            - For few-shot-cot: examples (List[Dict])
            - For self-consistency: base_strategy (str), n_samples (int)

    Returns:
        ReasoningStrategy instance

    Examples:
        >>> strategy = create_strategy('direct', max_tokens=16000)
        >>> strategy = create_strategy('zero-shot-cot', max_tokens=32000)
        >>> strategy = create_strategy('few-shot-cot', examples=[...], max_tokens=32000)
        >>> strategy = create_strategy('self-consistency', base_strategy='zero-shot-cot', n_samples=5, max_tokens=32000)
    """
    max_tokens = kwargs.get('max_tokens', None)  # None will use default 32000

    if strategy_name == 'direct':
        return DirectStrategy(max_tokens=max_tokens)

    elif strategy_name == 'zero-shot-cot':
        cot_trigger = kwargs.get('cot_trigger', "Let's think step by step:")
        return ZeroShotCoTStrategy(cot_trigger=cot_trigger, max_tokens=max_tokens)

    elif strategy_name == 'few-shot-cot':
        examples = kwargs.get('examples', [])
        if not examples:
            raise ValueError("few-shot-cot requires 'examples' argument")
        return FewShotCoTStrategy(examples, max_tokens=max_tokens)

    elif strategy_name == 'self-consistency':
        base_name = kwargs.get('base_strategy', 'zero-shot-cot')
        n_samples = kwargs.get('n_samples', 5)

        # Recursively create base strategy (pass max_tokens to base)
        base_kwargs = {k: v for k, v in kwargs.items()
                      if k not in ['base_strategy', 'n_samples']}
        base_strategy = create_strategy(base_name, **base_kwargs)

        return SelfConsistencyStrategy(base_strategy, n_samples)

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Choose from: direct, zero-shot-cot, few-shot-cot, self-consistency")
