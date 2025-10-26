"""
HumanEval specific utility functions

HumanEval is a dataset for evaluating code generation capabilities.
- 164 programming problems
- Python function completion tasks
- Requires code execution for evaluation
"""

from datasets import load_dataset
from typing import Dict, Tuple, Optional
import re
import sys
from io import StringIO
import contextlib
import signal


def load_humaneval_dataset(split: str = "test"):
    """
    Load HumanEval dataset

    Args:
        split: Dataset split (only 'test' available)

    Returns:
        Dataset object or None if error
    """
    try:
        dataset = load_dataset("openai/openai_humaneval", split=split)
        return dataset
    except Exception as e:
        print(f"Error loading HumanEval dataset: {e}")
        return None


def parse_humaneval_sample(sample: Dict) -> Tuple[str, str, str, str]:
    """
    Parse HumanEval sample

    Args:
        sample: HumanEval sample dictionary with fields:
            - task_id: str (e.g., "HumanEval/0")
            - prompt: str (function signature + docstring)
            - test: str (test cases)
            - entry_point: str (function name to test)

    Returns:
        Tuple of (task_id, prompt, test_code, entry_point)
    """
    task_id = sample['task_id']
    prompt = sample['prompt']
    test_code = sample['test']
    entry_point = sample['entry_point']

    return task_id, prompt, test_code, entry_point


def extract_code_from_response(response: str, entry_point: str) -> Optional[str]:
    """
    Extract Python code from model response

    Handles various formats:
    - Plain code
    - Code in markdown blocks (```python ... ```)
    - Code with explanations

    Args:
        response: Model response text
        entry_point: Expected function name

    Returns:
        Extracted code as string, or None if extraction fails
    """
    # Try to extract code from markdown blocks first
    code_block_pattern = r'```(?:python)?\n(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)

    if matches:
        # Use the first code block
        code = matches[0].strip()
        if entry_point in code:
            return code

    # If no markdown blocks, try to extract function definition
    # Look for "def entry_point" pattern
    func_pattern = rf'(def\s+{re.escape(entry_point)}\s*\(.*?\):.*?)(?=\n(?:def\s+|\Z))'
    match = re.search(func_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If still nothing, return the whole response (might be plain code)
    if entry_point in response:
        return response.strip()

    return None


class TimeoutError(Exception):
    """Custom exception for code execution timeout"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Code execution timed out")


@contextlib.contextmanager
def time_limit(seconds):
    """Context manager for limiting execution time"""
    if sys.platform == "win32":
        # Windows doesn't support SIGALRM, so skip timeout
        yield
    else:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)


def execute_code_safely(code: str, test_code: str, timeout_seconds: int = 5) -> Tuple[bool, str]:
    """
    Execute generated code with test cases in a safe manner

    Args:
        code: Generated function code
        test_code: Test cases to run
        timeout_seconds: Maximum execution time

    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    try:
        # Create isolated namespace
        namespace = {}

        # Execute the generated code
        with time_limit(timeout_seconds):
            exec(code, namespace)

            # Execute the test code
            exec(test_code, namespace)

        # If we got here, all tests passed
        return True, None

    except TimeoutError:
        return False, "Timeout: Code execution exceeded time limit"
    except AssertionError as e:
        return False, f"Test failed: {str(e)}"
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except Exception as e:
        return False, f"Runtime error: {type(e).__name__}: {str(e)}"


def get_dataset_stats(dataset) -> Dict:
    """
    Get statistics about the dataset

    Args:
        dataset: HumanEval dataset

    Returns:
        Dictionary with statistics
    """
    return {
        "total_samples": len(dataset),
    }


def format_humaneval_instruction() -> str:
    """
    Get custom instruction for HumanEval

    Returns:
        Instruction string
    """
    return (
        "Complete the following Python function.\n"
        "Provide only the function implementation, no explanations.\n"
        "Make sure your code is syntactically correct and handles all edge cases."
    )
