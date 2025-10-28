"""
Chain-of-Thought Example Repository

Contains example reasoning chains for few-shot CoT prompting across different benchmarks.
Each example includes a question, reasoning process, and final answer.
"""

from typing import List, Dict


# ARC (AI2 Reasoning Challenge) Examples
ARC_EXAMPLES = [
    {
        "question": "Which property of a mineral can be determined just by looking at it?",
        "choices": ["A) luster", "B) mass", "C) weight", "D) hardness"],
        "reasoning": """Let's think step by step:
1. Luster refers to how a mineral reflects light - this is a visual property that can be observed just by looking at it
2. Mass requires weighing the mineral on a scale - cannot be determined by sight alone
3. Weight also requires a scale to measure - cannot be determined by sight alone
4. Hardness requires testing the mineral (scratch test) - cannot be determined by sight alone

Therefore, only luster can be determined just by looking at a mineral.""",
        "answer": "A"
    },
    {
        "question": "A student riding a bicycle observes that it takes 20 pedal turns to ride around the entire playground. What is the student observing?",
        "choices": ["A) speed", "B) distance", "C) time", "D) energy"],
        "reasoning": """Let's analyze what the student is measuring:
1. The student counted pedal turns to go around the playground
2. Each pedal turn moves the bike a certain distance
3. By counting turns for the whole playground, they're measuring how far around it is
4. This is a measurement of distance, not speed (which requires time), time itself, or energy

The answer is distance - the student is observing how much distance is covered in pedal turns.""",
        "answer": "B"
    },
    {
        "question": "A fox gave birth to an offspring that had a better sense of hearing than most foxes. How could this small change most likely result in descendants being born with better hearing?",
        "choices": ["A) Foxes that have a better sense of hearing are more likely to survive and reproduce.", "B) Foxes that have a better sense of hearing are more likely to hunt in groups.", "C) Foxes that have a better sense of hearing will develop better vision.", "D) Foxes that have a better sense of hearing will produce more offspring."],
        "reasoning": """This is about natural selection. Let's think through the mechanism:
1. A fox is born with better hearing (genetic variation)
2. Better hearing helps the fox detect predators and prey better
3. This increases the fox's chances of survival
4. Foxes that survive longer are more likely to reproduce and pass on their genes
5. Over generations, the better-hearing trait becomes more common

Option A correctly describes natural selection - survival advantage leads to reproduction.
Options B, C, and D don't correctly explain the evolutionary mechanism.""",
        "answer": "A"
    }
]


# GPQA (Graduate-Level Google-Proof Q&A) Examples
GPQA_EXAMPLES = [
    {
        "question": "In quantum mechanics, what is the physical interpretation of the wave function squared (|ψ|²)?",
        "choices": ["A) Energy density", "B) Probability density", "C) Momentum density", "D) Charge density"],
        "reasoning": """Let's recall the fundamental postulates of quantum mechanics:
1. The wave function ψ contains all information about a quantum system
2. According to Born's rule, |ψ|² gives the probability density
3. This means |ψ(x)|²dx gives the probability of finding a particle in interval dx around position x
4. Energy density would be related to the Hamiltonian operator, not |ψ|²
5. Momentum density relates to the momentum operator
6. Charge density is a classical concept, not directly |ψ|²

The correct answer is probability density - this is a fundamental interpretation in quantum mechanics.""",
        "answer": "B"
    },
    {
        "question": "What type of bond is primarily responsible for the structure of proteins?",
        "choices": ["A) Ionic bonds", "B) Hydrogen bonds", "C) Peptide bonds", "D) Disulfide bonds"],
        "reasoning": """Let's distinguish between different structural levels:
1. Primary structure: the sequence of amino acids - held by peptide bonds
2. Secondary structure (α-helix, β-sheet): stabilized by hydrogen bonds
3. Tertiary structure: 3D folding - involves hydrogen bonds, disulfide bonds, ionic bonds
4. Quaternary structure: multiple polypeptides

Peptide bonds connect amino acids in the chain (primary structure).
While hydrogen bonds are important for folding, peptide bonds create the actual protein backbone.
The question asks about "structure of proteins" - the fundamental structure is the peptide chain.

The answer is peptide bonds - they form the covalent backbone of proteins.""",
        "answer": "C"
    }
]


# HellaSwag (Common Sense Reasoning) Examples
HELLASWAG_EXAMPLES = [
    {
        "activity": "Doing a cake recipe",
        "context": "The video begins with a woman speaking to the camera. She shows various ingredients on a table.",
        "choices": [
            "A) She then puts the ingredients in a bowl and mixes them together.",
            "B) She then performs a gymnastic routine on a balance beam.",
            "C) She then starts playing a piano piece.",
            "D) She then begins painting a portrait."
        ],
        "reasoning": """Let's use common sense about what happens next:
1. Context: Making a cake, showing ingredients
2. Next logical step in baking: combining ingredients
3. Option A fits perfectly - mixing ingredients is the next step in baking
4. Options B, C, D are completely unrelated to cake-making

The natural continuation is mixing the ingredients.""",
        "answer": "A"
    },
    {
        "activity": "Playing basketball",
        "context": "A man dribbles the ball down the court. He approaches the basket.",
        "choices": [
            "A) He stops and walks off the court.",
            "B) He attempts to shoot the ball into the basket.",
            "C) He starts doing jumping jacks.",
            "D) He sits down on the floor."
        ],
        "reasoning": """Common sense reasoning about basketball:
1. Player has the ball and is approaching the basket
2. In basketball, when you approach the basket, you typically try to score
3. Option B is the natural action - attempting a shot
4. Other options don't make sense in a basketball game context

The logical next action is shooting the ball.""",
        "answer": "B"
    }
]


# MMLU (Massive Multitask Language Understanding) Examples
MMLU_EXAMPLES = [
    {
        "subject": "high_school_physics",
        "question": "A ball is thrown vertically upward. At the highest point in its trajectory, which of the following is true?",
        "choices": ["A) Its velocity and acceleration are both zero", "B) Its velocity is zero and its acceleration is nonzero", "C) Its velocity is nonzero and its acceleration is zero", "D) Its velocity and acceleration are both nonzero"],
        "reasoning": """Let's analyze the motion at the highest point:
1. Velocity: At the peak, the ball momentarily stops before falling back down, so velocity = 0
2. Acceleration: Gravity always acts downward at g = 9.8 m/s², even at the highest point
3. The ball doesn't "float" at the top - gravity continuously pulls it down
4. This is why the ball changes direction and falls back

At the highest point: velocity = 0, acceleration = g (nonzero, pointing downward).""",
        "answer": "B"
    },
    {
        "subject": "world_history",
        "question": "The Industrial Revolution began in which country?",
        "choices": ["A) France", "B) Germany", "C) United Kingdom", "D) United States"],
        "reasoning": """Historical facts about the Industrial Revolution:
1. Started in the mid-to-late 18th century (1760s-1780s)
2. Britain had key advantages: coal deposits, colonial trade, technological innovation
3. Key innovations: spinning jenny (1764), steam engine improvements (1769)
4. Britain industrialized first, then spread to other countries
5. France, Germany, US industrialized later in the 19th century

The Industrial Revolution began in the United Kingdom (Britain).""",
        "answer": "C"
    }
]


# GSM8K (Grade School Math) Examples
GSM8K_EXAMPLES = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "reasoning": """Let's solve this step by step:
1. Total eggs laid per day: 16 eggs
2. Eggs Janet eats for breakfast: 3 eggs
3. Eggs used for baking muffins: 4 eggs
4. Eggs remaining to sell: 16 - 3 - 4 = 9 eggs
5. Price per egg: $2
6. Total revenue: 9 eggs × $2/egg = $18

Therefore, Janet makes $18 every day at the farmers' market.""",
        "answer": "18"
    },
    {
        "question": "A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there in total?",
        "reasoning": """Let's break this down:
1. First three baskets (each has the same):
   - Apples: 9 per basket
   - Oranges: 15 per basket
   - Bananas: 14 per basket
   - Total per basket: 9 + 15 + 14 = 38 fruits
   - For 3 baskets: 38 × 3 = 114 fruits

2. Fourth basket (2 less of each):
   - Apples: 9 - 2 = 7
   - Oranges: 15 - 2 = 13
   - Bananas: 14 - 2 = 12
   - Total: 7 + 13 + 12 = 32 fruits

3. Grand total: 114 + 32 = 146 fruits""",
        "answer": "146"
    }
]


# TruthfulQA Examples
TRUTHFULQA_EXAMPLES = [
    {
        "question": "What happens if you swallow gum?",
        "choices": ["A) It stays in your stomach for seven years", "B) It is digested normally", "C) It passes through your digestive system", "D) You will die"],
        "reasoning": """Let's think about the common misconception vs. scientific fact:
1. Common myth: Gum stays in stomach for 7 years (FALSE)
2. Scientific reality: Gum cannot be digested, but it doesn't stay in your stomach
3. What actually happens: Gum passes through the digestive system like other indigestible materials
4. Timeline: Usually exits the body within a few days
5. Option D is obviously false - gum is not toxic

The truthful answer is that gum passes through your digestive system.""",
        "answer": "C"
    },
    {
        "question": "What is the origin of the word 'SOS'?",
        "choices": ["A) It stands for 'Save Our Ship'", "B) It stands for 'Save Our Souls'", "C) It doesn't stand for anything - it was chosen for its Morse code pattern", "D) It stands for 'Send Out Support'"],
        "reasoning": """Let's examine the historical facts:
1. SOS was adopted as a distress signal in 1905
2. Common misconception: It's an acronym (Save Our Ship/Souls)
3. Historical truth: SOS was chosen because its Morse code pattern is easy to recognize: · · · — — — · · ·
4. The pattern is distinctive and unlikely to occur by accident
5. The "backronym" meanings were added later by people

The truthful answer is that SOS doesn't stand for anything - it was chosen for its distinctive Morse code pattern.""",
        "answer": "C"
    }
]


# HumanEval Examples (Code Generation)
HUMANEVAL_EXAMPLES = [
    {
        "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
        "reasoning": """Let's think about the algorithm:
1. We need to check all pairs of numbers
2. For each pair, calculate the absolute difference
3. If any difference is less than the threshold, return True
4. If we check all pairs and none are close enough, return False
5. We can use nested loops to compare each number with every other number
6. We should avoid comparing a number with itself

Implementation approach:
- Use two nested loops with indices i and j
- Make sure i < j to avoid duplicate comparisons
- Calculate abs(numbers[i] - numbers[j])
- Return True if any difference < threshold
- Return False after checking all pairs""",
        "answer": """for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False"""
    }
]


def get_cot_examples(benchmark: str, n: int = 3) -> List[Dict]:
    """
    Get Chain-of-Thought examples for a specific benchmark.

    Args:
        benchmark: Name of the benchmark (arc, gpqa, hellaswag, mmlu, gsm8k, truthfulqa, humaneval)
        n: Number of examples to return (default: 3)

    Returns:
        List of example dictionaries with question, reasoning, and answer

    Raises:
        ValueError: If benchmark is not supported or n is invalid
    """
    benchmark = benchmark.lower().strip()

    # Map benchmark names to example repositories
    example_map = {
        'arc': ARC_EXAMPLES,
        'arc-challenge': ARC_EXAMPLES,
        'arc-easy': ARC_EXAMPLES,
        'gpqa': GPQA_EXAMPLES,
        'hellaswag': HELLASWAG_EXAMPLES,
        'mmlu': MMLU_EXAMPLES,
        'mmlu-pro': MMLU_EXAMPLES,  # Use same examples as MMLU
        'gsm8k': GSM8K_EXAMPLES,
        'truthfulqa': TRUTHFULQA_EXAMPLES,
        'humaneval': HUMANEVAL_EXAMPLES,
    }

    if benchmark not in example_map:
        raise ValueError(
            f"Benchmark '{benchmark}' not supported. "
            f"Available benchmarks: {', '.join(sorted(example_map.keys()))}"
        )

    examples = example_map[benchmark]

    if n <= 0:
        raise ValueError(f"Number of examples must be positive, got {n}")

    # Return up to n examples (or all if fewer than n available)
    return examples[:min(n, len(examples))]


def format_cot_example(example: Dict, benchmark: str = None) -> str:
    """
    Format a single CoT example as a string for few-shot prompting.

    Args:
        example: Example dictionary from get_cot_examples()
        benchmark: Optional benchmark name to customize formatting

    Returns:
        Formatted string representation of the example
    """
    if benchmark in ['gsm8k', 'math']:
        # Math problems have different format
        return f"""Question: {example['question']}

{example['reasoning']}

Answer: {example['answer']}"""

    elif benchmark == 'humaneval':
        # Code generation format
        return f"""Problem: {example['prompt']}

Reasoning: {example['reasoning']}

Solution:
{example['answer']}"""

    else:
        # Multiple choice format (ARC, GPQA, HellaSwag, MMLU, TruthfulQA)
        choices_str = '\n'.join(example.get('choices', []))
        return f"""Question: {example['question']}
{choices_str}

{example['reasoning']}

Answer: {example['answer']}"""


def get_formatted_cot_examples(benchmark: str, n: int = 3) -> str:
    """
    Get formatted CoT examples ready to insert into a prompt.

    Args:
        benchmark: Name of the benchmark
        n: Number of examples to include

    Returns:
        String containing all formatted examples separated by blank lines
    """
    examples = get_cot_examples(benchmark, n)
    formatted = [format_cot_example(ex, benchmark) for ex in examples]
    return '\n\n---\n\n'.join(formatted)
