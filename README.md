# LLM Evaluation Toolkit

A comprehensive, modular toolkit for evaluating large language models using Ollama. Supports 8 diverse benchmarks covering reasoning, math, coding, and truthfulness.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **Eight Comprehensive Benchmarks**
  - MMLU: 14,000 questions, 4 choices, 57 subjects (general knowledge)
  - MMLU-Pro: 12,000+ questions, 10 choices, 14 categories (harder reasoning)
  - TruthfulQA: 817 questions testing factual accuracy and truthfulness
  - ARC: 7,800 science questions with Easy/Challenge difficulty levels
  - HellaSwag: 70,000 commonsense reasoning questions
  - GPQA: 448 graduate-level science questions (expert knowledge)
  - GSM8K: 8,500 grade school math word problems
  - HumanEval: 164 Python programming problems with auto-evaluation

- **Modular Architecture**
  - Reusable utility modules
  - Easy to extend with new benchmarks
  - Clean separation of concerns

- **Powerful Visualization**
  - 4 plot types: bar charts, grouped comparisons, radar charts, heatmaps
  - Publication-ready 300 DPI output
  - Auto-detection and filtering

- **Production Ready**
  - Progress tracking with tqdm
  - Comprehensive error handling
  - Flexible CLI options
  - Detailed JSON output

## 📦 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd LLM-Evaluation

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running
ollama pull gemma3:latest
```

### Run Your First Benchmark

```bash
# Quick MMLU test (10 samples per subject)
python mmlu_benchmark_modular.py --model gemma3:latest --max-samples 10

# Quick MMLU-Pro test
python mmlu_pro_benchmark.py --model gemma3:latest --max-samples 10

# Visualize results
python visualize_results.py --input-dir output/json --plot-types overall
```

## 🎯 What's Inside

### Benchmark Scripts

| Script | Purpose | Benchmark | Questions | Format |
|--------|---------|-----------|-----------|--------|
| **mmlu_benchmark.py** | MMLU evaluation | MMLU | 14,000 | 4 choices (A-D) |
| **mmlu_pro_benchmark.py** | MMLU-Pro evaluation | MMLU-Pro | 12,000+ | 10 choices (A-J) |
| **truthfulqa_benchmark.py** | TruthfulQA evaluation | TruthfulQA | 817 | 4-5 choices |
| **arc_benchmark.py** | ARC evaluation | ARC | 7,800 | 4 choices (A-D) |
| **hellaswag_benchmark.py** | HellaSwag evaluation | HellaSwag | 70,000 | 4 choices (A-D) |
| **gpqa_benchmark.py** | GPQA evaluation | GPQA | 448 | 4 choices (A-D) |
| **gsm8k_benchmark.py** | GSM8K evaluation | GSM8K | 8,500 | Numerical answer |
| **humaneval_benchmark.py** | HumanEval evaluation | HumanEval | 164 | Code generation |
| **visualize_results.py** | Results visualization | All | - | - |

### Utility Modules (Reusable!)

Located in `utils/` directory:

| Module | Purpose | Reusable For |
|--------|---------|--------------|
| **utils/ollama.py** | Ollama API interaction | Any Ollama project |
| **utils/prompts.py** | Prompt formatting | Any multiple-choice benchmark |
| **utils/evaluation.py** | Evaluation framework | Any dataset evaluation |
| **utils/mmlu.py** | MMLU dataset handling | MMLU benchmarks |
| **utils/mmlu_pro.py** | MMLU-Pro dataset handling | MMLU-Pro benchmarks |
| **utils/truthfulqa.py** | TruthfulQA dataset handling | TruthfulQA benchmarks |
| **utils/arc.py** | ARC dataset handling | ARC benchmarks |
| **utils/hellaswag.py** | HellaSwag dataset handling | HellaSwag benchmarks |
| **utils/gpqa.py** | GPQA dataset handling | GPQA benchmarks |
| **utils/gsm8k.py** | GSM8K dataset + numerical extraction | Math benchmarks |
| **utils/humaneval.py** | HumanEval dataset + code execution | Code benchmarks |
| **utils/visualization.py** | Plotting utilities | Any result visualization |

### Example Scripts

| Script | Purpose |
|--------|---------|
| **example_custom_benchmark.py** | Template for custom benchmarks |
| **quick_start.py** | Usage pattern examples |

## 📚 Documentation

- **[README_MODULAR.md](README_MODULAR.md)** - Architecture and modular design
- **[README_MMLU_PRO.md](README_MMLU_PRO.md)** - MMLU-Pro benchmark guide
- **[README_VISUALIZATION.md](README_VISUALIZATION.md)** - Visualization documentation
- **[MMLU_vs_MMLU_PRO.md](MMLU_vs_MMLU_PRO.md)** - Benchmark comparison
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Complete project overview

## 🚀 Usage Examples

### Benchmark a Model on MMLU

```bash
# Quick test
python mmlu_benchmark_modular.py --model gemma3:latest --max-samples 10

# Specific subjects
python mmlu_benchmark_modular.py --model gemma3:latest \
    --subjects college_mathematics college_physics --max-samples 50

# Full evaluation (takes 2-4 hours)
python mmlu_benchmark_modular.py --model gemma3:latest
```

### Benchmark on MMLU-Pro (More Challenging)

```bash
# Quick test
python mmlu_pro_benchmark.py --model gemma3:latest --max-samples 10

# Specific categories
python mmlu_pro_benchmark.py --model gemma3:latest \
    --categories math physics chemistry --max-samples 50

# Full evaluation
python mmlu_pro_benchmark.py --model gemma3:latest
```

### Compare Multiple Models

```bash
# Benchmark different models
python mmlu_benchmark_modular.py --model gemma3:4b --output output/json/gemma_4b.json
python mmlu_benchmark_modular.py --model gemma3:27b --output output/json/gemma_27b.json

# Visualize comparison
python visualize_results.py --benchmark mmlu --plot-types overall subject radar heatmap
```

### Benchmark on TruthfulQA (Test Truthfulness)

```bash
# Quick test (MC1 format - single correct answer)
python truthfulqa_benchmark.py --model gemma3:latest --max-samples 10

# Test MC2 format (multiple correct answers)
python truthfulqa_benchmark.py --model gemma3:latest --format mc2 --max-samples 10

# Filter by specific categories
python truthfulqa_benchmark.py --model gemma3:latest --categories Health Law

# Full evaluation (817 questions)
python truthfulqa_benchmark.py --model gemma3:latest
```

### Benchmark on ARC (Science Reasoning)

```bash
# Quick test (Challenge difficulty)
python arc_benchmark.py --model gemma3:latest --max-samples 10

# Easy difficulty
python arc_benchmark.py --model gemma3:latest --difficulty ARC-Easy --max-samples 50

# Full Challenge evaluation
python arc_benchmark.py --model gemma3:latest --difficulty ARC-Challenge
```

### Benchmark on HellaSwag (Commonsense Reasoning)

```bash
# Quick test
python hellaswag_benchmark.py --model gemma3:latest --max-samples 10

# Full evaluation (70,000 questions - takes several hours)
python hellaswag_benchmark.py --model gemma3:latest
```

### Benchmark on GPQA (Graduate-Level Questions)

```bash
# Quick test (main subset)
python gpqa_benchmark.py --model gemma3:latest --max-samples 10

# Diamond subset (highest quality)
python gpqa_benchmark.py --model gemma3:latest --subset gpqa_diamond

# Extended subset (all questions)
python gpqa_benchmark.py --model gemma3:latest --subset gpqa_extended
```

### Benchmark on GSM8K (Math Word Problems)

```bash
# Quick test
python gsm8k_benchmark.py --model gemma3:latest --max-samples 10

# Full test evaluation (1,000 questions)
python gsm8k_benchmark.py --model gemma3:latest

# Train set evaluation
python gsm8k_benchmark.py --model gemma3:latest --split train --max-samples 100
```

### Benchmark on HumanEval (Code Generation)

```bash
# Quick test
python humaneval_benchmark.py --model gemma3:latest --max-samples 10

# Full evaluation (164 problems)
python humaneval_benchmark.py --model gemma3:latest
```

### Visualize Results

```bash
# All MMLU results
python visualize_results.py --benchmark mmlu --input-dir output/json

# Specific models only
python visualize_results.py --models "gemma3:4b" "gemma3:27b" --plot-types overall

# Generate all plot types
python visualize_results.py --input-dir output/json \
    --plot-types overall subject radar heatmap \
    --output-dir plots/
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Benchmark Scripts Layer                    │
│  - mmlu_benchmark.py                                        │
│  - mmlu_pro_benchmark.py                                    │
│  - visualize_results.py                                     │
└────────────────────┬────────────────────────────────────────┘
                     │ imports
┌────────────────────▼────────────────────────────────────────┐
│              Reusable Utilities Layer (utils/)               │
│  - utils/ollama.py        (Ollama API)                      │
│  - utils/prompts.py       (Prompt formatting)               │
│  - utils/evaluation.py    (Evaluation framework)            │
│  - utils/mmlu.py          (MMLU specific)                   │
│  - utils/mmlu_pro.py      (MMLU-Pro specific)               │
│  - utils/visualization.py (Plotting)                        │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle**: Each utility module is independent and reusable across different projects!

## 📊 Benchmarks Comparison

| Benchmark | Questions | Format | Focus Area | Difficulty | Random Baseline |
|-----------|-----------|--------|------------|------------|-----------------|
| **MMLU** | 14,000 | 4 choices | General knowledge (57 subjects) | Moderate | 25% |
| **MMLU-Pro** | 12,000+ | 10 choices | Advanced reasoning (14 categories) | Hard | 10% |
| **TruthfulQA** | 817 | 4-5 choices | Truthfulness & misconceptions | Moderate | 25% |
| **ARC** | 7,800 | 4 choices | Science reasoning | Easy/Challenge | 25% |
| **HellaSwag** | 70,000 | 4 choices | Commonsense reasoning | Moderate | 25% |
| **GPQA** | 448 | 4 choices | Graduate-level science | Very Hard | 25% |
| **GSM8K** | 8,500 | Numerical | Grade school math | Moderate | 0% |
| **HumanEval** | 164 | Code | Python programming | Hard | 0% |

**Performance Expectations**:
- Models typically score 15-20% lower on MMLU-Pro than MMLU
- GPQA is extremely challenging (even GPT-4 scores ~50%)
- GSM8K requires multi-step reasoning
- HumanEval tests practical coding ability (Pass@1 metric)

## 📈 Visualization Types

| Plot Type | Description | Use Case |
|-----------|-------------|----------|
| **Overall** | Bar chart of overall accuracy | Quick model comparison |
| **Subject** | Grouped bars per subject | Domain-specific analysis |
| **Radar** | Spider chart across subjects | Multi-dimensional view |
| **Heatmap** | Color-coded accuracy matrix | Detailed analysis |

All plots are publication-ready at 300 DPI.

## 🎨 Creating Custom Benchmarks

The modular design makes it easy to add new benchmarks:

```python
# 1. Import utilities
from utils.ollama import query_ollama
from utils.prompts import format_multiple_choice_prompt, extract_letter_answer
from utils.evaluation import evaluate_dataset, save_results

# 2. Define your evaluator
def my_evaluator(sample):
    question = sample['question']
    choices = sample['choices']
    correct = sample['answer']

    prompt = format_multiple_choice_prompt(question, choices)
    response = query_ollama(prompt, "gemma3:latest")
    predicted = extract_letter_answer(response)

    return predicted == correct, predicted, correct

# 3. Evaluate your dataset
results = evaluate_dataset(my_dataset, my_evaluator)
save_results(results, "my_results.json")
```

See [example_custom_benchmark.py](example_custom_benchmark.py) for a complete example.

## 🔧 Command-Line Reference

### MMLU Benchmark

```bash
python mmlu_benchmark_modular.py \
    --model MODEL_NAME \
    --subjects SUBJECT1 SUBJECT2 \
    --max-samples N \
    --split {test,validation,dev} \
    --output OUTPUT.json
```

### MMLU-Pro Benchmark

```bash
python mmlu_pro_benchmark.py \
    --model MODEL_NAME \
    --categories CAT1 CAT2 \
    --max-samples N \
    --split {test,validation} \
    --output OUTPUT.json
```

### Visualization

```bash
python visualize_results.py \
    --benchmark {mmlu,mmlu-pro} \
    --input-dir DIR \
    --output-dir DIR \
    --models MODEL1 MODEL2 \
    --plot-types overall subject radar heatmap \
    --filename BASENAME
```

## 📁 Project Structure

```
LLM-Evaluation/
├── utils/                   (Reusable utility modules)
│   ├── __init__.py
│   ├── ollama.py           (Ollama API interaction)
│   ├── prompts.py          (Prompt formatting)
│   ├── evaluation.py       (Evaluation framework)
│   ├── mmlu.py             (MMLU dataset handling)
│   ├── mmlu_pro.py         (MMLU-Pro dataset handling)
│   ├── truthfulqa.py       (TruthfulQA dataset handling)
│   ├── arc.py              (ARC dataset handling)
│   ├── hellaswag.py        (HellaSwag dataset handling)
│   ├── gpqa.py             (GPQA dataset handling)
│   ├── gsm8k.py            (GSM8K dataset + numerical extraction)
│   ├── humaneval.py        (HumanEval dataset + code execution)
│   └── visualization.py    (Plotting utilities)
│
├── Benchmark Scripts
│   ├── mmlu_benchmark.py
│   ├── mmlu_pro_benchmark.py
│   ├── truthfulqa_benchmark.py
│   ├── arc_benchmark.py
│   ├── hellaswag_benchmark.py
│   ├── gpqa_benchmark.py
│   ├── gsm8k_benchmark.py
│   ├── humaneval_benchmark.py
│   └── visualize_results.py
│
├── Output Directories
│   ├── output/json/        (benchmark results)
│   └── output/plots/       (visualization plots)
│
├── Configuration
│   ├── requirements.txt
│   └── README.md (this file)
```

## 🛠️ Requirements

- Python 3.8+
- Ollama installed and running
- Dependencies (installed via `pip install -r requirements.txt`):
  - datasets >= 2.14.0
  - requests >= 2.31.0
  - tqdm >= 4.66.0
  - matplotlib >= 3.7.0
  - numpy >= 1.24.0

## 🎓 Performance Expectations

### Multiple Choice Benchmarks

| Benchmark | Small (<7B) | Medium (7-13B) | Large (30-70B) | GPT-4 |
|-----------|-------------|----------------|----------------|-------|
| **MMLU** | 40-55% | 55-65% | 65-75% | 85-90% |
| **MMLU-Pro** | 20-35% | 35-50% | 45-60% | 65-75% |
| **TruthfulQA** | 30-45% | 45-60% | 55-70% | 75-85% |
| **ARC-Easy** | 60-75% | 75-85% | 85-95% | 95-98% |
| **ARC-Challenge** | 40-55% | 55-70% | 70-80% | 85-95% |
| **HellaSwag** | 50-65% | 65-80% | 80-90% | 90-95% |
| **GPQA** | 25-30% | 30-40% | 40-50% | 50-60% |

### Math & Code Benchmarks

| Benchmark | Small (<7B) | Medium (7-13B) | Large (30-70B) | GPT-4 |
|-----------|-------------|----------------|----------------|-------|
| **GSM8K** | 10-30% | 30-50% | 50-70% | 85-95% |
| **HumanEval** | 10-25% | 25-45% | 45-65% | 70-85% |

**Note**: Performance varies significantly based on model architecture, training data, and specific model versions.

## 💡 Tips for Best Results

1. **Start small**: Use `--max-samples 10` for quick testing
2. **Test specific domains**: Use `--subjects` or `--categories` to focus on areas of interest
3. **Compare systematically**: Keep test conditions consistent across models
4. **Mix benchmark types**: Combine multiple-choice (MMLU, ARC), math (GSM8K), and code (HumanEval) for comprehensive evaluation
5. **Watch for strengths**: Some models excel at reasoning, others at math or coding
6. **Visualize results**: Use all plot types to understand model strengths/weaknesses
7. **Be patient**: Full evaluations can take hours (especially HellaSwag with 70K questions)

## 🐛 Troubleshooting

### "Cannot connect to Ollama"
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama if needed
ollama serve
```

### "Model not found"
```bash
# List available models
ollama list

# Pull the model you need
ollama pull gemma3:latest
```

### "No results to plot"
```bash
# Verify JSON files exist
ls output/json/*.json

# Check file contents
python -c "import json; print(json.load(open('output/json/mmlu_results.json')))"
```

### Import errors
```bash
pip install -r requirements.txt
```

## 🤝 Contributing

This toolkit is designed to be easily extensible. To add a new benchmark:

1. Create `<benchmark>_utils.py` with dataset loading and parsing functions
2. Create `<benchmark>_benchmark.py` following the modular pattern
3. Reuse existing utilities (`ollama_utils`, `prompt_utils`, `evaluation_utils`)
4. Add documentation

See [README_MODULAR.md](README_MODULAR.md) for architectural details.

## 📝 Citation

If you use this toolkit in your research, please cite the original benchmark papers:

**MMLU**:
```bibtex
@article{hendrycks2021measuring,
  title={Measuring Massive Multitask Language Understanding},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and others},
  journal={ICLR},
  year={2021}
}
```

**MMLU-Pro**:
```bibtex
@article{wang2024mmlu,
  title={MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark},
  author={Wang, Yubo and Ma, Xueguang and Zhang, Ge and others},
  journal={NeurIPS},
  year={2024}
}
```

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- MMLU dataset from [Hendrycks et al.](https://github.com/hendrycks/test)
- MMLU-Pro dataset from [TIGER-AI-Lab](https://github.com/TIGER-AI-Lab/MMLU-Pro)
- Built for use with [Ollama](https://ollama.ai)

## 🔗 Links

- **MMLU Dataset**: [HuggingFace](https://huggingface.co/datasets/cais/mmlu)
- **MMLU-Pro Dataset**: [HuggingFace](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- **Ollama**: [Website](https://ollama.ai)

---

**Ready to start?** Jump to [Quick Start](#-quick-start) or check out the [examples](example_custom_benchmark.py)!

For questions or issues, please check the documentation files or create an issue on GitHub.
