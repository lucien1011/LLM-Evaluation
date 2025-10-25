# LLM Evaluation Toolkit

A comprehensive, modular toolkit for evaluating large language models using Ollama. Supports MMLU, MMLU-Pro benchmarks, and includes powerful visualization tools.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **Two Comprehensive Benchmarks**
  - MMLU: 14,000 questions, 4 choices, 57 subjects
  - MMLU-Pro: 12,000+ questions, 10 choices, 14 categories

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

| Script | Purpose | Benchmark | Questions | Choices |
|--------|---------|-----------|-----------|---------|
| **mmlu_benchmark_modular.py** | MMLU evaluation | MMLU | 14,000 | 4 (A-D) |
| **mmlu_pro_benchmark.py** | MMLU-Pro evaluation | MMLU-Pro | 12,000+ | 10 (A-J) |
| **visualize_results.py** | Results visualization | Both | - | - |

### Utility Modules (Reusable!)

| Module | Purpose | Reusable For |
|--------|---------|--------------|
| **ollama_utils.py** | Ollama API interaction | Any Ollama project |
| **prompt_utils.py** | Prompt formatting | Any multiple-choice benchmark |
| **evaluation_utils.py** | Evaluation framework | Any dataset evaluation |
| **mmlu_utils.py** | MMLU dataset handling | MMLU benchmarks |
| **mmlu_pro_utils.py** | MMLU-Pro dataset handling | MMLU-Pro benchmarks |
| **visualization_utils.py** | Plotting utilities | Any result visualization |

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
│  - mmlu_benchmark_modular.py                                │
│  - mmlu_pro_benchmark.py                                    │
│  - visualize_results.py                                     │
└────────────────────┬────────────────────────────────────────┘
                     │ imports
┌────────────────────▼────────────────────────────────────────┐
│                  Reusable Utilities Layer                    │
│  - ollama_utils.py       (Ollama API)                       │
│  - prompt_utils.py       (Prompt formatting)                │
│  - evaluation_utils.py   (Evaluation framework)             │
│  - mmlu_utils.py         (MMLU specific)                    │
│  - mmlu_pro_utils.py     (MMLU-Pro specific)                │
│  - visualization_utils.py (Plotting)                        │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle**: Each utility module is independent and reusable across different projects!

## 📊 Benchmarks Comparison

| Feature | MMLU | MMLU-Pro |
|---------|------|----------|
| Questions | 14,000 | 12,000+ |
| Answer Choices | 4 (A-D) | 10 (A-J) |
| Organization | 57 subjects | 14 categories |
| Difficulty | Moderate | Hard |
| Random Baseline | 25% | 10% |
| Dataset | `cais/mmlu` | `TIGER-Lab/MMLU-Pro` |
| Published | 2020 | 2024 (NeurIPS) |

**Expected Performance**: Models typically score 15-20% lower on MMLU-Pro than MMLU.

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
# 1. Create your dataset utilities
from ollama_utils import query_ollama
from prompt_utils import format_multiple_choice_prompt, extract_letter_answer
from evaluation_utils import evaluate_dataset, save_results

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
├── Core Utilities
│   ├── ollama_utils.py
│   ├── prompt_utils.py
│   ├── evaluation_utils.py
│   ├── mmlu_utils.py
│   ├── mmlu_pro_utils.py
│   └── visualization_utils.py
│
├── Benchmark Scripts
│   ├── mmlu_benchmark_modular.py
│   ├── mmlu_pro_benchmark.py
│   └── visualize_results.py
│
├── Examples & Documentation
│   ├── example_custom_benchmark.py
│   ├── quick_start.py
│   ├── README.md (this file)
│   ├── README_MODULAR.md
│   ├── README_MMLU_PRO.md
│   ├── README_VISUALIZATION.md
│   ├── MMLU_vs_MMLU_PRO.md
│   └── PROJECT_STRUCTURE.md
│
├── Output Directories
│   ├── output/json/          (benchmark results)
│   └── output/plots/         (visualization plots)
│
├── Configuration
│   └── requirements.txt
│
└── Legacy
    └── mmlu_benchmark.py     (original class-based version)
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

### MMLU (4 choices)
- Random guessing: 25%
- Small models (<7B): 40-55%
- Medium models (7-13B): 55-65%
- Large models (30-70B): 65-75%
- GPT-4: 85-90%

### MMLU-Pro (10 choices)
- Random guessing: 10%
- Small models (<7B): 20-35%
- Medium models (7-13B): 35-50%
- Large models (30-70B): 45-60%
- GPT-4: 65-75%

## 💡 Tips for Best Results

1. **Start small**: Use `--max-samples 10` for quick testing
2. **Test specific domains**: Use `--subjects` or `--categories` to focus on areas of interest
3. **Compare systematically**: Keep test conditions consistent across models
4. **Use both benchmarks**: MMLU for comparison with literature, MMLU-Pro for harder evaluation
5. **Visualize results**: Use all plot types to understand model strengths/weaknesses

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
