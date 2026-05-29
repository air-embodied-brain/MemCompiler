# MemCompiler: Compile, Don't Inject -- State-Conditioned Memory for Embodied Agents


## 👋 Introduction
This repo is the official implementation of [***MemCompiler: Compile, Don't Inject -- State-Conditioned Memory for Embodied Agents***](https://arxiv.org/abs/2605.07594).

MemCompiler introduces a **state-conditioned memory compilation** approach for embodied agents. Instead of injecting raw memory into prompts, MemCompiler compiles past experiences into structured, state-aware memory representations that can be efficiently retrieved and applied. The system uses a **Manager-Executor architecture** where a Manager model orchestrates high-level planning while an Executor model handles low-level action execution, with a hierarchical memory system (Insight Graph, Query Graph, and Interaction Graph) supporting both components.

## 🌎 Setup

### Option 1: Using pip
```bash
conda create -n memcompiler python=3.12
conda activate memcompiler
pip install -r requirements.txt
```

### Option 2: Using conda environment file
```bash
conda env create -f environment.yml
conda activate memcompiler
```

## 🚀 Quick Start

### 🌳 Environments

We support two embodied environments:
- 🏠 [ALFWorld](https://github.com/alfworld/alfworld) — Text-based household robot tasks
- 🔬 [ScienceWorld](https://github.com/allenai/ScienceWorld) — Science experiment tasks

**ALFWorld Setup**: Download the ALFWorld data and update the paths in `tasks/env_configs/alfworld_config.yaml`.

**ScienceWorld Setup**: Install ScienceWorld following the [official instructions](https://github.com/allenai/ScienceWorld).

The data directory structure:
```
data
├── alfworld
│   ├── alfworld_tasks_suffix.json
│   ├── alfworld_tasks_train.json
│   └── alfworld_tasks_train_sampled_40_seed42.json
└── sciworld
    └── test.jsonl
```

### 🔑 API Keys Configuration

The Manager-Executor architecture supports multiple LLM backends. Configure API keys as environment variables before running:

```bash
# OpenAI API (when using GPT as Manager)
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Azure OpenAI (when using Azure as Manager)
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"

# Gemini (when using Gemini as Manager)
export GEMINI_API_KEY="your-gemini-api-key"
```

Or copy `template.env` to `.env` and fill in your keys.

### ⚙️ Configuration

Key parameters in the run scripts (`tasks/run_manager_executor.py` and `tasks/run_manager_executor_scienceworld.py`):

| Parameter | Description |
|-----------|-------------|
| `MANAGER_MODEL` | Manager LLM: `"gpt"`, `"qwen"`, or `"gemini"` |
| `API_TYPE` | API backend: `"openai"` or `"azure"` |
| `EXECUTOR_QWEN_MODEL_NAME` | Path to local Executor model |
| `MANAGER_QWEN_MODEL_NAME` | Path to local Manager model (when `MANAGER_MODEL="qwen"`) |
| `NUM_PROCESSES` | Number of parallel evaluation processes |
| `UPDATE_MEMORY` | Whether to update memory during execution |
| `OUTPUT_PATH` | Directory for output results |
| `MEMCOMPILER_PATH` | Path to the memory database |

### ▶️ How to Run

**ALFWorld:**
```bash
bash run_alfworld.sh
```
Or directly:
```bash
python tasks/run_manager_executor.py
```

**ScienceWorld:**
```bash
bash run_scienceworld.sh
```
Or directly:
```bash
python tasks/run_manager_executor_scienceworld.py
```

### 🧪 Evaluate Trained Models

To evaluate a trained Manager-Executor model (with soft token projection), use the evaluation script:

```bash
python tasks/eval_with_trained_model.py
```

Key configuration at the top of the script:

| Parameter | Description |
|-----------|-------------|
| `EVAL_TASK` | Which environment to evaluate: `"alfworld"` or `"sciworld"` |
| `LOAD_MODE` | Model loading mode: `"checkpoint"`, `"bin"`, or `"base"` |
| `CHECKPOINT_PATH` | Path to the training checkpoint directory |
| `MANAGER_BASE_MODEL` | Base model path for the Manager |
| `EXECUTOR_QWEN_MODEL_NAME` | Path to the Executor model |
| `USE_SOFT_TOKENS` | Enable/disable soft token projection (`True`/`False`) |
| `NUM_THOUGHT_TOKENS` | Number of soft tokens (must match training, default: 16) |
| `NUM_PROCESSES` / `GPU_ASSIGNMENTS` | Multi-GPU parallel evaluation |

This script loads trained model weights (including the soft token projection layer) and runs full evaluation episodes with the Manager-Executor pipeline.

## 📁 Project Structure
```
MemCompiler/
├── core/                        # Core modules
│   ├── llm.py                   # LLM interfaces (GPT, Gemini, Local)
│   ├── azure.py                 # Azure OpenAI utilities
│   ├── utils.py                 # Embedding, I/O utilities
│   └── memory/                  # Memory system
│       ├── common.py            # MASMessage, StateChain
│       └── core_memory/
│           ├── memcompiler.py   # Hierarchical memory implementation
│           ├── memory_base.py   # Base memory class
│           └── prompt.py        # Memory prompt templates
├── tasks/                       # Task runners
│   ├── run_manager_executor.py              # ALFWorld runner
│   ├── run_manager_executor_scienceworld.py # ScienceWorld runner
│   ├── eval_with_trained_model.py           # Evaluate trained models
│   ├── prompts/                 # Task-specific prompts
│   ├── envs/                    # Environment wrappers
│   └── env_configs/             # Environment configurations
├── data/                        # Task datasets
├── run_alfworld.sh              # ALFWorld launch script
├── run_scienceworld.sh          # ScienceWorld launch script
└── requirements.txt
```

## 🫡 Citation
If you find this repository helpful, a citation to our paper would be greatly appreciated:
```bibtex
@misc{ding2026memcompilercompiledontinject,
      title={MemCompiler: Compile, Don't Inject -- State-Conditioned Memory for Embodied Agents}, 
      author={Xin Ding and Xinrui Wang and Yifan Yang and Hao Wu and Shiqi Jiang and Qianxi Zhang and Liang Mi and Hanxin Zhu and Kun Li and Yunxin Liu and Zhibo Chen and Ting Cao},
      year={2026},
      eprint={2605.07594},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.07594}, 
}
```

## 🙏 Acknowledgement
- This codebase is built upon [G-Memory](https://github.com/bingreeky/GMemory). We sincerely thank the authors for their excellent work on hierarchical memory for multi-agent systems.
