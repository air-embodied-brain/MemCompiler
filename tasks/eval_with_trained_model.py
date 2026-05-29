#!/usr/bin/env python
"""
Evaluation script with trained Manager-Executor model.
Supports both ALFWorld and ScienceWorld environments.

Usage:
    python tasks/eval_with_trained_model.py

Prerequisites:
    1. Set config: tasks/env_configs/alfworld_config.yaml
    2. Start ChromaDB server for Memcompiler memory
    3. Set EVAL_TASK, model paths, and API keys below
"""

import numpy as np
import json
import os
import re
import sys
import time
import torch
import fcntl
import traceback
import multiprocessing as mp
from typing import List, Optional, Dict, Any, Tuple
import torch.nn as nn

# =============================================================================
# Configuration
# =============================================================================
# Global task switch: "alfworld" or "sciworld"
EVAL_TASK = "sciworld"

NUM_PROCESSES = 1
NUM_GPUS = 1
GPU_ASSIGNMENTS = [0]

# --- ALFWorld Configuration ---
OUTPUT_PATH = "./output_eval_epoch3_7Bins_16token_1e-5_unseen_gm3flash"
MEMCOMPILER_PATH = "./.db_eval_epoch3_7Bins_16token_1e-5_unseen_gm3flash/memcompiler"
CONFIG_FILE = "tasks/env_configs/alfworld_config.yaml"

# --- ScienceWorld Configuration ---
SCIWORLD_OUTPUT_PATH = "./output_eval_epoch3_7Bins_16token_1e-5_unseen_sci_gm3flash"
SCIWORLD_MEMCOMPILER_PATH = "./.db_eval_epoch3_7Bins_16token_1e-5_unseen_sci_gm3flash/memcompiler"
SCIWORLD_CHROMA_PORT = 8075  # Independent from ALFWorld's CHROMA_PORT
SCIWORLD_TEST_JSONL_PATH = "./data/sciworld/test.jsonl"
SCIWORLD_MAX_STEPS = 30
SENT_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ACTION_MATCH_THRESHOLD = 0.5

# Executor model - changed to 7B for soft token integration
EXECUTOR_QWEN_MODEL_NAME = "YOUR_EXECUTOR_MODEL_PATH"

# GPT Model Configuration for Memcompiler internal LLM calls
GPT_MODEL = "gpt-5.2"
API_TYPE = "gemini"  # "azure", "openai", or "gemini"
API_MAX_RETRIES = 3
API_RETRY_DELAY = 2

AZURE_OPENAI_CONFIG = {
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY", ""),
    "endpoints": {
        "gpt-5": {
            "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            "api_version": "2025-01-01-preview",
        },
    }
}

# Gemini configuration (used when API_TYPE = "gemini")
# Get API key from: https://aistudio.google.com/apikey
GEMINI_CONFIG = {
    "api_key": os.environ.get("GEMINI_API_KEY", ""),  # or hardcode here
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "model": "gemini-3-flash-preview",  # or "gemini-2.5-pro", "gemini-2.0-flash", etc.
}

# Manager model - VL model base
MANAGER_BASE_MODEL = "YOUR_MANAGER_MODEL_PATH"

# =============================================================================
# Model Loading Mode Configuration
# =============================================================================
# LOAD_MODE: "bin", "checkpoint", or "base"
#   - "bin": Load from separate .bin files (assistant_model.bin + projection.bin)
#   - "checkpoint": Load from training checkpoint (pytorch_model.bin)
#   - "base": Use base model directly without loading fine-tuned weights
LOAD_MODE = "checkpoint"  # Options: "bin", "checkpoint", "base"

# For LOAD_MODE = "bin": paths to separate .bin files
MANAGER_WEIGHTS_PATH = "YOUR_MANAGER_WEIGHTS_PATH"
PROJECTION_WEIGHTS_PATH = "YOUR_PROJECTION_WEIGHTS_PATH"

# For LOAD_MODE = "checkpoint": path to checkpoint directory
CHECKPOINT_PATH = "YOUR_CHECKPOINT_PATH"


# Number of soft tokens (must match training)
NUM_THOUGHT_TOKENS = 16

# Ablation switch:
# - False: disable soft token projection/injection (text-only manager guidance experiment).
# - True: enable normal soft-token experiment (requires a projection checkpoint whose output dim
#         matches the executor hidden size).
USE_SOFT_TOKENS = True
LATENT_SAMPLE_BASE_SEED = 3407

MAX_STEPS = 30
USE_VISION = False  # Text environment, no vision
ENABLE_STEP_STATS = True  # Enable per-step statistics: executor input token count + step time

# Add Memcompiler path
# sys.path adjusted for project root
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Memory configuration
UPDATE_MEMORY = True
CHROMA_HOST = "localhost"
CHROMA_PORT = 8074

# =============================================================================
# Import shared functions from original script
# =============================================================================

from openai import OpenAI, AzureOpenAI
from typing import Union

from run_manager_executor import (
    extract_task,
    parse_action,
    format_task_memory,
    parse_gpt_response,
    apply_memory_operation,
    qwen_choose_action,
    FileLock,
    # GPT_SYSTEM_PROMPT,
    # QWEN_SYSTEM_PROMPT,
    QwenLLMWrapper,
    GPTLLMWrapper,
    generate_runtime_key_steps,
)
from transformers import BitsAndBytesConfig

# Import Thor SoftCoT prompts for manager and executor (training-consistent)
# sys.path.insert(0, "/workspace/Embodied-MemoryAgent/examples/LatentMemory")
from run_manager_executor import GPT_SYSTEM_PROMPT as THOR_MANAGER_SYSTEM_PROMPT
from run_manager_executor import QWEN_SYSTEM_PROMPT as THOR_EXECUTOR_SYSTEM_PROMPT

# Import ScienceWorld prompts, action matching, task loading, and environment
from prompts.scienceworld_prompt import (
    GPT_SYSTEM_PROMPT_SCIENCEWORLD,
    QWEN_SYSTEM_PROMPT_SCIENCEWORLD,
    create_gpt_user_content_scienceworld,
    create_qwen_user_content_scienceworld,
    format_task_memory as format_task_memory_sciworld,
)
# from prompts.scienceworld_mas_prompt import scienceworld_few_shots
from run_manager_executor_scienceworld import (
    match_action_scienceworld,
    parse_action_scienceworld,
    load_scienceworld_tasks_from_jsonl,
)
from envs.scienceworld_env.scienceworld_env import ScienceWorldEnvWrapper

# =============================================================================
# GeminiLLMWrapper - Gemini via OpenAI-compatible API for Memcompiler
# =============================================================================
class GeminiLLMWrapper:
    """Wrapper to make Gemini API (via OpenAI-compatible endpoint) compatible with Memcompiler's LLMCallable interface."""

    def __init__(self, api_key: str, model: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.accumulated_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def reset_token_counter(self):
        self.accumulated_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_accumulated_tokens(self) -> Dict[str, int]:
        return self.accumulated_tokens.copy()

    def __call__(self, messages, temperature: float = 0.0, max_tokens: int = 4096, **kwargs) -> str:
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        for attempt in range(API_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=120,
                )
                print(f"[Gemini] response in Memcompiler: {response}")

                if response.usage:
                    self.accumulated_tokens["prompt_tokens"] += response.usage.prompt_tokens
                    self.accumulated_tokens["completion_tokens"] += response.usage.completion_tokens
                    self.accumulated_tokens["total_tokens"] += response.usage.total_tokens

                content = response.choices[0].message.content
                if content is None or content == "":
                    print(f"[Gemini] returned empty content, retrying (attempt {attempt + 1}/{API_MAX_RETRIES})...")
                    time.sleep(API_RETRY_DELAY)
                    continue
                return content
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    print(f"  [Gemini] rate limit (attempt {attempt + 1}/{API_MAX_RETRIES}), waiting {API_RETRY_DELAY}s...")
                    time.sleep(API_RETRY_DELAY)
                    continue
                print(f"[Gemini] API error in Memcompiler: {e}")
                return ""

        print(f"  [Gemini] API failed after {API_MAX_RETRIES} retries")
        return ""


# =============================================================================
# SafeMemcompiler - Independent implementation for eval_text_sample.py
# =============================================================================
def create_safe_memory_class(chroma_port=None):
    """Create SafeMemcompiler class with local CHROMA_HOST/CHROMA_PORT settings.
    If chroma_port is provided, use it instead of the global CHROMA_PORT.
    """
    _chroma_port = chroma_port if chroma_port is not None else CHROMA_PORT
    import pickle
    from core.memory.core_memory import Memcompiler
    from core.memory.core_memory.memcompiler import TaskLayer, InsightsManager
    from core.utils import write_json

    class SafeTaskLayer(TaskLayer):
        """TaskLayer with file locking for pickle operations and safe retrieval."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._lock_file = self._graph_save_path + ".lock"

        def _index_done(self) -> None:
            """Save graph with file locking (called within add_task_node which already holds lock)."""
            with open(self._graph_save_path, "wb") as f:
                pickle.dump(self.graph, f)

        def add_task_node(self, task_main: str) -> None:
            """Add a task node with proper load-modify-save under lock."""
            import networkx as nx

            with FileLock(self._lock_file):
                if os.path.exists(self._graph_save_path):
                    with open(self._graph_save_path, 'rb') as f:
                        self.graph = pickle.load(f)

                if task_main in self.graph:
                    return

                self.graph.add_node(task_main)

                results = self.task_storage.similarity_search_with_score(
                    query=task_main,
                    k=10
                )

                for doc, distance in results:
                    similarity = 1 - distance
                    if similarity < self.similarity_threshold:
                        continue

                    neighbor = doc.page_content

                    if neighbor not in self.graph:
                        self.graph.add_node(neighbor)

                    self.graph.add_edge(task_main, neighbor, weight=similarity)

                self._index_done()

        def retrieve_related_task(self, query_task: str, node_num: int, hop: int = 1) -> list:
            """Safe version that handles nodes not in graph (multi-process sync issues)."""
            import networkx as nx

            with FileLock(self._lock_file):
                if os.path.exists(self._graph_save_path):
                    with open(self._graph_save_path, 'rb') as f:
                        self.graph = pickle.load(f)

            tasks = self.task_storage.similarity_search_with_score(query=query_task, k=node_num)
            top_nodes = [doc[0].page_content for doc in tasks]

            related_nodes = set()
            for node in top_nodes:
                if node in self.graph:
                    related_nodes.add(node)
                    try:
                        neighbours = nx.single_source_shortest_path_length(self.graph, node, cutoff=hop).keys()
                        related_nodes.update(neighbours)
                    except Exception:
                        pass
            return list(related_nodes)

    class SafeInsightsManager(InsightsManager):
        """InsightsManager with file locking for JSON operations."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._lock_file = self.persist_file + ".lock"

        def _index_done(self):
            """Save insights with file locking."""
            with FileLock(self._lock_file):
                write_json(self.insights_memory, self.persist_file)

    class SafeMemcompiler(Memcompiler):
        """Memcompiler with file locking for concurrent multi-process access and ChromaDB HTTP client."""

        def __post_init__(self):
            import chromadb
            from langchain_chroma import Chroma

            persist_base_dir = self.global_config["working_dir"]
            os.makedirs(persist_base_dir, exist_ok=True)
            self.persist_dir = persist_base_dir

            # Use _chroma_port from closure (supports per-task independent ports)
            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=_chroma_port)
            self.main_memory = Chroma(
                client=chroma_client,
                collection_name="langchain",
                embedding_function=self.embedding_func,
            )

            self._hop = self.global_config.get('hop', 1)
            self._start_insights_threshold = self.global_config.get('start_insights_threshold', 5)
            self._rounds_per_insights = self.global_config.get('rounds_per_insights', 5)
            self._insights_point_num = self.global_config.get('insights_point_num', 5)

            self.task_layer = SafeTaskLayer(
                working_dir=self.persist_dir,
                namespace='task_layer',
                task_storage=self.main_memory
            )

            self.insights_layer = SafeInsightsManager(
                working_dir=self.persist_dir,
                namespace='insights',
                llm_model=self.llm_model,
                task_storage=self.main_memory,
                task_layer=self.task_layer
            )

            self.insights_cache = []
            self._insights_lock_file = os.path.join(self.persist_dir, "insights_update.lock")

            print(
                f"[SafeMemcompiler] Initialized with ChromaDB HTTP client at {CHROMA_HOST}:{_chroma_port} | "
                f"persist_dir={self.persist_dir} | collection=langchain"
            )

        def add_memory(self, mas_message) -> None:
            """Add memory with locked insights update."""
            from langchain.docstore.document import Document
            from core.memory.common import MASMessage

            mas_message = self._extract_mas_message(mas_message=mas_message)

            self.task_layer.add_task_node(mas_message.task_main)

            meta_data = MASMessage.to_dict(mas_message)
            memory_doc = Document(
                page_content=mas_message.task_main,
                metadata=meta_data
            )
            if mas_message.label == True or mas_message.label == False:
                self.main_memory.add_documents([memory_doc])
            else:
                raise ValueError('The mas_message must have label!')

            current_size = self.memory_size
            should_finetune = (current_size >= self._start_insights_threshold and
                              current_size % self._rounds_per_insights == 0)
            should_merge = (current_size % 20 == 0)

            if should_finetune or should_merge:
                try:
                    lock = FileLock(self._insights_lock_file)
                    lock.acquire()
                    try:
                        current_size = self.memory_size
                        if current_size >= self._start_insights_threshold and current_size % self._rounds_per_insights == 0:
                            print(f"[SafeMemcompiler] Finetuning insights at size {current_size}")
                            self.insights_layer.finetune_insights(self._insights_point_num)
                        if current_size % 20 == 0:
                            print(f"[SafeMemcompiler] Merging insights at size {current_size}")
                            self.insights_layer.merge_insights()
                    finally:
                        lock.release()
                except Exception as e:
                    print(f"[SafeMemcompiler] Insights update skipped or failed: {e}")

            self._index_done()

        def _index_done(self):
            """No-op for main Memcompiler - components handle their own persistence."""
            pass

    return SafeMemcompiler


# from llm_model import LatentPolicyWithProjection
class LatentPolicyWithProjection(nn.Module):
    """
    Latent Head with projection: maps from assistant hidden size to base hidden size.
    Outputs a Gaussian distribution and samples using reparameterization trick.
    """
    def __init__(self, input_size, output_size, intermediate_size=512, deterministic=False):
        super().__init__()
        self.deterministic = deterministic
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
        )
        self.mean = nn.Linear(intermediate_size, output_size)
        if not deterministic:
            self.log_std = nn.Linear(intermediate_size, output_size)

    def forward(self, x, temperature=1.0, return_distribution=False):
        """
        Args:
            x: hidden states from assistant model, shape (batch, seq_len, input_size)
            temperature: controls sampling variance
            return_distribution: if True, return (sampled, distribution) tuple
        Returns:
            if return_distribution=False: sampled hidden states, shape (batch, seq_len, output_size)
            if return_distribution=True: (sampled, distribution) tuple
        """
        x = self.fc(x)
        mean = self.mean(x)
        if self.deterministic:
            if return_distribution:
                return mean, None
            return mean
        log_std = self.log_std(x)
        std = log_std.exp() * temperature
        dist = torch.distributions.Normal(mean, std)
        sampled = dist.rsample()  # reparameterization trick for gradient flow
        if return_distribution:
            return sampled, dist
        return sampled

# =============================================================================
# Soft Token Helper Functions
# =============================================================================
def get_qwen_unk_token():
    """Get Qwen UNK token string (used as soft token placeholder)."""
    return "<" + "|" + "endoftext" + "|" + ">"


def extract_initial_observation(first_step_obs: str) -> str:
    """Extract initial observation from the first step's observation.
    
    The first step's observation format:
    "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room...\n\nYour task is to: ..."
    
    We extract only the middle part: "You are in the middle of a room..."
    (Aligned with data_loader.py's _extract_initial_observation)
    """
    if not first_step_obs:
        return ''
    
    # Split by "Your task is to:" to remove task description
    parts = first_step_obs.split('\n\nYour task is to:')
    content = parts[0] if parts else first_step_obs
    
    # Find "You are" to skip welcome message
    you_are_idx = content.find('You are')
    if you_are_idx != -1:
        return content[you_are_idx:].strip()
    
    return content.strip()


def build_soft_token_section(num_tokens: int) -> str:
    """Build the soft token section for executor prompt."""
    unk_token = get_qwen_unk_token()
    soft_thoughts = unk_token * num_tokens
    section = (
        "\n## Prompts from Assistant Model\n"
        "There are some prompts generated by a weaker assistant model. "
        "Some prompts maybe useful while others maybe unuseful for your reasoning. "
        "If the prompts are correct, you can use it as reference. "
        "If the prompts are not correct, you can ignore them and focus back to working on the task.\n"
        f"Here are prompts: {soft_thoughts}\n"
    )
    return section


def summarize_tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    tensor = tensor.detach().float()
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def make_latent_sample_seed(process_id: int, env_index: int, step_count: int) -> int:
    return LATENT_SAMPLE_BASE_SEED + process_id * 100000 + env_index * 1000 + step_count


def load_projection_layer(projection_path: str, input_dim: int, output_dim: int, device: str = "cuda"):
    """Load the trained sampling projection layer from .bin file."""
    projection = LatentPolicyWithProjection(
        input_size=input_dim,
        output_size=output_dim,
        intermediate_size=512,
        deterministic=False,
    ).to(device=device, dtype=torch.bfloat16)
    state_dict = torch.load(projection_path, map_location=device)
    expected_keys = {"fc.0.weight", "fc.2.weight", "mean.weight", "log_std.weight"}
    missing_expected = [key for key in expected_keys if key not in state_dict]
    if missing_expected:
        raise ValueError(
            f"Projection checkpoint does not look like LatentPolicyWithProjection. Missing keys: {missing_expected}"
        )
    projection.load_state_dict(state_dict)
    projection.eval()
    return projection


def _convert_peft_state_dict(raw_state_dict, lora_alpha=32, lora_r=16):
    """
    Convert a PeftModel (LoRA) state_dict to a plain model state_dict by:
    1. Stripping the 'base_model.model.' prefix added by PeftModel.
    2. Renaming '*.base_layer.weight/bias' back to '*.weight/bias'.
    3. Merging LoRA weights: merged = base + lora_B @ lora_A * (alpha / r).

    Returns:
        merged_state_dict: dict compatible with a plain AutoModelForCausalLM.
        is_peft: bool, whether PeftModel keys were detected.
    """
    # Detect if this is a PeftModel state_dict
    has_peft_prefix = any(k.startswith("base_model.model.") for k in raw_state_dict)
    has_lora_keys = any("lora_A" in k for k in raw_state_dict)

    if not has_peft_prefix and not has_lora_keys:
        # Already a plain state_dict, return as-is
        return raw_state_dict, False

    scaling = lora_alpha / lora_r
    print(f"  Detected PeftModel (LoRA) checkpoint, merging with scaling={scaling}")

    # Step 1: Strip 'base_model.model.' prefix and categorise keys
    base_weights = {}   # module_path -> {weight: tensor, bias: tensor}
    lora_a = {}         # module_path -> tensor
    lora_b = {}         # module_path -> tensor
    plain_params = {}   # key -> tensor  (non-LoRA params)

    peft_prefix = "base_model.model."
    for key, value in raw_state_dict.items():
        # Strip PeftModel prefix
        if key.startswith(peft_prefix):
            stripped = key[len(peft_prefix):]
        else:
            stripped = key

        if ".base_layer." in stripped:
            # e.g. model.layers.0.self_attn.q_proj.base_layer.weight
            #   -> module_path = model.layers.0.self_attn.q_proj
            #   -> param_name = weight
            parts = stripped.split(".base_layer.")
            module_path = parts[0]
            param_name = parts[1]  # 'weight' or 'bias'
            if module_path not in base_weights:
                base_weights[module_path] = {}
            base_weights[module_path][param_name] = value
        elif ".lora_A." in stripped:
            # e.g. model.layers.0.self_attn.q_proj.lora_A.default.weight
            module_path = stripped.split(".lora_A.")[0]
            lora_a[module_path] = value
        elif ".lora_B." in stripped:
            module_path = stripped.split(".lora_B.")[0]
            lora_b[module_path] = value
        else:
            plain_params[stripped] = value

    # Step 2: Merge LoRA into base weights
    merged_state_dict = dict(plain_params)
    merged_count = 0

    for module_path, params in base_weights.items():
        for param_name, base_tensor in params.items():
            full_key = f"{module_path}.{param_name}"
            if param_name == "weight" and module_path in lora_a and module_path in lora_b:
                # merged_weight = base_weight + lora_B @ lora_A * scaling
                delta = lora_b[module_path] @ lora_a[module_path] * scaling
                merged_state_dict[full_key] = base_tensor + delta.to(base_tensor.dtype)
                merged_count += 1
            else:
                merged_state_dict[full_key] = base_tensor

    print(f"  Merged LoRA into {merged_count} weight matrices")
    return merged_state_dict, True


def load_from_checkpoint(checkpoint_path: str, manager_model, input_dim: int, output_dim: int, device: str = "cuda"):
    """
    Load assistant model and projection from a training checkpoint directory.
    
    The checkpoint contains pytorch_model.bin with keys:
    - assistant_model.* : assistant model weights (possibly PeftModel/LoRA wrapped)
    - projection.* : projection layer weights
    
    Returns:
        projection: loaded LatentPolicyWithProjection
    """
    checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    
    print(f"Loading from checkpoint: {checkpoint_file}")
    full_state_dict = torch.load(checkpoint_file, map_location=device, weights_only=False)
    
    # Extract assistant_model weights (remove 'assistant_model.' prefix)
    assistant_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith("assistant_model."):
            new_key = key[len("assistant_model."):]
            assistant_state_dict[new_key] = value
    
    print(f"  Found {len(assistant_state_dict)} assistant_model keys")
    
    # Handle PeftModel (LoRA) checkpoint: remap keys and merge LoRA weights
    assistant_state_dict, is_peft = _convert_peft_state_dict(assistant_state_dict)
    if is_peft:
        print(f"  After LoRA merge: {len(assistant_state_dict)} keys")
    
    # Load assistant model weights
    load_result = manager_model.load_state_dict(assistant_state_dict, strict=False)
    if load_result.missing_keys:
        print(f"  Note: {len(load_result.missing_keys)} keys not in checkpoint (visual encoder kept at pretrained)")
    if load_result.unexpected_keys:
        print(f"  Warning: {len(load_result.unexpected_keys)} unexpected keys in checkpoint")
    
    # Extract projection weights (remove 'projection.' prefix)
    projection_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith("projection."):
            new_key = key[len("projection."):]
            projection_state_dict[new_key] = value
    
    print(f"  Found {len(projection_state_dict)} projection keys")
    
    # Create and load projection layer
    projection = LatentPolicyWithProjection(
        input_size=input_dim,
        output_size=output_dim,
        intermediate_size=512,
        deterministic=False,
    ).to(device=device, dtype=torch.bfloat16)
    
    projection.load_state_dict(projection_state_dict)
    projection.eval()
    
    print(f"  Successfully loaded assistant_model and projection from checkpoint")
    return projection


@torch.no_grad()
def call_assistant_with_soft_tokens(
    manager_model,
    manager_processor,
    manager_tokenizer,
    projection,
    working_memory: str,
    task_text: str,
    history: List[Dict[str, Any]],
    current_obs: str,
    task_memory: Dict[str, Any],
    num_thought_tokens: int,
    latent_sample_seed: Optional[int] = None,
    initial_observation: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], str, Dict[str, Any]]:
    """
    Two-pass manager inference with KV Cache reuse optimization.

    Pass-1 and Pass-2 share a long common token prefix (system prompt +
    manager_text_content). We forward this prefix ONCE, then:
      Pass-1: generate() with KV cache + short suffix → manager text response
      Pass-2: forward()  with KV cache + short suffix → extract UNK hidden states

    This avoids redundant computation of the shared prefix (~30-50% speedup).

    Returns:
        - sampled_latent_embeddings: (1, num_thought_tokens, executor_hidden_size) or None if failed
        - text_response: manager raw generated response
        - input_fields: dict with input info for logging
    """
    # Step 1: Get manager_text_content using aligned function
    content, manager_input_fields = create_manager_qwen_user_content_text(
        working_memory=working_memory,
        task_text=task_text,
        history=history,
        current_obs=current_obs,
        admissible_commands=None,
        task_memory=task_memory,
        initial_observation=initial_observation,
    )
    
    # Extract text from content
    if isinstance(content, list) and len(content) > 0:
        first = content[0]
        if isinstance(first, dict) and first.get("type") == "text":
            manager_text_content = first.get("text", "")
        else:
            manager_text_content = str(first)
    else:
        manager_text_content = ""
    
    # =========================================================================
    # KV Cache Reuse Optimization (v2)
    # =========================================================================
    # Strategy: Run Pass-1 generate() normally (full prompt). Then extract
    # the KV cache from generate's output, crop it to the shared prefix
    # between Pass-1 and Pass-2, and reuse it for Pass-2's forward call.
    # This saves recomputing the shared prefix (~90%+ of tokens) in Pass-2.
    # =========================================================================
    tokenizer = manager_processor.tokenizer if manager_processor is not None else manager_tokenizer
    unk_token = get_qwen_unk_token()
    assistant_token_part = unk_token * num_thought_tokens
    soft_token_prefix = f'Here are {num_thought_tokens} tokens to help the language model solve this task: '
    soft_token_prompt = (
        f'You are required to generate {num_thought_tokens} tokens to help the Low-Level Executor '
        f'to complete a household robot task efficiently. '
        f'Here are the requirements of your generated tokens:\n'
        f'- The tokens should include some useful information for the task, '
        f'for example, the key objects, locations, and actions needed.\n'
        f'- Generate the tokens starts from the most important or the highest related tokens.\n'
        f'- **Informative tokens are required**: (1) Do not need to generate a sentence or paragraph, '
        f'(2) Do not need to generate the uninformative tokens such as serial number.\n'
        f'- The tokens should be useful for the Low-Level Executor to decide the next action.\n'
        f'- The Executor is good enough to understand the task, so what you need to do is '
        f'generate some informative key tokens that can represent the condensed essence of TASK MEMORY.\n'
        f'...'
    )

    # -- Pass-1: Normal full generate (unchanged from original) --
    pass1_messages = [
        {'role': 'system', 'content': THOR_MANAGER_SYSTEM_PROMPT},
        {'role': 'user', 'content': manager_text_content},
    ]
    if manager_processor is not None:
        pass1_prompt = manager_processor.apply_chat_template(pass1_messages, tokenize=False, add_generation_prompt=True)
        pass1_inputs = manager_processor(text=[pass1_prompt], padding=True, return_tensors="pt").to(manager_model.device)
    else:
        pass1_prompt = manager_tokenizer.apply_chat_template(pass1_messages, tokenize=False, add_generation_prompt=True)
        pass1_inputs = manager_tokenizer([pass1_prompt], padding=True, return_tensors="pt").to(manager_model.device)
    pass1_ids = pass1_inputs.input_ids  # (1, L1)
    prefix_len = pass1_ids.size(1)
    manager_pass1_input_token_count = int(prefix_len)

    _t_pass1_start = time.time()
    generated_ids = manager_model.generate(
        **pass1_inputs,
        max_new_tokens=512,
        do_sample=False,
        return_dict_in_generate=True,
        use_cache=True,
    )
    torch.cuda.synchronize()
    _t_pass1_end = time.time()
    _manager_generate_sec = round(_t_pass1_end - _t_pass1_start, 4)
    if generated_ids is not None:
        print(f"[DEBUG] generated_ids shape: {generated_ids.sequences.shape}")

    full_generated_ids = generated_ids.sequences
    generated_text_ids = full_generated_ids[0, prefix_len:]
    manager_output_token_count = int(generated_text_ids.size(0))
    print(f"[DEBUG] generated_text_ids shape: {generated_text_ids.shape}")

    if manager_processor is not None:
        text_response_raw = manager_processor.batch_decode([generated_text_ids], skip_special_tokens=False)[0]
        text_response = manager_processor.batch_decode([generated_text_ids], skip_special_tokens=True)[0].strip()
    else:
        text_response_raw = manager_tokenizer.batch_decode([generated_text_ids], skip_special_tokens=False)[0]
        text_response = manager_tokenizer.batch_decode([generated_text_ids], skip_special_tokens=True)[0].strip()
    print(f"[DEBUG] text_response_raw: {repr(text_response_raw)}")

    # Fallback when Pass-1 generation is empty/unusable.
    plain_parse_mode = "exact"
    if not text_response:
        plain_parse_mode = "fallback"
        text_response = "Here is the strategic response: <response_type>NOACTION</response_type>"

    # In no-soft-token ablation, skip Pass-2 projection path entirely.
    if projection is None:
        manager_input_fields["plain_parse_mode"] = plain_parse_mode
        manager_input_fields["projection_type"] = "disabled"
        manager_input_fields["latent_sample_seed"] = latent_sample_seed
        manager_input_fields["found_unk_positions"] = 0
        manager_input_fields["selected_unk_positions"] = []
        manager_input_fields["padding_applied"] = False
        manager_input_fields["final_soft_tokens"] = 0
        manager_input_fields["manager_pass1_input_token_count"] = manager_pass1_input_token_count
        manager_input_fields["manager_pass2_input_token_count"] = 0
        manager_input_fields["manager_output_token_count"] = manager_output_token_count
        manager_input_fields["manager_generate_sec"] = _manager_generate_sec
        manager_input_fields["manager_soft_forward_sec"] = 0.0
        return None, text_response, manager_input_fields

    # -- Pass-2: Extract hidden states, reusing Pass-1 KV cache for the shared prefix --
    user_content_for_hs = (
        f'{manager_text_content}\n\n'
        f'## Manager Decision\n{text_response}\n\n'
        f'## Soft Token Generation Task\n{soft_token_prompt}'
    )
    assistant_content_for_hs = f'{soft_token_prefix}{assistant_token_part}'
    pass2_messages = [
        {'role': 'system', 'content': THOR_MANAGER_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_content_for_hs},
        {'role': 'assistant', 'content': assistant_content_for_hs},
    ]
    if manager_processor is not None:
        pass2_prompt = manager_processor.apply_chat_template(pass2_messages, tokenize=False, add_generation_prompt=False)
        pass2_inputs = manager_processor(text=[pass2_prompt], padding=True, return_tensors="pt").to(manager_model.device)
    else:
        pass2_prompt = manager_tokenizer.apply_chat_template(pass2_messages, tokenize=False, add_generation_prompt=False)
        pass2_inputs = manager_tokenizer([pass2_prompt], padding=True, return_tensors="pt").to(manager_model.device)
    pass2_ids = pass2_inputs.input_ids  # (1, L2)
    manager_pass2_input_token_count = int(pass2_ids.size(1))

    # Find longest common token prefix between Pass-1 and Pass-2
    min_len = min(pass1_ids.size(1), pass2_ids.size(1))
    match_mask = (pass1_ids[0, :min_len] == pass2_ids[0, :min_len])
    if match_mask.all():
        shared_prefix_len = min_len
    else:
        shared_prefix_len = int(match_mask.long().argmin().item())
    shared_prefix_len = max(shared_prefix_len, 1)
    print(f"[KV-OPT] Pass-1 tokens: {pass1_ids.size(1)}, Pass-2 tokens: {pass2_ids.size(1)}, shared prefix: {shared_prefix_len}")

    # Try to extract + crop KV cache from Pass-1 generate output
    shared_kv_cache = None
    pass1_kv = getattr(generated_ids, 'past_key_values', None)
    if pass1_kv is not None:
        try:
            if isinstance(pass1_kv, tuple):
                # Legacy tuple format: each layer is (key, value) with shape (B, H, S, D)
                shared_kv_cache = tuple(
                    tuple(t[:, :, :shared_prefix_len, :] for t in layer)
                    for layer in pass1_kv
                )
            elif hasattr(pass1_kv, 'crop'):
                # Modern Cache API (transformers >= 4.45): use built-in crop()
                # crop() modifies in-place; we no longer need the full generate cache
                pass1_kv.crop(shared_prefix_len)
                shared_kv_cache = pass1_kv
            elif hasattr(pass1_kv, 'layers'):
                # DynamicCache with layers API but no crop() (unlikely, safety fallback)
                from transformers.cache_utils import DynamicCache
                cropped_data = []
                for layer_idx in range(len(pass1_kv)):
                    keys, values = pass1_kv[layer_idx]
                    cropped_data.append((
                        keys[:, :, :shared_prefix_len, :],
                        values[:, :, :shared_prefix_len, :],
                    ))
                shared_kv_cache = DynamicCache(ddp_cache_data=cropped_data)
            else:
                print(f"[KV-OPT] Unknown cache type: {type(pass1_kv)}, cannot crop")

            if shared_kv_cache is not None:
                cache_seq_len = shared_kv_cache.get_seq_length() if hasattr(shared_kv_cache, 'get_seq_length') else '?'
                print(f"[KV-OPT] Cropped KV cache from generate output: seq_len={cache_seq_len}, saved {shared_prefix_len} token recomputation for Pass-2")
        except Exception as e:
            print(f"[KV-OPT] 👉👉👉Failed to crop KV cache: {e}, falling back to recompute")
            import traceback; traceback.print_exc()
            shared_kv_cache = None

    # Fallback: recompute shared prefix KV cache from scratch (no savings)
    if shared_kv_cache is None:
        print(f"[KV-OPT] WARNING: Recomputing shared prefix KV cache from scratch ({shared_prefix_len} tokens) —👉👉👉 no speedup!")
        shared_prefix_ids = pass2_ids[:, :shared_prefix_len]
        shared_prefix_attn = torch.ones_like(shared_prefix_ids)
        prefix_outputs = manager_model(
            input_ids=shared_prefix_ids,
            attention_mask=shared_prefix_attn,
            use_cache=True,
            return_dict=True,
        )
        shared_kv_cache = prefix_outputs.past_key_values

    # Forward only the suffix tokens with the shared KV cache
    _t_pass2_start = time.time()
    pass2_suffix_ids = pass2_ids[:, shared_prefix_len:]  # (1, L2 - K)
    pass2_full_attn = torch.ones(1, pass2_ids.size(1), device=pass2_ids.device, dtype=pass2_ids.dtype)

    forward_outputs = manager_model(
        input_ids=pass2_suffix_ids,
        attention_mask=pass2_full_attn,
        past_key_values=shared_kv_cache,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    # Hidden states from suffix forward; positions are relative to suffix (0-indexed)
    suffix_hidden_states = forward_outputs.hidden_states[-1]  # (1, L2 - K, hidden_size)

    # Find UNK positions in the FULL Pass-2 sequence, remap to suffix-relative positions
    unk_token_id = tokenizer.convert_tokens_to_ids(unk_token)
    if unk_token_id is None or unk_token_id < 0:
        raise ValueError(f"Failed to resolve UNK token id for token: {unk_token}")

    unk_positions_full = (pass2_ids[0] == unk_token_id).nonzero(as_tuple=True)[0]
    if len(unk_positions_full) < num_thought_tokens:
        print(
            f"[WARNING] Found only {len(unk_positions_full)} UNK positions in Pass-2 prompt, "
            f"expected {num_thought_tokens}; will pad with EOT embeddings."
        )
    selected_positions_full = unk_positions_full[-num_thought_tokens:] if len(unk_positions_full) >= num_thought_tokens else unk_positions_full
    # Remap to suffix-relative positions
    selected_positions_suffix = selected_positions_full - shared_prefix_len
    print(f"[KV-OPT] Pass-2 suffix tokens: {pass2_suffix_ids.size(1)}, UNK positions (suffix-relative): {selected_positions_suffix.tolist()}")

    # Extract hidden states using suffix-relative positions
    full_hidden_states = suffix_hidden_states  # (1, suffix_len, hidden_size)

    if len(selected_positions_suffix) > 0:
        soft_hidden = full_hidden_states[0, selected_positions_suffix, :]
    else:
        soft_hidden = full_hidden_states.new_zeros((0, full_hidden_states.size(-1)))

    eot_token_id = tokenizer.convert_tokens_to_ids(unk_token)
    if eot_token_id is None or eot_token_id < 0:
        eot_token_id = tokenizer.eos_token_id
    if eot_token_id is None:
        raise ValueError("Failed to resolve <|endoftext|>/eos token id for padding fallback")

    padding_applied = soft_hidden.size(0) < num_thought_tokens
    if padding_applied:
        pad_count = num_thought_tokens - soft_hidden.size(0)
        eot_input = torch.tensor([eot_token_id], device=full_hidden_states.device, dtype=torch.long)
        eot_embed = manager_model.get_input_embeddings()(eot_input).to(dtype=full_hidden_states.dtype)
        eot_embed = eot_embed.repeat(pad_count, 1)
        soft_hidden = torch.cat([soft_hidden, eot_embed], dim=0)
    else:
        soft_hidden = soft_hidden[:num_thought_tokens, :]

    soft_hidden_batch = soft_hidden.unsqueeze(0).to(device=next(projection.parameters()).device, dtype=torch.bfloat16)

    if latent_sample_seed is not None:
        torch.manual_seed(latent_sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(latent_sample_seed)

    sampled_latent, distribution = projection(soft_hidden_batch, return_distribution=True)
    torch.cuda.synchronize()
    _t_pass2_end = time.time()
    _manager_soft_forward_sec = round(_t_pass2_end - _t_pass2_start, 4)
    distribution_mean = distribution.mean
    distribution_std = distribution.stddev

    print(
        f"[DEBUG] plain_parse_mode={plain_parse_mode}, "
        f"latent_sample_seed={latent_sample_seed}, "
        f"found_unk_positions={len(unk_positions_full)}, "
        f"selected_positions_full={selected_positions_full.tolist()}, "
        f"padding_applied={padding_applied}, "
        f"final_soft_tokens={sampled_latent.size(1)}"
    )
    print(f"[DEBUG] assistant soft_hidden stats: {summarize_tensor_stats(soft_hidden_batch)}")
    print(f"[DEBUG] projection mean stats: {summarize_tensor_stats(distribution_mean)}")
    print(f"[DEBUG] projection std stats: {summarize_tensor_stats(distribution_std)}")
    print(f"[DEBUG] sampled latent stats: {summarize_tensor_stats(sampled_latent)}")

    manager_input_fields["plain_parse_mode"] = plain_parse_mode
    manager_input_fields["projection_type"] = "sample"
    manager_input_fields["latent_sample_seed"] = latent_sample_seed
    manager_input_fields["kv_shared_prefix_len"] = shared_prefix_len
    manager_input_fields["found_unk_positions"] = int(len(unk_positions_full))
    manager_input_fields["selected_unk_positions"] = [int(pos) for pos in selected_positions_full.tolist()]
    manager_input_fields["padding_applied"] = bool(padding_applied)
    manager_input_fields["soft_hidden_shape"] = list(soft_hidden_batch.shape)
    manager_input_fields["distribution_mean_shape"] = list(distribution_mean.shape)
    manager_input_fields["distribution_std_shape"] = list(distribution_std.shape)
    manager_input_fields["sampled_latent_shape"] = list(sampled_latent.shape)
    manager_input_fields["soft_hidden_stats"] = summarize_tensor_stats(soft_hidden_batch)
    manager_input_fields["distribution_mean_stats"] = summarize_tensor_stats(distribution_mean)
    manager_input_fields["distribution_std_stats"] = summarize_tensor_stats(distribution_std)
    manager_input_fields["sampled_latent_stats"] = summarize_tensor_stats(sampled_latent)
    manager_input_fields["final_soft_tokens"] = int(sampled_latent.size(1))
    manager_input_fields["manager_pass1_input_token_count"] = manager_pass1_input_token_count
    manager_input_fields["manager_pass2_input_token_count"] = manager_pass2_input_token_count
    manager_input_fields["manager_output_token_count"] = manager_output_token_count
    manager_input_fields["manager_generate_sec"] = _manager_generate_sec
    manager_input_fields["manager_soft_forward_sec"] = _manager_soft_forward_sec

    return sampled_latent, text_response, manager_input_fields


# =============================================================================
# ScienceWorld Manager: Two-pass soft token inference (aligned with training)
# =============================================================================
@torch.no_grad()
def call_assistant_with_soft_tokens_sciworld(
    manager_model,
    manager_processor,
    manager_tokenizer,
    projection,
    working_memory: str,
    task_text: str,
    history: List[Dict[str, Any]],
    current_obs: str,
    current_score: int,
    task_memory: Dict[str, Any],
    num_thought_tokens: int,
    latent_sample_seed: Optional[int] = None,
    initial_observation: Optional[str] = None,
    few_shots: Optional[List[str]] = None,
) -> Tuple[Optional[torch.Tensor], str, Dict[str, Any]]:
    """
    ScienceWorld version of two-pass manager inference with KV Cache reuse.
    Mirrors call_assistant_with_soft_tokens but uses ScienceWorld prompts.
    """
    # Step 1: Build manager content using scienceworld prompt builder
    content, manager_input_fields = create_gpt_user_content_scienceworld(
        working_memory=working_memory,
        task_text=task_text,
        history=history,
        current_obs=current_obs,
        current_score=current_score,
        task_memory=task_memory,
        initial_observation=initial_observation,
        few_shots=few_shots,
    )

    # Extract text from content
    if isinstance(content, list) and len(content) > 0:
        first = content[0]
        if isinstance(first, dict) and first.get("type") == "text":
            manager_text_content = first.get("text", "")
        else:
            manager_text_content = str(first)
    else:
        manager_text_content = ""

    # =========================================================================
    # KV Cache Reuse Optimization (same mechanism as ALFWorld version)
    # =========================================================================
    tokenizer = manager_processor.tokenizer if manager_processor is not None else manager_tokenizer
    unk_token = get_qwen_unk_token()
    assistant_token_part = unk_token * num_thought_tokens
    soft_token_prefix = f'Here are {num_thought_tokens} tokens to help the language model solve this task: '
    soft_token_prompt = (
            f'Generate {num_thought_tokens} compressed tokens that represent the condensed essence of TASK MEMORY '
            f'for the Low-Level Executor.\n'
            f'Requirements:\n'
            f'- Each token must be maximally information-dense: encode important infomation of the content of TASK MEMORY section above.\n'
            f'- Prioritize by importance: the most critical information first.\n'
            f'- These tokens are NOT natural language — they are continuous signals that the Executor '
            f'will directly consume to decide its next action.\n'
            f'- Avoid redundancy: do not repeat what the Executor already knows from the task description. do not repeat the manager decision already given.'
    )

    # -- Pass-1: Normal full generate --
    pass1_messages = [
        {'role': 'system', 'content': GPT_SYSTEM_PROMPT_SCIENCEWORLD},
        {'role': 'user', 'content': manager_text_content},
    ]
    if manager_processor is not None:
        pass1_prompt = manager_processor.apply_chat_template(pass1_messages, tokenize=False, add_generation_prompt=True)
        pass1_inputs = manager_processor(text=[pass1_prompt], padding=True, return_tensors="pt").to(manager_model.device)
    else:
        pass1_prompt = manager_tokenizer.apply_chat_template(pass1_messages, tokenize=False, add_generation_prompt=True)
        pass1_inputs = manager_tokenizer([pass1_prompt], padding=True, return_tensors="pt").to(manager_model.device)
    pass1_ids = pass1_inputs.input_ids
    prefix_len = pass1_ids.size(1)
    manager_pass1_input_token_count = int(prefix_len)

    _t_pass1_start = time.time()
    generated_ids = manager_model.generate(
        **pass1_inputs,
        max_new_tokens=512,
        do_sample=False,
        return_dict_in_generate=True,
        use_cache=True,
    )
    torch.cuda.synchronize()
    _t_pass1_end = time.time()
    _manager_generate_sec = round(_t_pass1_end - _t_pass1_start, 4)

    full_generated_ids = generated_ids.sequences
    generated_text_ids = full_generated_ids[0, prefix_len:]
    manager_output_token_count = int(generated_text_ids.size(0))

    if manager_processor is not None:
        text_response_raw = manager_processor.batch_decode([generated_text_ids], skip_special_tokens=False)[0]
        text_response = manager_processor.batch_decode([generated_text_ids], skip_special_tokens=True)[0].strip()
    else:
        text_response_raw = manager_tokenizer.batch_decode([generated_text_ids], skip_special_tokens=False)[0]
        text_response = manager_tokenizer.batch_decode([generated_text_ids], skip_special_tokens=True)[0].strip()

    plain_parse_mode = "exact"
    if not text_response:
        plain_parse_mode = "fallback"
        text_response = "Here is the strategic response: <response_type>NOACTION</response_type>"

    # In no-soft-token ablation, skip Pass-2 projection path entirely.
    if projection is None:
        manager_input_fields["plain_parse_mode"] = plain_parse_mode
        manager_input_fields["projection_type"] = "disabled"
        manager_input_fields["latent_sample_seed"] = latent_sample_seed
        manager_input_fields["found_unk_positions"] = 0
        manager_input_fields["selected_unk_positions"] = []
        manager_input_fields["padding_applied"] = False
        manager_input_fields["final_soft_tokens"] = 0
        manager_input_fields["manager_pass1_input_token_count"] = manager_pass1_input_token_count
        manager_input_fields["manager_pass2_input_token_count"] = 0
        manager_input_fields["manager_output_token_count"] = manager_output_token_count
        manager_input_fields["manager_generate_sec"] = _manager_generate_sec
        manager_input_fields["manager_soft_forward_sec"] = 0.0
        return None, text_response, manager_input_fields

    # -- Pass-2: Extract hidden states, reusing Pass-1 KV cache --
    user_content_for_hs = (
        f'{manager_text_content}\n\n'
        f'## Manager Decision\n{text_response}\n\n'
        f'## Soft Token Generation Task\n{soft_token_prompt}'
    )
    assistant_content_for_hs = f'{soft_token_prefix}{assistant_token_part}'
    pass2_messages = [
        {'role': 'system', 'content': GPT_SYSTEM_PROMPT_SCIENCEWORLD},
        {'role': 'user', 'content': user_content_for_hs},
        {'role': 'assistant', 'content': assistant_content_for_hs},
    ]
    if manager_processor is not None:
        pass2_prompt = manager_processor.apply_chat_template(pass2_messages, tokenize=False, add_generation_prompt=False)
        pass2_inputs = manager_processor(text=[pass2_prompt], padding=True, return_tensors="pt").to(manager_model.device)
    else:
        pass2_prompt = manager_tokenizer.apply_chat_template(pass2_messages, tokenize=False, add_generation_prompt=False)
        pass2_inputs = manager_tokenizer([pass2_prompt], padding=True, return_tensors="pt").to(manager_model.device)
    pass2_ids = pass2_inputs.input_ids
    manager_pass2_input_token_count = int(pass2_ids.size(1))

    # Find longest common token prefix between Pass-1 and Pass-2
    min_len = min(pass1_ids.size(1), pass2_ids.size(1))
    match_mask = (pass1_ids[0, :min_len] == pass2_ids[0, :min_len])
    if match_mask.all():
        shared_prefix_len = min_len
    else:
        shared_prefix_len = int(match_mask.long().argmin().item())
    shared_prefix_len = max(shared_prefix_len, 1)

    # Try to extract + crop KV cache from Pass-1 generate output
    shared_kv_cache = None
    pass1_kv = getattr(generated_ids, 'past_key_values', None)
    if pass1_kv is not None:
        try:
            if isinstance(pass1_kv, tuple):
                shared_kv_cache = tuple(
                    tuple(t[:, :, :shared_prefix_len, :] for t in layer)
                    for layer in pass1_kv
                )
            elif hasattr(pass1_kv, 'crop'):
                pass1_kv.crop(shared_prefix_len)
                shared_kv_cache = pass1_kv
            elif hasattr(pass1_kv, 'layers'):
                from transformers.cache_utils import DynamicCache
                cropped_data = []
                for layer_idx in range(len(pass1_kv)):
                    keys, values = pass1_kv[layer_idx]
                    cropped_data.append((
                        keys[:, :, :shared_prefix_len, :],
                        values[:, :, :shared_prefix_len, :],
                    ))
                shared_kv_cache = DynamicCache(ddp_cache_data=cropped_data)
        except Exception as e:
            print(f"[KV-OPT-SW] Failed to crop KV cache: {e}")
            shared_kv_cache = None

    if shared_kv_cache is None:
        shared_prefix_ids = pass2_ids[:, :shared_prefix_len]
        shared_prefix_attn = torch.ones_like(shared_prefix_ids)
        prefix_outputs = manager_model(
            input_ids=shared_prefix_ids,
            attention_mask=shared_prefix_attn,
            use_cache=True,
            return_dict=True,
        )
        shared_kv_cache = prefix_outputs.past_key_values

    # Forward only the suffix tokens with the shared KV cache
    _t_pass2_start = time.time()
    pass2_suffix_ids = pass2_ids[:, shared_prefix_len:]
    pass2_full_attn = torch.ones(1, pass2_ids.size(1), device=pass2_ids.device, dtype=pass2_ids.dtype)

    forward_outputs = manager_model(
        input_ids=pass2_suffix_ids,
        attention_mask=pass2_full_attn,
        past_key_values=shared_kv_cache,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    suffix_hidden_states = forward_outputs.hidden_states[-1]

    # Find UNK positions and extract hidden states
    unk_token_id = tokenizer.convert_tokens_to_ids(unk_token)
    unk_positions_full = (pass2_ids[0] == unk_token_id).nonzero(as_tuple=True)[0]
    selected_positions_full = unk_positions_full[-num_thought_tokens:] if len(unk_positions_full) >= num_thought_tokens else unk_positions_full
    selected_positions_suffix = selected_positions_full - shared_prefix_len

    full_hidden_states = suffix_hidden_states
    if len(selected_positions_suffix) > 0:
        soft_hidden = full_hidden_states[0, selected_positions_suffix, :]
    else:
        soft_hidden = full_hidden_states.new_zeros((0, full_hidden_states.size(-1)))

    eot_token_id = tokenizer.convert_tokens_to_ids(unk_token)
    if eot_token_id is None or eot_token_id < 0:
        eot_token_id = tokenizer.eos_token_id

    padding_applied = soft_hidden.size(0) < num_thought_tokens
    if padding_applied:
        pad_count = num_thought_tokens - soft_hidden.size(0)
        eot_input = torch.tensor([eot_token_id], device=full_hidden_states.device, dtype=torch.long)
        eot_embed = manager_model.get_input_embeddings()(eot_input).to(dtype=full_hidden_states.dtype)
        eot_embed = eot_embed.repeat(pad_count, 1)
        soft_hidden = torch.cat([soft_hidden, eot_embed], dim=0)
    else:
        soft_hidden = soft_hidden[:num_thought_tokens, :]

    soft_hidden_batch = soft_hidden.unsqueeze(0).to(device=next(projection.parameters()).device, dtype=torch.bfloat16)

    if latent_sample_seed is not None:
        torch.manual_seed(latent_sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(latent_sample_seed)

    sampled_latent, distribution = projection(soft_hidden_batch, return_distribution=True)
    torch.cuda.synchronize()
    _t_pass2_end = time.time()
    _manager_soft_forward_sec = round(_t_pass2_end - _t_pass2_start, 4)

    manager_input_fields["plain_parse_mode"] = plain_parse_mode
    manager_input_fields["projection_type"] = "sample"
    manager_input_fields["latent_sample_seed"] = latent_sample_seed
    manager_input_fields["kv_shared_prefix_len"] = shared_prefix_len
    manager_input_fields["found_unk_positions"] = int(len(unk_positions_full))
    manager_input_fields["selected_unk_positions"] = [int(pos) for pos in selected_positions_full.tolist()]
    manager_input_fields["padding_applied"] = bool(padding_applied)
    manager_input_fields["soft_hidden_shape"] = list(soft_hidden_batch.shape)
    manager_input_fields["sampled_latent_shape"] = list(sampled_latent.shape)
    manager_input_fields["final_soft_tokens"] = int(sampled_latent.size(1))
    manager_input_fields["manager_pass1_input_token_count"] = manager_pass1_input_token_count
    manager_input_fields["manager_pass2_input_token_count"] = manager_pass2_input_token_count
    manager_input_fields["manager_output_token_count"] = manager_output_token_count
    manager_input_fields["manager_generate_sec"] = _manager_generate_sec
    manager_input_fields["manager_soft_forward_sec"] = _manager_soft_forward_sec

    return sampled_latent, text_response, manager_input_fields


@torch.no_grad()
def executor_generate_with_soft_tokens(
    executor_model,
    executor_tokenizer,
    sampled_latent_embeddings: Optional[torch.Tensor],
    working_memory: str,
    task_text: str,
    admissible: List[str],
    history: List[Dict[str, Any]],
    current_obs: str,
    gpt_guidance: str,
    num_thought_tokens: int,
    use_soft_tokens: bool = True,
    max_history_turns: int = 12,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Generate executor response with soft token embeddings injected.
    
    1. Build executor prompt with soft token placeholders
    2. Get input embeddings
    3. Replace placeholder embeddings with projected soft token embeddings
    4. Generate using inputs_embeds
    """
    # Build executor prompt (similar to qwen_choose_action but with soft token section)
    prompt_parts = []
    
    prompt_parts.append("## Working Memory")
    if working_memory and working_memory.strip():
        prompt_parts.append(working_memory)
    else:
        prompt_parts.append("(Empty for the time being)")
    
    prompt_parts.append("\n## ATTENTION: Extremely useful guidance for your next action")
    if gpt_guidance:
        prompt_parts.append(gpt_guidance)
    else:
        prompt_parts.append("(Empty for the time being)")
    
    prompt_parts.append("\n## Task Description")
    prompt_parts.append(f"Your task is: {task_text}")
    
    prompt_parts.append("\n## Executed Action History")
    if history:
        recent_history = history[-max_history_turns:]
        for i, item in enumerate(recent_history):
            action = item.get('action', 'Unknown')
            obs = item.get('observation', '')
            obs_short = obs[:200] + "..." if len(obs) > 200 else obs
            prompt_parts.append(f"Step {i+1}: {action}")
            if obs_short:
                prompt_parts.append(f"   Obs: {obs_short}")
    else:
        prompt_parts.append("(No actions taken yet)")
    
    prompt_parts.append("\n## Current State")
    prompt_parts.append(current_obs)
    
    prompt_parts.append("\n## Current Executable Commands")
    if admissible:
        unique_cmds = list(dict.fromkeys(admissible))[:120]
        admissible_commands_parts = []
        for cmd in unique_cmds:
            admissible_commands_parts.append(f"- {cmd}")
        prompt_parts.append("\n".join(admissible_commands_parts))
    else:
        prompt_parts.append("- (no specific commands)")
    
    # Add soft token section only in normal soft-token experiment.
    # For text-only ablation, set use_soft_tokens=False and keep this section out of the prompt.
    if use_soft_tokens:
        soft_section = build_soft_token_section(num_thought_tokens)
        prompt_parts.append(soft_section)
    
    prompt_parts.append("\n## Output Requirements (Strictly Follow!)")
    prompt_parts.append("Please select one from the executable commands above and output exactly.")
    prompt_parts.append("\n## Your Turn: Take Action!")
    prompt_parts.append("Use the above guidance and insights as a foundation, and now work on the following task:")
    prompt_parts.append(f"Your task is to: {task_text}")
    
    user_content = "\n".join(prompt_parts)
    
    messages = [
        {"role": "system", "content": THOR_EXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    # Tokenize
    text_prompt = executor_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = executor_tokenizer(text_prompt, padding=True, return_tensors="pt").to(executor_model.device)
    
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    if use_soft_tokens:
        # Get UNK token ID
        unk_token = get_qwen_unk_token()
        unk_token_id = executor_tokenizer.convert_tokens_to_ids(unk_token)

        # Find positions of UNK tokens (soft token placeholders)
        unk_positions = (input_ids[0] == unk_token_id).nonzero(as_tuple=True)[0]

        if len(unk_positions) < num_thought_tokens:
            print(f"Warning: Found {len(unk_positions)} UNK positions in executor input, expected {num_thought_tokens}")
        else:
            print(f"[DEBUG] Found {len(unk_positions)} UNK positions in executor input, will use last {num_thought_tokens} as soft token slots")

        # Take the last num_thought_tokens positions
        thought_positions = unk_positions[-num_thought_tokens:] if len(unk_positions) >= num_thought_tokens else unk_positions
        print(f"[DEBUG] thought_positions (len={len(thought_positions)}): {thought_positions.tolist()}")

        # Get input embeddings
        inputs_embeds = executor_model.get_input_embeddings()(input_ids)
        num_to_replace = 0
        selected_positions = torch.tensor([], device=input_ids.device, dtype=torch.long)
        all_positions_are_unk = False

        # Replace placeholder embeddings with sampled latent embeddings
        if len(thought_positions) > 0 and sampled_latent_embeddings is not None:
            num_to_replace = min(len(thought_positions), sampled_latent_embeddings.size(1))
            sampled_latent_embeddings = sampled_latent_embeddings.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

            selected_positions = thought_positions[:num_to_replace]
            selected_input_ids = input_ids[0, selected_positions]
            all_positions_are_unk = bool(torch.all(selected_input_ids == unk_token_id).item())
            print(f"[DEBUG] Replacing {num_to_replace} positions in inputs_embeds with sampled latent soft tokens")
            print(f"[DEBUG] selected_positions: {selected_positions.tolist()}, all_positions_are_unk={all_positions_are_unk}")
            print(f"[DEBUG] sampled_latent_embeddings.shape={tuple(sampled_latent_embeddings.shape)} (batch, tokens, dim), expected tokens={num_thought_tokens}")
            if num_to_replace != sampled_latent_embeddings.size(1):
                print(f"[DEBUG][WARNING] num_to_replace ({num_to_replace}) != sampled_latent_embeddings.size(1) ({sampled_latent_embeddings.size(1)})")

            inputs_embeds[0, selected_positions, :] = sampled_latent_embeddings[0, :num_to_replace, :]
        else:
            print("[DEBUG] No sampled latent embeddings available for executor injection.")

        # Generate using inputs_embeds
        _t_exec_start = time.time()
        generated_ids = executor_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
        )
        torch.cuda.synchronize()
        _t_exec_end = time.time()
        _executor_generate_sec = round(_t_exec_end - _t_exec_start, 4)
        executor_output_token_count = int(generated_ids.shape[-1])

        # Decode output (when using inputs_embeds, generated_ids only contains new tokens)
        output_text = executor_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        # Text-only ablation path: no soft token placeholders, no embedding replacement.
        _t_exec_start = time.time()
        generated_ids = executor_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
        )
        torch.cuda.synchronize()
        _t_exec_end = time.time()
        _executor_generate_sec = round(_t_exec_end - _t_exec_start, 4)
        new_token_ids = generated_ids[0, input_ids.size(1):]
        executor_output_token_count = int(new_token_ids.size(0))
        output_text = executor_tokenizer.batch_decode([new_token_ids], skip_special_tokens=True)[0]
    # print(f"[DEBUG] 👉👉👉Executor output: {output_text}")
    
    # Parse action from output (fallback to first admissible command or "look" if parsing fails)
    action = parse_action(output_text, admissible_commands=admissible)
    action = action.strip() if action else (admissible[0] if admissible else "look")
    
    input_fields = {
        "working_memory": working_memory,
        "task_text": task_text,
        "gpt_guidance": gpt_guidance,
        "current_obs": current_obs,
        "soft_tokens_injected": bool(use_soft_tokens and sampled_latent_embeddings is not None and num_to_replace > 0),
        "executor_found_unk_positions": int(len(unk_positions)) if use_soft_tokens else 0,
        "executor_selected_positions": [int(pos) for pos in selected_positions.tolist()] if use_soft_tokens else [],
        "executor_replaced_count": int(num_to_replace) if use_soft_tokens else 0,
        "all_positions_are_unk": bool(all_positions_are_unk) if use_soft_tokens else False,
        "executor_input_token_count": int(input_ids.shape[1]),
        "executor_output_token_count": executor_output_token_count,
        "executor_generate_sec": _executor_generate_sec,
    }

    return action, output_text.strip(), input_fields


# =============================================================================
# ScienceWorld Executor: Soft token injection + SentenceTransformer matching
# =============================================================================
@torch.no_grad()
def executor_generate_with_soft_tokens_sciworld(
    executor_model,
    executor_tokenizer,
    sampled_latent_embeddings: Optional[torch.Tensor],
    working_memory: str,
    task_text: str,
    action_templates: List[str],
    available_objects: List[str],
    current_inventory: str,
    valid_actions: List[str],
    sent_model,
    history: List[Dict[str, Any]],
    current_obs: str,
    gpt_guidance: str,
    num_thought_tokens: int,
    use_soft_tokens: bool = True,
    max_history_turns: int = 10,
    task_description: Optional[str] = None,
) -> Tuple[str, float, str, str, Dict[str, Any]]:
    """
    ScienceWorld executor with soft token injection and SentenceTransformer action matching.

    Returns:
        matched_action, match_score, raw_action, qwen_raw_response, input_fields
    """
    # Build executor prompt using scienceworld prompt builder
    user_content, qwen_input_fields = create_qwen_user_content_scienceworld(
        working_memory=working_memory,
        task_text=task_text,
        action_templates=action_templates,
        available_objects=available_objects,
        history=history,
        current_obs=current_obs,
        current_inventory=current_inventory,
        gpt_guidance=gpt_guidance,
        task_description=task_description,
    )

    # Add soft token section if enabled
    if use_soft_tokens:
        soft_section = build_soft_token_section(num_thought_tokens)
        user_content += soft_section

    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT_SCIENCEWORLD},
        {"role": "user", "content": user_content}
    ]

    # Tokenize
    text_prompt = executor_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = executor_tokenizer(text_prompt, padding=True, return_tensors="pt").to(executor_model.device)

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    if use_soft_tokens:
        unk_token = get_qwen_unk_token()
        unk_token_id = executor_tokenizer.convert_tokens_to_ids(unk_token)
        unk_positions = (input_ids[0] == unk_token_id).nonzero(as_tuple=True)[0]
        thought_positions = unk_positions[-num_thought_tokens:] if len(unk_positions) >= num_thought_tokens else unk_positions

        inputs_embeds = executor_model.get_input_embeddings()(input_ids)
        num_to_replace = 0
        selected_positions = torch.tensor([], device=input_ids.device, dtype=torch.long)
        all_positions_are_unk = False

        if len(thought_positions) > 0 and sampled_latent_embeddings is not None:
            num_to_replace = min(len(thought_positions), sampled_latent_embeddings.size(1))
            sampled_latent_embeddings = sampled_latent_embeddings.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            selected_positions = thought_positions[:num_to_replace]
            selected_input_ids = input_ids[0, selected_positions]
            all_positions_are_unk = bool(torch.all(selected_input_ids == unk_token_id).item())
            inputs_embeds[0, selected_positions, :] = sampled_latent_embeddings[0, :num_to_replace, :]

        _t_exec_start = time.time()
        generated_ids = executor_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
        )
        torch.cuda.synchronize()
        _t_exec_end = time.time()
        _executor_generate_sec = round(_t_exec_end - _t_exec_start, 4)
        executor_output_token_count = int(generated_ids.shape[-1])
        output_text = executor_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        _t_exec_start = time.time()
        generated_ids = executor_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
        )
        torch.cuda.synchronize()
        _t_exec_end = time.time()
        _executor_generate_sec = round(_t_exec_end - _t_exec_start, 4)
        new_token_ids = generated_ids[0, input_ids.size(1):]
        executor_output_token_count = int(new_token_ids.size(0))
        output_text = executor_tokenizer.batch_decode([new_token_ids], skip_special_tokens=True)[0]

    # Parse and match action using ScienceWorld-specific logic
    raw_action = parse_action_scienceworld(output_text)
    matched_action, match_score, top5 = match_action_scienceworld(
        raw_action, valid_actions, sent_model, threshold=ACTION_MATCH_THRESHOLD
    )

    input_fields = {
        **qwen_input_fields,
        "soft_tokens_injected": bool(use_soft_tokens and sampled_latent_embeddings is not None and num_to_replace > 0) if use_soft_tokens else False,
        "executor_found_unk_positions": int(len(unk_positions)) if use_soft_tokens else 0,
        "executor_selected_positions": [int(pos) for pos in selected_positions.tolist()] if use_soft_tokens else [],
        "executor_replaced_count": int(num_to_replace) if use_soft_tokens else 0,
        "all_positions_are_unk": bool(all_positions_are_unk) if use_soft_tokens else False,
        "executor_input_token_count": int(input_ids.shape[1]),
        "executor_output_token_count": executor_output_token_count,
        "executor_generate_sec": _executor_generate_sec,
    }

    return matched_action, match_score, raw_action, output_text.strip(), input_fields


# =============================================================================
# Text Environment Prompt for Manager (no image references)
# =============================================================================
def create_manager_qwen_user_content_text(
    working_memory: str,
    task_text: str,
    history: Optional[List[Dict[str, Any]]],
    current_obs: str,
    admissible_commands: Optional[List[str]],
    task_memory: Optional[Dict[str, Any]] = None,
    initial_observation: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build Manager Qwen user message content for TEXT environment (no image).
    Aligned with pre_process_thor's manager_text_content format.
    """
    prompt_parts = []
    
    prompt_parts.append("## 🎯 TASK DESCRIPTION")
    if initial_observation and initial_observation.strip():
        prompt_parts.append(initial_observation)
    prompt_parts.append("")
    prompt_parts.append(f"The ultimate goal is: **{task_text}**")
    
    prompt_parts.append("\n## 👁️ CURRENT STATE")
    prompt_parts.append(f"\n{current_obs}")

    prompt_parts.append("\n## 📚 TASK MEMORY")
    formated_task_memory = format_task_memory(task_memory)
    prompt_parts.append(formated_task_memory)
    
    prompt_parts.append("\n## 🧠 WORKING MEMORY")
    if working_memory and working_memory.strip():
        working_memory_processed = working_memory
    else:
        working_memory_processed = "(Memory is currently empty. If you find important objects, record them.)"
    prompt_parts.append(working_memory_processed)

    prompt_parts.append("\n## 👣 RECENT TRAJECTORY (Last 15 Steps)")
    prompt_parts.append("\nPay attention to the item states contained in the observation corresponding to each executed action")
    
    # Aligned with pre_process_thor: build action_history string
    action_history_parts = []
    if history:
        for i, item in enumerate(history[-15:], 1):
            action = item.get('action', 'Unknown')
            observation = item.get('observation', '')
            feedback = observation[:150] + "..." if len(observation) > 150 else observation
            action_history_parts.append(f"\nStep {i}: Action=[{action}]")
            if feedback:
                action_history_parts.append(f"\n> Observation=[{feedback}]")
        action_history = "\n".join(action_history_parts)
    else:
        action_history = "(No actions taken yet. This is the start of the episode.)"
    
    # Truncate action_history to 1500 characters (aligned with pre_process_thor)
    # if len(action_history) > 1500:
    #     action_history = action_history[:1500] + "\n... (older actions truncated)"
    prompt_parts.append(action_history)

    prompt_parts.append("\n## ⚡ DECISION PROTOCOL")
    prompt_parts.append("Analyze the input above explicitly:")
    prompt_parts.append("1. **Check Strategy:** Is the Executor stuck, looping, or deviating? (-> EMBODIED)")
    prompt_parts.append("2. **Check Memory:** Is there new info (locations/states) that contradicts or adds to Working Memory? (-> CONTEXT)")
    prompt_parts.append("2. **Check Progress:** If moving smoothly towards the goal with no new info? (-> NOACTION)")
    prompt_parts.append("\nOutput your decision strictly in the XML format defined in the system prompt.")
    
    text_content = "\n".join(prompt_parts)
    
    # Text-only content (no image)
    content = [{"type": "text", "text": text_content}]
    
    # Collect input fields for logging
    input_fields = {
        "task_text": task_text,
        "initial_observation": initial_observation if initial_observation else "",
        "current_obs": current_obs,
        "task_memory": formated_task_memory,
        # "working_memory": working_memory_processed,
        "action_history": action_history
    }
    
    return content, input_fields


# =============================================================================
# Worker Process
# =============================================================================
def worker_process(
    process_id: int,
    gpu_id: int,
    task_indices: List[int],
    game_file_list: List[str],
):
    """Worker process that handles a subset of tasks."""
    print(f"[Process {process_id}] Starting on GPU {gpu_id} with {len(task_indices)} tasks")
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Setup directories
    traj_path = os.path.join(OUTPUT_PATH, "trajectory")
    detailed_traj_path = os.path.join(OUTPUT_PATH, "detailed_trajectory")
    overview_file = os.path.join(OUTPUT_PATH, "overview.jsonl")
    
    # Load completed tasks
    completed_indices = set()
    if os.path.exists(overview_file):
        with FileLock(overview_file + ".lock"):
            with open(overview_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "env_index" in data:
                            completed_indices.add(data["env_index"])
                    except:
                        pass
    
    remaining_tasks = [idx for idx in task_indices if idx not in completed_indices]
    print(f"[Process {process_id}] Completed: {len(task_indices) - len(remaining_tasks)}, Remaining: {len(remaining_tasks)}")
    
    if not remaining_tasks:
        print(f"[Process {process_id}] All tasks completed!")
        return
    
    # ==========================================================================
    # Load Models
    # ==========================================================================
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
    
    # Load Executor Qwen model (text model, 8-bit quantized)
    print(f"[Process {process_id}] Loading Executor Qwen model (text)...")
    executor_tokenizer = AutoTokenizer.from_pretrained(EXECUTOR_QWEN_MODEL_NAME)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    executor_model = AutoModelForCausalLM.from_pretrained(
        EXECUTOR_QWEN_MODEL_NAME,
        quantization_config=quantization_config,
        device_map={"": 0},
    ).eval()
    print(f"[Process {process_id}] Executor Qwen model loaded.")
    
    # Auto-detect VL model based on model name
    is_vl_model = "VL" in MANAGER_BASE_MODEL or "vl" in MANAGER_BASE_MODEL.lower()
    
    if is_vl_model:
        print(f"[Process {process_id}] Loading Manager Qwen VL model...")
        manager_processor = AutoProcessor.from_pretrained(MANAGER_BASE_MODEL, use_fast=False)
        manager_tokenizer = AutoTokenizer.from_pretrained(MANAGER_BASE_MODEL)
        manager_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MANAGER_BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map={"":  0},
        )
        manager_base_model = manager_model
        print(f"[Process {process_id}] Manager VL model loaded.")
    else:
        print(f"[Process {process_id}] Loading Manager text-only Qwen model...")
        manager_processor = None
        manager_tokenizer = AutoTokenizer.from_pretrained(MANAGER_BASE_MODEL)
        manager_model = AutoModelForCausalLM.from_pretrained(
            MANAGER_BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map={"":  0},
        )
        manager_base_model = manager_model
        print(f"[Process {process_id}] Manager text-only model loaded.")
    # Load projection layer for soft token mapping
    projection = None
    manager_hidden_size = manager_model.config.hidden_size
    executor_hidden_size = executor_model.config.hidden_size
    
    if LOAD_MODE == "base":
        # Use base model directly without loading fine-tuned weights
        print(f"[Process {process_id}] Using base model directly (no fine-tuned weights)")
        # projection remains None, no additional weights loaded
    elif LOAD_MODE == "checkpoint":
        # Load from training checkpoint (pytorch_model.bin)
        print(f"[Process {process_id}] Loading from checkpoint: {CHECKPOINT_PATH}")
        if USE_SOFT_TOKENS:
            projection = load_from_checkpoint(
                CHECKPOINT_PATH,
                manager_model,
                input_dim=manager_hidden_size,
                output_dim=executor_hidden_size,
                device="cuda"
            )
            print(f"[Process {process_id}] Checkpoint loaded: projection {manager_hidden_size} -> {executor_hidden_size}")
        else:
            # Load only assistant model from checkpoint
            checkpoint_file = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")
            full_state_dict = torch.load(checkpoint_file, map_location='cuda', weights_only=False)
            assistant_state_dict = {k[len("assistant_model."):]: v for k, v in full_state_dict.items() if k.startswith("assistant_model.")}
            # Handle PeftModel (LoRA) checkpoint: remap keys and merge LoRA weights
            assistant_state_dict, is_peft = _convert_peft_state_dict(assistant_state_dict)
            if is_peft:
                print(f"[Process {process_id}] Converted PeftModel keys, after merge: {len(assistant_state_dict)} keys")
            load_result = manager_model.load_state_dict(assistant_state_dict, strict=False)
            if load_result.missing_keys:
                print(f"[Process {process_id}] Note: {len(load_result.missing_keys)} keys not in checkpoint")
            if load_result.unexpected_keys:
                print(f"[Process {process_id}] Warning: {len(load_result.unexpected_keys)} unexpected keys in checkpoint")
            print(f"[Process {process_id}] Loaded {len(assistant_state_dict)} assistant_model keys from checkpoint")
    elif LOAD_MODE == "bin":
        # Load from separate .bin files
        print(f"[Process {process_id}] Loading from .bin files...")
        state_dict = torch.load(MANAGER_WEIGHTS_PATH, map_location='cuda')
        load_result = manager_model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            print(f"[Process {process_id}] Note: {len(load_result.missing_keys)} keys not in saved weights (visual encoder kept at pretrained)")
        if load_result.unexpected_keys:
            print(f"[Process {process_id}] Warning: {len(load_result.unexpected_keys)} unexpected keys in saved weights")
        
        if USE_SOFT_TOKENS:
            print(f"[Process {process_id}] Loading projection layer...")
            projection = load_projection_layer(
                PROJECTION_WEIGHTS_PATH,
                input_dim=manager_hidden_size,
                output_dim=executor_hidden_size,
                device="cuda"
            )
            print(f"[Process {process_id}] Projection layer loaded: {manager_hidden_size} -> {executor_hidden_size}")
    else:
        raise ValueError(f"Invalid LOAD_MODE: {LOAD_MODE}. Must be 'base', 'bin', or 'checkpoint'.")
    
    manager_model = manager_model.eval()
    print(f"[Process {process_id}] Manager Qwen model loaded with fine-tuned weights (mode={LOAD_MODE}).")
    
    if not USE_SOFT_TOKENS:
        print(f"[Process {process_id}] Soft tokens disabled (USE_SOFT_TOKENS=False), projection not loaded.")
    
    # ==========================================================================
    # Setup Memcompiler
    # ==========================================================================
    print(f"[Process {process_id}] Loading Memcompiler from: {MEMCOMPILER_PATH} (UPDATE_MEMORY={UPDATE_MEMORY})")
    from core.memory.core_memory.memcompiler import Memcompiler
    from core.utils import EmbeddingFunc
    from core.memory.common import MASMessage
    
    print(f"[Process {process_id}] Loading Memcompiler from {MEMCOMPILER_PATH}...")
    
    # Use same global_config and wrapper pattern as Thor_gpt_qwen_train60_multiprocess
    global_config = {
        "working_dir": MEMCOMPILER_PATH,
        "hop": 1,
        "start_insights_threshold": 5,
        "rounds_per_insights": 5,
        "insights_point_num": 5,
    }
    
    
    # Use LLM as Memcompiler's internal LLM for insights/key steps extraction
    if API_TYPE == "gemini":
        gemini_key = GEMINI_CONFIG["api_key"]
        if not gemini_key:
            print(f"[Process {process_id}] ERROR: GEMINI_API_KEY not set! Set env var or hardcode in GEMINI_CONFIG")
            return
        llm_model = GeminiLLMWrapper(
            api_key=gemini_key,
            model=GEMINI_CONFIG["model"],
            base_url=GEMINI_CONFIG["base_url"],
        )
        print(f"[Process {process_id}] Using Gemini API with model: {GEMINI_CONFIG['model']}")
    elif API_TYPE == "azure":
        endpoint_config = AZURE_OPENAI_CONFIG["endpoints"].get(GPT_MODEL, {})
        gpt_client = AzureOpenAI(
            api_key=AZURE_OPENAI_CONFIG["api_key"],
            api_version=endpoint_config["api_version"],
            azure_endpoint=endpoint_config["azure_endpoint"]
        )
        print(f"[Process {process_id}] Using Azure OpenAI API with model: {GPT_MODEL}")
        llm_model = GPTLLMWrapper(client=gpt_client, model=GPT_MODEL)
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if not api_key:
            print(f"[Process {process_id}] ERROR: OPENAI_API_KEY not set!")
            return
        gpt_client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"[Process {process_id}] Using standard OpenAI API")
        llm_model = GPTLLMWrapper(client=gpt_client, model=GPT_MODEL)
    # llm_model = QwenLLMWrapper(model=executor_model, tokenizer=executor_tokenizer)
    embedding_func = EmbeddingFunc(model_type=SENT_TRANSFORMER_MODEL)
    
    # Use SafeMemcompiler with ChromaDB HTTP client when UPDATE_MEMORY is True
    if UPDATE_MEMORY:
        SafeMemcompiler = create_safe_memory_class()
        memory = SafeMemcompiler(
            namespace="memcompiler",
            global_config=global_config,
            llm_model=llm_model,
            embedding_func=embedding_func,
        )
    else:
        memory = Memcompiler(
            namespace="memcompiler",
            global_config=global_config,
            llm_model=llm_model,
            embedding_func=embedding_func,
        )
    print(f"[Process {process_id}] Memcompiler loaded. Memory size: {memory.memory_size}")
    
    # ==========================================================================
    # Setup Environment
    # ==========================================================================
    print(f"[Process {process_id}] Loading environment...")
    
    import alfworld.agents.modules.generic as generic
    import alfworld.agents.environment.alfred_tw_env as alfred_tw_env
    
    generic.ALFWORLD_CONFIG = CONFIG_FILE
    config = generic.load_config()
    
    # Directly instantiate the text environment (AlfredTWEnv)
    thor_env = alfred_tw_env.AlfredTWEnv(config, train_eval='eval_out_of_distribution')
    env_type = 'AlfredTWEnv'
    
    is_text_env = True
    print(f"[Process {process_id}] Environment type: {env_type}, is_text_env: {is_text_env}")
    
    # Setup text environment
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    
    def reset_to_game(game_index: int):
        """Reset to specific game for text environment."""
        task_file = game_file_list[game_index]
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games(
            [task_file], 
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=50,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos]
        )
        env = textworld.gym.make(env_id)
        obs, infos = env.reset()
        if isinstance(obs, (list, tuple)) and len(obs) > 0:
            obs_str = obs[0]
        else:
            obs_str = obs
        return [obs_str], infos, env
    
    env = None
    
    # Statistics
    stats = {'total': 0, 'success': 0, 'failed': 0}
    all_step_stats = []  # worker-level accumulator for step stats
    
    # ==========================================================================
    # Main Loop
    # ==========================================================================
    for task_num, target_idx in enumerate(remaining_tasks):
        try:
            print(f"\n[Process {process_id}] Task {task_num + 1}/{len(remaining_tasks)} (env index: {target_idx})")
            
            # Reset to target game
            obs, info, new_env = reset_to_game(target_idx)
            if new_env is not None:
                env = new_env
            
            initial_obs_raw = obs[0] if isinstance(obs, (list, tuple)) else obs
            gamefile = info.get('extra.gamefile', ['unknown'])
            if isinstance(gamefile, list):
                gamefile = gamefile[0]
            
            task_text = extract_task(initial_obs_raw)
            # Extract only the environment description (aligned with training flow)
            initial_obs = extract_initial_observation(initial_obs_raw)
            print(f"[Process {process_id}] Task: {task_text[:80]}...")
            
            # Get Task Memory from Memcompiler (same as Thor script)
            # try:
            #     successful_tasks, failed_tasks, insights = memory.retrieve_memory(
            #         query_task=task_text,
            #         successful_topk=1,
            #         failed_topk=0,
            #         insight_topk=5,
            #         threshold=0.0
            #     )
            #     print(f"[Process {process_id}] Task Memory: {len(successful_tasks)} success, {len(failed_tasks)} fail, {len(insights)} insights")
            # except Exception as e:
            #     print(f"[Process {process_id}] Memcompiler retrieval error: {e}")
            #     successful_tasks, failed_tasks, insights = [], [], []

            successful_tasks, failed_tasks, insights = memory.retrieve_memory(
                query_task=task_text,
                successful_topk=1,
                failed_topk=0,
                insight_topk=5,
                threshold=0.0
            )
            print(f"[Process {process_id}] Task Memory: {len(successful_tasks)} success, {len(failed_tasks)} fail, {len(insights)} insights")
            # Generate runtime key steps for successful tasks
            runtime_key_steps = []
            for success_task in list(successful_tasks or [])[:3]:
                key_steps = generate_runtime_key_steps(success_task=success_task, llm_model=llm_model)
                runtime_key_steps.append(key_steps)
            
            # Build task_memory dict for call_manager_qwen_text
            task_memory = {
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "insights": insights,
                "runtime_key_steps": runtime_key_steps,
            }
            print(f"[Process {process_id}] Task Memory: {len(successful_tasks or [])} success, {len(failed_tasks or [])} fail, {len(insights or [])} insights, {len([k for k in runtime_key_steps if k])} key_steps")
            
            # Episode loop
            working_memory = ""
            history = []
            detailed_steps = []
            episode_done = False
            episode_won = False
            failure_reason = None
            step_count = 0
            episode_step_stats = []  # per-step stats: (step_time, input_tokens)
            
            while not episode_done:
                step_count += 1
                if step_count > MAX_STEPS:
                    print(f"[Process {process_id}] Max steps reached.")
                    failure_reason = "max_steps_exceeded"
                    break
                
                # Get admissible commands
                admissible = None
                if isinstance(info, dict) and "admissible_commands" in info:
                    try:
                        ac = info["admissible_commands"]
                        if ac:
                            if isinstance(ac, (list, tuple)) and len(ac) > 0:
                                if isinstance(ac[0], (list, tuple)):
                                    admissible = list(ac[0])
                                else:
                                    admissible = list(ac)
                    except:
                        pass
                
                current_obs = obs[0] if isinstance(obs, (list, tuple)) else obs
                
                latent_sample_seed = make_latent_sample_seed(process_id, target_idx, step_count)

                if ENABLE_STEP_STATS:
                    _step_start_time = time.time()

                # Step 1: Call Assistant model to generate XML response AND extract sampled soft latent embeddings
                # This uses assistant_template (aligned with pre_process_thor training format)
                # try:
                #     sampled_latent_embeddings, manager_raw_response, manager_input_fields = call_assistant_with_soft_tokens(
                #         manager_model=manager_model,
                #         manager_processor=manager_processor,
                #         manager_tokenizer=manager_tokenizer,
                #         projection=projection,
                #         working_memory=working_memory,
                #         task_text=task_text,
                #         history=history,
                #         current_obs=current_obs,
                #         task_memory=task_memory,
                #         num_thought_tokens=NUM_THOUGHT_TOKENS,
                #         latent_sample_seed=latent_sample_seed,
                #         initial_observation=initial_obs,
                #     )
                # except Exception as e:
                #     print(f"[Process {process_id}] Assistant call error: {e}")
                #     traceback.print_exc()
                #     sampled_latent_embeddings = None
                #     manager_raw_response = "<response_type>NOACTION</response_type>"
                #     manager_input_fields = {"latent_sample_seed": latent_sample_seed}
                sampled_latent_embeddings, manager_raw_response, manager_input_fields = call_assistant_with_soft_tokens(
                    manager_model=manager_model,
                    manager_processor=manager_processor,
                    manager_tokenizer=manager_tokenizer,
                    projection=projection,
                    working_memory=working_memory,
                    task_text=task_text,
                    history=history,
                    current_obs=current_obs,
                    task_memory=task_memory,
                    num_thought_tokens=NUM_THOUGHT_TOKENS,
                    latent_sample_seed=latent_sample_seed,
                    initial_observation=initial_obs,
                )
                print(f"[Process {process_id}] 👉👉👉Manager raw response: {manager_raw_response}")
                
                # Step 2: Parse Manager response and build enhanced guidance
                parsed_gpt = parse_gpt_response(manager_raw_response)
                manager_type = parsed_gpt.get("type", "NOACTION")
                
                # Build enhanced guidance: {guidance} REASON: {brief_reason}
                def build_enhanced_guidance(parsed: Dict[str, Any]) -> Optional[str]:
                    guidance = parsed.get("guidance", "")
                    brief_reason = parsed.get("brief_reason", "")
                    if brief_reason:
                        brief_reason = re.sub(r'\bexecutor\b', r'\g<0>(You)', brief_reason, flags=re.IGNORECASE)
                    if guidance:
                        if brief_reason:
                            return f"{guidance} REASON: {brief_reason}"
                        return guidance
                    return None
                
                manager_guidance = None
                if manager_type == "EMBODIED":
                    manager_guidance = build_enhanced_guidance(parsed_gpt)
                elif manager_type == "CONTEXT":
                    working_memory = apply_memory_operation(working_memory, parsed_gpt)
                elif manager_type == "HYBRID":
                    manager_guidance = build_enhanced_guidance(parsed_gpt)
                    working_memory = apply_memory_operation(working_memory, parsed_gpt)
                # NOACTION: manager_guidance stays None, no memory update
                
                # Step 3: Call Executor with soft tokens
                action, qwen_raw_response, qwen_input_fields = executor_generate_with_soft_tokens(
                    executor_model=executor_model,
                    executor_tokenizer=executor_tokenizer,
                    sampled_latent_embeddings=sampled_latent_embeddings,
                    working_memory=working_memory,
                    task_text=task_text,
                    admissible=admissible,
                    history=history,
                    current_obs=current_obs,
                    gpt_guidance=manager_guidance,
                    num_thought_tokens=NUM_THOUGHT_TOKENS,
                    use_soft_tokens=USE_SOFT_TOKENS,
                    max_history_turns=12
                )
                print(f"[Process {process_id}] 👉👉👉Executor raw response: {qwen_raw_response}")

                if ENABLE_STEP_STATS:
                    _step_end_time = time.time()
                    _step_time = _step_end_time - _step_start_time
                    _executor_input_tokens = qwen_input_fields.get("executor_input_token_count", 0)
                    episode_step_stats.append({
                        "task_idx": target_idx,
                        "step": step_count,
                        "step_time_sec": round(_step_time, 4),
                        "input_tokens": _executor_input_tokens,
                        "manager_pass1_input_tokens": manager_input_fields.get("manager_pass1_input_token_count", 0),
                        "manager_pass2_input_tokens": manager_input_fields.get("manager_pass2_input_token_count", 0),
                        "manager_output_tokens": manager_input_fields.get("manager_output_token_count", 0),
                        "executor_output_tokens": qwen_input_fields.get("executor_output_token_count", 0),
                        "manager_generate_sec": manager_input_fields.get("manager_generate_sec", 0.0),
                        "manager_soft_forward_sec": manager_input_fields.get("manager_soft_forward_sec", 0.0),
                        "executor_generate_sec": qwen_input_fields.get("executor_generate_sec", 0.0),
                    })
                
                # Step 5: Execute action
                obs_result, scores, dones, info = env.step([action])
                if isinstance(obs_result, (list, tuple)) and len(obs_result) > 0:
                    observation = obs_result[0]
                else:
                    observation = obs_result
                obs = [observation]
                dones = list(dones) if isinstance(dones, (list, tuple)) else [dones]
                
                history.append({"action": action, "observation": observation})
                
                # Record step
                step_record = {
                    "step": step_count,
                    "manager": {
                        "input": manager_input_fields,
                        "output": {"raw": manager_raw_response}
                    },
                    "executor": {
                        "input": qwen_input_fields,
                        "output": {"raw": qwen_raw_response, "action": action}
                    },
                    "observation": observation
                }
                detailed_steps.append(step_record)
                
                # Check if done
                if dones[0]:
                    episode_done = True
                    won_value = info.get('won', [False])[0] if isinstance(info.get('won'), list) else info.get('won', False)
                    episode_won = bool(won_value)
                    print(f"[Process {process_id}] Episode finished! Won: {episode_won}, Steps: {len(history)}")
            
            # Update stats
            stats['total'] += 1
            if episode_won:
                stats['success'] += 1
            else:
                stats['failed'] += 1

            if ENABLE_STEP_STATS:
                all_step_stats.extend(episode_step_stats)
            
            # Save trajectory
            traj_file = os.path.join(traj_path, f"{target_idx}.json")
            out_traj = {
                "env_index": target_idx,
                "gamefile": gamefile,
                "task": task_text,
                "won": episode_won,
                "failure_reason": failure_reason,
                "steps": len(history),
                "trajectory": history
            }
            with open(traj_file, "w") as f:
                json.dump(out_traj, f, indent=4)
            
            # Save detailed trajectory
            detailed_file = os.path.join(detailed_traj_path, f"{target_idx}.json")
            detailed_out = {
                "env_index": target_idx,
                "gamefile": gamefile,
                "task": task_text,
                "won": episode_won,
                "failure_reason": failure_reason,
                "task_memory_summary": f"{len(successful_tasks or [])} success, {len(failed_tasks or [])} fail, {len(insights or [])} insights",
                "steps": detailed_steps
            }
            with open(detailed_file, "w") as f:
                json.dump(detailed_out, f, indent=4)
            
            # Append to overview
            out_overview = {
                "env_index": target_idx,
                "task": task_text,
                "won": episode_won,
                "failure_reason": failure_reason,
                "steps": len(history),
                "gamefile": gamefile
            }
            with FileLock(overview_file + ".lock"):
                with open(overview_file, "a") as f:
                    f.write(json.dumps(out_overview) + "\n")
            
            # Add to Memcompiler if UPDATE_MEMORY is enabled
            if UPDATE_MEMORY:
                # try:
                #     mas_message = MASMessage(
                #         task_main=task_text,
                #         task_description=task_text,
                #         label=episode_won,
                #     )
                #     mas_message.add_extra_field("gamefile", gamefile)
                #     for h in history:
                #         mas_message.move_state(h["action"], h["observation"])

                #     # Add to memory
                #     memory.add_memory(mas_message)
                #     print(f"[Process {process_id}] Added task to Memcompiler. New size: {memory.memory_size}")
                # except Exception as e:
                #     print(f"[Process {process_id}] Failed to add memory: {e}")
                mas_message = MASMessage(
                    task_main=task_text,
                    task_description=task_text,
                    label=episode_won,
                )
                mas_message.add_extra_field("gamefile", gamefile)
                for h in history:
                    mas_message.move_state(h["action"], h["observation"])

                # Add to memory
                memory.add_memory(mas_message)
                print(f"[Process {process_id}] Added task to Memcompiler. New size: {memory.memory_size}")
            # Cleanup text env
            if env is not None:
                # try:
                #     env.close()
                # except:
                #     pass
                env.close()
                env = None
                
        except Exception as e:
            print(f"[Process {process_id}] ❌❌❌Error on task {target_idx}: {e}")
            traceback.print_exc()
            continue
    
    # Print final stats
    print(f"\n[Process {process_id}] FINISHED")
    print(f"Total: {stats['total']}, Success: {stats['success']}, Failed: {stats['failed']}")
    if stats['total'] > 0:
        print(f"Accuracy: {stats['success'] / stats['total'] * 100:.2f}%")

    # Write step stats file
    if ENABLE_STEP_STATS and all_step_stats:
        step_times = [s["step_time_sec"] for s in all_step_stats]
        input_tokens_list = [s["input_tokens"] for s in all_step_stats]
        manager_pass1_input_tokens_list = [s.get("manager_pass1_input_tokens", 0) for s in all_step_stats]
        manager_pass2_input_tokens_list = [s.get("manager_pass2_input_tokens", 0) for s in all_step_stats]
        manager_output_tokens_list = [s.get("manager_output_tokens", 0) for s in all_step_stats]
        executor_output_tokens_list = [s.get("executor_output_tokens", 0) for s in all_step_stats]
        sorted_times = sorted(step_times)
        sorted_tokens = sorted(input_tokens_list)
        n = len(sorted_times)
        median_time = sorted_times[n // 2] if n % 2 == 1 else (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        median_tokens = sorted_tokens[n // 2] if n % 2 == 1 else (sorted_tokens[n // 2 - 1] + sorted_tokens[n // 2]) / 2

        # Sub-phase timing lists
        mgr_gen_times = [s.get("manager_generate_sec", 0.0) for s in all_step_stats]
        mgr_fwd_times = [s.get("manager_soft_forward_sec", 0.0) for s in all_step_stats]
        exec_gen_times = [s.get("executor_generate_sec", 0.0) for s in all_step_stats]

        def _phase_stats(vals):
            sv = sorted(vals)
            m = len(sv)
            med = sv[m // 2] if m % 2 == 1 else (sv[m // 2 - 1] + sv[m // 2]) / 2
            return {
                "avg": round(sum(sv) / m, 4),
                "median": round(med, 4),
                "max": round(max(sv), 4),
                "min": round(min(sv), 4),
            }

        def _token_stats(vals):
            sv = sorted(vals)
            m = len(sv)
            med = sv[m // 2] if m % 2 == 1 else (sv[m // 2 - 1] + sv[m // 2]) / 2
            return {
                "avg": round(sum(sv) / m, 2),
                "median": round(med, 2),
                "max": max(sv),
                "min": min(sv),
            }

        summary = {
            "total_steps": n,
            "avg_step_time_sec": round(sum(step_times) / n, 4),
            "median_step_time_sec": round(median_time, 4),
            "max_step_time_sec": round(max(step_times), 4),
            "min_step_time_sec": round(min(step_times), 4),
            "avg_input_tokens": round(sum(input_tokens_list) / n, 2),
            "median_input_tokens": round(median_tokens, 2),
            "max_input_tokens": max(input_tokens_list),
            "min_input_tokens": min(input_tokens_list),
            "manager_pass1_input_tokens": _token_stats(manager_pass1_input_tokens_list),
            "manager_pass2_input_tokens": _token_stats(manager_pass2_input_tokens_list),
            "manager_output_tokens": _token_stats(manager_output_tokens_list),
            "executor_output_tokens": _token_stats(executor_output_tokens_list),
            "manager_generate": _phase_stats(mgr_gen_times),
            "manager_soft_forward": _phase_stats(mgr_fwd_times),
            "executor_generate": _phase_stats(exec_gen_times),
        }
        stats_output = {"summary": summary, "per_step": all_step_stats}
        stats_file = os.path.join(OUTPUT_PATH, f"step_stats_process_{process_id}.json")
        with open(stats_file, "w") as f:
            json.dump(stats_output, f, indent=2)
        print(f"\n[Process {process_id}] Step Stats Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"  Saved to: {stats_file}")


# =============================================================================
# ScienceWorld Worker Process
# =============================================================================
def worker_process_sciworld(
    process_id: int,
    gpu_id: int,
    task_configs: List[Dict[str, Any]],
    task_indices: List[int],
):
    """Worker process for ScienceWorld evaluation tasks."""
    print(f"[SW-Process {process_id}] Starting on GPU {gpu_id} with {len(task_indices)} tasks")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Setup directories
    traj_path = os.path.join(SCIWORLD_OUTPUT_PATH, "trajectory")
    detailed_traj_path = os.path.join(SCIWORLD_OUTPUT_PATH, "detailed_trajectory")
    overview_file = os.path.join(SCIWORLD_OUTPUT_PATH, "overview.jsonl")

    # Load completed tasks
    completed_ids = set()
    if os.path.exists(overview_file):
        with FileLock(overview_file + ".lock"):
            with open(overview_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "task_id" in data:
                            completed_ids.add(data["task_id"])
                    except:
                        pass

    remaining_indices = [idx for idx in task_indices if task_configs[idx].get("id", idx) not in completed_ids]
    print(f"[SW-Process {process_id}] Completed: {len(task_indices) - len(remaining_indices)}, Remaining: {len(remaining_indices)}")

    if not remaining_indices:
        print(f"[SW-Process {process_id}] All tasks completed!")
        return

    # ==========================================================================
    # Load Models (same as ALFWorld worker)
    # ==========================================================================
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration

    print(f"[SW-Process {process_id}] Loading Executor Qwen model (text)...")
    executor_tokenizer = AutoTokenizer.from_pretrained(EXECUTOR_QWEN_MODEL_NAME)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    executor_model = AutoModelForCausalLM.from_pretrained(
        EXECUTOR_QWEN_MODEL_NAME,
        quantization_config=quantization_config,
        device_map={"": 0},
    ).eval()
    print(f"[SW-Process {process_id}] Executor Qwen model loaded.")

    is_vl_model = "VL" in MANAGER_BASE_MODEL or "vl" in MANAGER_BASE_MODEL.lower()

    if is_vl_model:
        print(f"[SW-Process {process_id}] Loading Manager Qwen VL model...")
        manager_processor = AutoProcessor.from_pretrained(MANAGER_BASE_MODEL, use_fast=False)
        manager_tokenizer = AutoTokenizer.from_pretrained(MANAGER_BASE_MODEL)
        manager_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MANAGER_BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
    else:
        print(f"[SW-Process {process_id}] Loading Manager text-only Qwen model...")
        manager_processor = None
        manager_tokenizer = AutoTokenizer.from_pretrained(MANAGER_BASE_MODEL)
        manager_model = AutoModelForCausalLM.from_pretrained(
            MANAGER_BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )

    # Load projection layer
    projection = None
    manager_hidden_size = manager_model.config.hidden_size
    executor_hidden_size = executor_model.config.hidden_size

    if LOAD_MODE == "base":
        print(f"[SW-Process {process_id}] Using base model directly (no fine-tuned weights)")
    elif LOAD_MODE == "checkpoint":
        print(f"[SW-Process {process_id}] Loading from checkpoint: {CHECKPOINT_PATH}")
        if USE_SOFT_TOKENS:
            projection = load_from_checkpoint(
                CHECKPOINT_PATH, manager_model,
                input_dim=manager_hidden_size, output_dim=executor_hidden_size, device="cuda"
            )
        else:
            checkpoint_file = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")
            full_state_dict = torch.load(checkpoint_file, map_location='cuda', weights_only=False)
            assistant_state_dict = {k[len("assistant_model."):]: v for k, v in full_state_dict.items() if k.startswith("assistant_model.")}
            assistant_state_dict, _ = _convert_peft_state_dict(assistant_state_dict)
            manager_model.load_state_dict(assistant_state_dict, strict=False)
    elif LOAD_MODE == "bin":
        print(f"[SW-Process {process_id}] Loading from .bin files...")
        state_dict = torch.load(MANAGER_WEIGHTS_PATH, map_location='cuda')
        manager_model.load_state_dict(state_dict, strict=False)
        if USE_SOFT_TOKENS:
            projection = load_projection_layer(
                PROJECTION_WEIGHTS_PATH,
                input_dim=manager_hidden_size, output_dim=executor_hidden_size, device="cuda"
            )
    else:
        raise ValueError(f"Invalid LOAD_MODE: {LOAD_MODE}")

    manager_model = manager_model.eval()
    print(f"[SW-Process {process_id}] Manager model loaded (mode={LOAD_MODE}).")

    # Load SentenceTransformer for action matching
    from sentence_transformers import SentenceTransformer
    sent_model = SentenceTransformer(SENT_TRANSFORMER_MODEL)
    print(f"[SW-Process {process_id}] SentenceTransformer loaded: {SENT_TRANSFORMER_MODEL}")

    # ==========================================================================
    # Setup Memcompiler (with independent SCIWORLD_CHROMA_PORT)
    # ==========================================================================
    from core.memory.core_memory.memcompiler import Memcompiler
    from core.utils import EmbeddingFunc
    from core.memory.common import MASMessage

    global_config = {
        "working_dir": SCIWORLD_MEMCOMPILER_PATH,
        "hop": 1,
        "start_insights_threshold": 5,
        "rounds_per_insights": 5,
        "insights_point_num": 5,
    }

    if API_TYPE == "gemini":
        gemini_key = GEMINI_CONFIG["api_key"]
        if not gemini_key:
            print(f"[SW-Process {process_id}] ERROR: GEMINI_API_KEY not set! Set env var or hardcode in GEMINI_CONFIG")
            return
        llm_model = GeminiLLMWrapper(
            api_key=gemini_key,
            model=GEMINI_CONFIG["model"],
            base_url=GEMINI_CONFIG["base_url"],
        )
        print(f"[SW-Process {process_id}] Using Gemini API with model: {GEMINI_CONFIG['model']}")
    elif API_TYPE == "azure":
        endpoint_config = AZURE_OPENAI_CONFIG["endpoints"].get(GPT_MODEL, {})
        gpt_client = AzureOpenAI(
            api_key=AZURE_OPENAI_CONFIG["api_key"],
            api_version=endpoint_config["api_version"],
            azure_endpoint=endpoint_config["azure_endpoint"]
        )
        llm_model = GPTLLMWrapper(client=gpt_client, model=GPT_MODEL)
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if not api_key:
            print(f"[SW-Process {process_id}] ERROR: OPENAI_API_KEY not set!")
            return
        gpt_client = OpenAI(api_key=api_key, base_url=base_url)
        llm_model = GPTLLMWrapper(client=gpt_client, model=GPT_MODEL)
    embedding_func = EmbeddingFunc(model_type=SENT_TRANSFORMER_MODEL)

    if UPDATE_MEMORY:
        SafeMemcompiler = create_safe_memory_class(chroma_port=SCIWORLD_CHROMA_PORT)
        memory = SafeMemcompiler(
            namespace="memcompiler",
            global_config=global_config,
            llm_model=llm_model,
            embedding_func=embedding_func,
        )
    else:
        memory = Memcompiler(
            namespace="memcompiler",
            global_config=global_config,
            llm_model=llm_model,
            embedding_func=embedding_func,
        )
    print(f"[SW-Process {process_id}] Memcompiler loaded. Memory size: {memory.memory_size}")

    # ==========================================================================
    # Setup ScienceWorld Environment
    # ==========================================================================
    sw_env = ScienceWorldEnvWrapper(env_config={}, max_trials=SCIWORLD_MAX_STEPS)
    print(f"[SW-Process {process_id}] ScienceWorldEnvWrapper initialized.")

    stats = {'total': 0, 'success': 0, 'failed': 0, 'total_jvm_score': 0.0, 'total_progress': 0.0}
    all_step_stats = []

    # ==========================================================================
    # Main Loop
    # ==========================================================================
    for task_num, cfg_idx in enumerate(remaining_indices):
        task_config = task_configs[cfg_idx]
        task_id = task_config.get("id", cfg_idx)
        task_name = task_config.get("task_name", "unknown")
        variation_idx = task_config.get("variation_idx", -1)

        try:
            print(f"\n[SW-Process {process_id}] Task {task_num + 1}/{len(remaining_indices)} "
                  f"(id={task_id}, task={task_name}, var={variation_idx})")

            task_text, task_description = sw_env.set_env(task_config)
            print(f"[SW-Process {process_id}] Task: {task_text[:80]}...")

            # Get Task Memory from Memcompiler
            successful_tasks, failed_tasks, insights = memory.retrieve_memory(
                query_task=task_text,
                successful_topk=1,
                failed_topk=0,
                insight_topk=5,
                threshold=0.0
            )
            print(f"[SW-Process {process_id}] Task Memory: {len(successful_tasks or [])} success, "
                  f"{len(failed_tasks or [])} fail, {len(insights or [])} insights")

            runtime_key_steps = []
            for success_task in list(successful_tasks or [])[:3]:
                key_steps = generate_runtime_key_steps(success_task=success_task, llm_model=llm_model)
                runtime_key_steps.append(key_steps)

            task_memory = {
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "insights": insights,
                "runtime_key_steps": runtime_key_steps,
            }

            # Episode loop
            working_memory = ""
            history = []
            detailed_steps = []
            episode_done = False
            episode_won = False
            failure_reason = None
            step_count = 0
            episode_step_stats = []

            while not episode_done:
                step_count += 1
                if step_count > SCIWORLD_MAX_STEPS:
                    print(f"[SW-Process {process_id}] Max steps reached.")
                    failure_reason = "max_steps_exceeded"
                    break

                # Get environment info
                current_obs = sw_env.get_observation()
                current_inventory = sw_env.get_inventory()
                action_templates = sw_env.get_action_templates()
                available_objects = sw_env.get_available_objects()
                valid_actions = sw_env.get_valid_actions()
                current_score = sw_env.get_score()

                latent_sample_seed = make_latent_sample_seed(process_id, task_id, step_count)

                if ENABLE_STEP_STATS:
                    _step_start_time = time.time()

                # Step 1: Call Manager with soft tokens (ScienceWorld version)
                sampled_latent_embeddings, manager_raw_response, manager_input_fields = call_assistant_with_soft_tokens_sciworld(
                    manager_model=manager_model,
                    manager_processor=manager_processor,
                    manager_tokenizer=manager_tokenizer,
                    projection=projection,
                    working_memory=working_memory,
                    task_text=task_text,
                    history=history,
                    current_obs=current_obs,
                    current_score=current_score,
                    task_memory=task_memory,
                    num_thought_tokens=NUM_THOUGHT_TOKENS,
                    latent_sample_seed=latent_sample_seed,
                    initial_observation=task_description,
                    few_shots=None,
                )
                print(f"[SW-Process {process_id}] Manager raw response: {manager_raw_response[:200]}")

                # Step 2: Parse Manager response
                parsed_gpt = parse_gpt_response(manager_raw_response)
                manager_type = parsed_gpt.get("type", "NOACTION")

                def build_enhanced_guidance(parsed: Dict[str, Any]) -> Optional[str]:
                    guidance = parsed.get("guidance", "")
                    brief_reason = parsed.get("brief_reason", "")
                    if brief_reason:
                        brief_reason = re.sub(r'\bexecutor\b', r'\g<0>(You)', brief_reason, flags=re.IGNORECASE)
                    if guidance:
                        if brief_reason:
                            return f"{guidance} REASON: {brief_reason}"
                        return guidance
                    return None

                manager_guidance = None
                if manager_type == "EMBODIED":
                    manager_guidance = build_enhanced_guidance(parsed_gpt)
                elif manager_type == "CONTEXT":
                    working_memory = apply_memory_operation(working_memory, parsed_gpt)
                elif manager_type == "HYBRID":
                    manager_guidance = build_enhanced_guidance(parsed_gpt)
                    working_memory = apply_memory_operation(working_memory, parsed_gpt)

                # Step 3: Call Executor with soft tokens (ScienceWorld version)
                matched_action, match_score, raw_action, qwen_raw_response, qwen_input_fields = executor_generate_with_soft_tokens_sciworld(
                    executor_model=executor_model,
                    executor_tokenizer=executor_tokenizer,
                    sampled_latent_embeddings=sampled_latent_embeddings,
                    working_memory=working_memory,
                    task_text=task_text,
                    action_templates=action_templates,
                    available_objects=available_objects,
                    current_inventory=current_inventory,
                    valid_actions=valid_actions,
                    sent_model=sent_model,
                    history=history,
                    current_obs=current_obs,
                    gpt_guidance=manager_guidance,
                    num_thought_tokens=NUM_THOUGHT_TOKENS,
                    use_soft_tokens=USE_SOFT_TOKENS,
                    task_description=task_description,
                )
                print(f"[SW-Process {process_id}] Executor: raw='{raw_action}' -> matched='{matched_action}' (score={match_score:.3f})")

                if ENABLE_STEP_STATS:
                    _step_end_time = time.time()
                    episode_step_stats.append({
                        "task_id": task_id,
                        "step": step_count,
                        "step_time_sec": round(_step_end_time - _step_start_time, 4),
                        "input_tokens": qwen_input_fields.get("executor_input_token_count", 0),
                        "manager_pass1_input_tokens": manager_input_fields.get("manager_pass1_input_token_count", 0),
                        "manager_pass2_input_tokens": manager_input_fields.get("manager_pass2_input_token_count", 0),
                        "manager_output_tokens": manager_input_fields.get("manager_output_token_count", 0),
                        "executor_output_tokens": qwen_input_fields.get("executor_output_token_count", 0),
                        "manager_generate_sec": manager_input_fields.get("manager_generate_sec", 0.0),
                        "manager_soft_forward_sec": manager_input_fields.get("manager_soft_forward_sec", 0.0),
                        "executor_generate_sec": qwen_input_fields.get("executor_generate_sec", 0.0),
                    })

                # Step 4: Execute action in environment
                observation, reward, done = sw_env.step(matched_action)

                history.append({"action": matched_action, "observation": observation})

                # Record step
                step_record = {
                    "step": step_count,
                    "manager": {
                        "input": manager_input_fields,
                        "output": {"raw": manager_raw_response}
                    },
                    "executor": {
                        "input": qwen_input_fields,
                        "output": {
                            "raw": qwen_raw_response,
                            "parsed_action": raw_action,
                            "matched_action": matched_action,
                            "match_score": match_score,
                        }
                    },
                    "env": {
                        "action": matched_action,
                        "observation": observation,
                        "reward": reward,
                        "done": done,
                        "jvm_score": sw_env.get_score(),
                    }
                }
                detailed_steps.append(step_record)

                # Check if done
                final_jvm_score = sw_env.get_score()
                if done or final_jvm_score >= 100:
                    episode_done = True
                    episode_won = final_jvm_score >= 100
                    print(f"[SW-Process {process_id}] Episode finished! Won: {episode_won}, "
                          f"JVM Score: {final_jvm_score}, Steps: {len(history)}")

            # Get final metrics
            final_jvm_score = sw_env.get_score()
            progress_rate, _, _ = sw_env.feedback()

            stats['total'] += 1
            if episode_won:
                stats['success'] += 1
            else:
                stats['failed'] += 1
            stats['total_jvm_score'] += final_jvm_score
            stats['total_progress'] += progress_rate

            if ENABLE_STEP_STATS:
                all_step_stats.extend(episode_step_stats)

            # Save trajectory
            traj_file = os.path.join(traj_path, f"{task_id}.json")
            out_traj = {
                "task_id": task_id,
                "task_name": task_name,
                "variation_idx": variation_idx,
                "task": task_text,
                "won": episode_won,
                "jvm_score": final_jvm_score,
                "progress_rate": progress_rate,
                "failure_reason": failure_reason,
                "steps": len(history),
                "trajectory": history
            }
            with open(traj_file, "w") as f:
                json.dump(out_traj, f, indent=4)

            # Save detailed trajectory
            detailed_file = os.path.join(detailed_traj_path, f"{task_id}.json")
            detailed_out = {
                "task_id": task_id,
                "task_name": task_name,
                "variation_idx": variation_idx,
                "task": task_text,
                "won": episode_won,
                "jvm_score": final_jvm_score,
                "progress_rate": progress_rate,
                "failure_reason": failure_reason,
                "task_memory_summary": f"{len(successful_tasks or [])} success, {len(failed_tasks or [])} fail, {len(insights or [])} insights",
                "steps": detailed_steps
            }
            with open(detailed_file, "w") as f:
                json.dump(detailed_out, f, indent=4)

            # Append to overview
            out_overview = {
                "task_id": task_id,
                "task_name": task_name,
                "variation_idx": variation_idx,
                "task": task_text,
                "won": episode_won,
                "jvm_score": final_jvm_score,
                "progress_rate": progress_rate,
                "failure_reason": failure_reason,
                "steps": len(history),
            }
            with FileLock(overview_file + ".lock"):
                with open(overview_file, "a") as f:
                    f.write(json.dumps(out_overview) + "\n")

            # Add to Memcompiler if UPDATE_MEMORY is enabled
            if UPDATE_MEMORY:
                try:
                    mas_message = MASMessage(
                        task_main=task_text,
                        task_description=task_description,
                        label=episode_won,
                    )
                    mas_message.add_extra_field("task_name", task_name)
                    mas_message.add_extra_field("variation_idx", variation_idx)
                    mas_message.add_extra_field("jvm_score", final_jvm_score)
                    mas_message.add_extra_field("progress_rate", progress_rate)
                    for h in history:
                        mas_message.move_state(h["action"], h["observation"])
                    memory.add_memory(mas_message)
                    print(f"[SW-Process {process_id}] Added task to Memcompiler. Size: {memory.memory_size}")
                except Exception as e:
                    print(f"[SW-Process {process_id}] Failed to add memory: {e}")

        except Exception as e:
            print(f"[SW-Process {process_id}] Error on task {task_id}: {e}")
            traceback.print_exc()
            continue

    # Cleanup
    try:
        sw_env.close()
    except:
        pass

    # Print final stats
    print(f"\n[SW-Process {process_id}] FINISHED")
    print(f"Total: {stats['total']}, Success: {stats['success']}, Failed: {stats['failed']}")
    if stats['total'] > 0:
        print(f"Accuracy: {stats['success'] / stats['total'] * 100:.2f}%")
        print(f"Avg JVM Score: {stats['total_jvm_score'] / stats['total']:.2f}")
        print(f"Avg Progress: {stats['total_progress'] / stats['total']:.2f}")

    # Write step stats file
    if ENABLE_STEP_STATS and all_step_stats:
        step_times = [s["step_time_sec"] for s in all_step_stats]
        n = len(step_times)
        sorted_times = sorted(step_times)
        median_time = sorted_times[n // 2] if n % 2 == 1 else (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        summary = {
            "total_steps": n,
            "avg_step_time_sec": round(sum(step_times) / n, 4),
            "median_step_time_sec": round(median_time, 4),
            "max_step_time_sec": round(max(step_times), 4),
            "min_step_time_sec": round(min(step_times), 4),
        }
        stats_output = {"summary": summary, "per_step": all_step_stats}
        stats_file = os.path.join(SCIWORLD_OUTPUT_PATH, f"step_stats_process_{process_id}.json")
        with open(stats_file, "w") as f:
            json.dump(stats_output, f, indent=2)
        print(f"[SW-Process {process_id}] Step stats saved to: {stats_file}")


# =============================================================================
# Main Function
# =============================================================================
def main():
    print("=" * 60)
    print(f"Eval - EVAL_TASK={EVAL_TASK}")
    print("=" * 60)
    print(f"NUM_PROCESSES: {NUM_PROCESSES}")
    print(f"NUM_GPUS: {NUM_GPUS}")

    if EVAL_TASK == "sciworld":
        # =====================================================================
        # ScienceWorld Evaluation Branch
        # =====================================================================
        print(f"SCIWORLD_OUTPUT_PATH: {SCIWORLD_OUTPUT_PATH}")
        print(f"SCIWORLD_MEMCOMPILER_PATH: {SCIWORLD_MEMCOMPILER_PATH}")
        print(f"SCIWORLD_TEST_JSONL_PATH: {SCIWORLD_TEST_JSONL_PATH}")

        os.makedirs(SCIWORLD_OUTPUT_PATH, exist_ok=True)
        os.makedirs(os.path.join(SCIWORLD_OUTPUT_PATH, "trajectory"), exist_ok=True)
        os.makedirs(os.path.join(SCIWORLD_OUTPUT_PATH, "detailed_trajectory"), exist_ok=True)

        # Load task configs from JSONL
        print(f"\nLoading ScienceWorld tasks from: {SCIWORLD_TEST_JSONL_PATH}")
        task_configs = load_scienceworld_tasks_from_jsonl(SCIWORLD_TEST_JSONL_PATH)
        total_tasks = len(task_configs)
        print(f"Total tasks loaded: {total_tasks}")

        # Filter out completed tasks
        overview_path = os.path.join(SCIWORLD_OUTPUT_PATH, "overview.jsonl")
        completed_ids = set()
        if os.path.exists(overview_path):
            with open(overview_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        task_id = record.get('task_id')
                        if task_id is not None:
                            completed_ids.add(task_id)
                    except:
                        continue
            print(f"Already completed: {len(completed_ids)}")

        all_indices = list(range(total_tasks))
        uncompleted_indices = [idx for idx in all_indices if task_configs[idx].get("id", idx) not in completed_ids]
        print(f"Uncompleted tasks to run: {len(uncompleted_indices)}")

        if len(uncompleted_indices) == 0:
            print("All tasks already completed! Exiting.")
            return

        # Static task allocation
        task_assignments = [[] for _ in range(NUM_PROCESSES)]
        for i, idx in enumerate(uncompleted_indices):
            task_assignments[i % NUM_PROCESSES].append(idx)

        for i, tasks in enumerate(task_assignments):
            print(f"Process {i}: {len(tasks)} tasks")

        # Start worker processes
        print("\n" + "=" * 60)
        print("Starting ScienceWorld worker processes...")
        print("=" * 60)

        processes = []
        for i in range(NUM_PROCESSES):
            p = mp.Process(
                target=worker_process_sciworld,
                args=(i, GPU_ASSIGNMENTS[i], task_configs, task_assignments[i])
            )
            p.start()
            processes.append(p)
            time.sleep(2)

        for p in processes:
            p.join()

        print("\n" + "=" * 60)
        print("ALL SCIWORLD PROCESSES COMPLETED")
        print("=" * 60)

        # Calculate final stats
        total_completed = 0
        total_success = 0
        total_jvm = 0.0
        total_progress = 0.0
        if os.path.exists(overview_path):
            with open(overview_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        total_completed += 1
                        if record.get('won'):
                            total_success += 1
                        total_jvm += record.get('jvm_score', 0.0)
                        total_progress += record.get('progress_rate', 0.0)
                    except:
                        pass

        print(f"Total completed: {total_completed}")
        print(f"Success: {total_success}")
        if total_completed > 0:
            print(f"Accuracy: {total_success / total_completed * 100:.2f}%")
            print(f"Avg JVM Score: {total_jvm / total_completed:.2f}")
            print(f"Avg Progress Rate: {total_progress / total_completed:.2f}")

    else:
        # =====================================================================
        # ALFWorld Evaluation Branch (original logic)
        # =====================================================================
        print(f"OUTPUT_PATH: {OUTPUT_PATH}")
        print(f"MEMCOMPILER_PATH: {MEMCOMPILER_PATH}")

        # Setup directories
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, "trajectory"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, "detailed_trajectory"), exist_ok=True)

        # Get game file list from environment
        print("\nInitializing environment to get game list...")

        import alfworld.agents.modules.generic as generic
        import alfworld.agents.environment.alfred_tw_env as alfred_tw_env

        generic.ALFWORLD_CONFIG = CONFIG_FILE
        config = generic.load_config()
        env_type = config['env']['type']

        # Directly instantiate the text environment
        thor_env = alfred_tw_env.AlfredTWEnv(config, train_eval='eval_out_of_distribution')

        game_file_list = None
        total_games = None
        if hasattr(thor_env, 'game_files'):
            game_file_list = thor_env.game_files
            total_games = len(game_file_list)
        elif hasattr(thor_env, 'num_games'):
            total_games = int(thor_env.num_games)
        if not total_games:
            total_games = 500

        print(f"Total games in eval_unseen: {total_games}")
        print(f"Environment type: {env_type}")

        # All indices for eval_unseen
        all_indices = list(range(total_games))
        print(f"Total tasks to run: {len(all_indices)}")

        # Filter out completed tasks
        overview_path = os.path.join(OUTPUT_PATH, "overview.jsonl")
        completed_indices = set()
        if os.path.exists(overview_path):
            print(f"Checking completed tasks from: {overview_path}")
            with open(overview_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        env_index = record.get('env_index')
                        if env_index is not None:
                            completed_indices.add(env_index)
                    except:
                        continue
            print(f"Already completed: {len(completed_indices)}")

        uncompleted_indices = [idx for idx in all_indices if idx not in completed_indices]
        print(f"Uncompleted tasks to run: {len(uncompleted_indices)}")

        if len(uncompleted_indices) == 0:
            print("All tasks already completed! Exiting.")
            return

        # Static task allocation
        task_assignments = [[] for _ in range(NUM_PROCESSES)]
        for i, idx in enumerate(uncompleted_indices):
            task_assignments[i % NUM_PROCESSES].append(idx)

        for i, tasks in enumerate(task_assignments):
            print(f"Process {i}: {len(tasks)} tasks")

        # Start worker processes
        print("\n" + "=" * 60)
        print("Starting worker processes...")
        print("=" * 60)

        processes = []
        for i in range(NUM_PROCESSES):
            p = mp.Process(
                target=worker_process,
                args=(i, GPU_ASSIGNMENTS[i], task_assignments[i], game_file_list)
            )
            p.start()
            processes.append(p)
            time.sleep(2)

        # Wait for all processes
        for p in processes:
            p.join()

        print("\n" + "=" * 60)
        print("ALL PROCESSES COMPLETED")
        print("=" * 60)

        # Calculate final stats
        total_completed = 0
        total_success = 0
        if os.path.exists(overview_path):
            with open(overview_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        total_completed += 1
                        if record.get('won'):
                            total_success += 1
                    except:
                        pass

        print(f"Total completed: {total_completed}")
        print(f"Success: {total_success}")
        if total_completed > 0:
            print(f"Accuracy: {total_success / total_completed * 100:.2f}%")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
