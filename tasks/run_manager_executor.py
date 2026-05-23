#!/usr/bin/env python
"""
AlfredTWEnv (text) script with GPT + Qwen executor collaboration - Multi-process version.
Modified version for 60% training data with Memcompiler Task Memory integration (read-only).

4 processes × 4 GPUs, static task pre-allocation.

Usage:
    python tasks/Thor_gpt_qwen_train60_multiprocess.py

Environment variables:
    OPENAI_API_KEY: API key for OpenAI (only needed when API_TYPE="openai")
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
from openai import OpenAI, AzureOpenAI
from typing import List, Optional, Dict, Any, Tuple, Union
from transformers import BitsAndBytesConfig
# =============================================================================
# Configuration
# =============================================================================
NUM_PROCESSES = 4
NUM_GPUS = 4

OUTPUT_PATH = "./output/alfworld"
MEMCOMPILER_PATH = "./output/memcompiler"
SAMPLED_INDICES_FILE = "data/alfworld/sampled_indices.json"
CONFIG_FILE = "tasks/env_configs/alfworld_config.yaml"
DATASET_SPLIT = "eval_out_of_distribution"  # remain | train | eval_out_of_distribution


# Executor model (low-level action) - Text-only Qwen
EXECUTOR_QWEN_MODEL_NAME = "YOUR_EXECUTOR_MODEL_PATH"

# Manager model configuration: "gpt" or "qwen"
MANAGER_MODEL = "qwen"
MANAGER_QWEN_MODEL_NAME = "YOUR_MANAGER_MODEL_PATH"  # only used when MANAGER_MODEL="qwen"
GPT_MODEL = "gpt-5"

# API type configuration: "openai" or "azure"
API_TYPE = "azure"

# Azure OpenAI configuration (used when API_TYPE = "azure")
# Note: azure_endpoint should be the base endpoint URL, SDK constructs full path
AZURE_OPENAI_CONFIG = {
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY", "YOUR_AZURE_API_KEY"),
    "endpoints": {
        "gpt-4o": {
            "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", "https://YOUR_ENDPOINT.openai.azure.com/"),
            "api_version": "2025-01-01-preview",
        },
    }
}

# API retry configuration (for rate limit errors)
API_MAX_RETRIES = 5
API_RETRY_DELAY = 5  # seconds

MAX_STEPS = 30

GPU_ASSIGNMENTS = [0,1,2,3]  # Adjust to your available GPUs

# Memory update configuration
UPDATE_MEMORY = True  # Set to True to enable memory updates via ChromaDB server
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001

DEFAULT_MEMCOMPILER_FALLBACK_PATHS = [
    "./output/memcompiler",
]


def get_memcompiler_embedding_count(memcompiler_working_dir: str) -> int:
    """Return number of stored Chroma embeddings under working_dir/namespace."""
    chroma_db_path = os.path.join(memcompiler_working_dir, "memcompiler", "chroma.sqlite3")
    if not os.path.exists(chroma_db_path):
        return 0

    import sqlite3

    try:
        conn = sqlite3.connect(chroma_db_path)
        try:
            cur = conn.cursor()
            row = cur.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except Exception as e:
        print(f"[Memcompiler] Failed to inspect Chroma DB at {chroma_db_path}: {e}")
        return 0


def resolve_memcompiler_path(configured_path: str, allow_fallback: bool = True) -> str:
    """Resolve to a non-empty Memcompiler working_dir when possible."""
    if not allow_fallback:
        print(
            "[Memcompiler] Fallback disabled (UPDATE_MEMORY=True). Using configured path: "
            f"{configured_path}"
        )
        return configured_path

    candidate_paths = [configured_path]
    for fallback_path in DEFAULT_MEMCOMPILER_FALLBACK_PATHS:
        if fallback_path not in candidate_paths:
            candidate_paths.append(fallback_path)

    print("[Memcompiler] Inspecting candidate memory paths...")
    best_path = configured_path
    best_count = -1
    for path in candidate_paths:
        count = get_memcompiler_embedding_count(path)
        exists = os.path.isdir(path)
        print(f"[Memcompiler] Candidate: {path} | exists={exists} | embeddings={count}")
        if count > best_count:
            best_count = count
            best_path = path

    configured_count = get_memcompiler_embedding_count(configured_path)
    if configured_count > 0:
        return configured_path

    if best_count > 0 and best_path != configured_path:
        print(
            f"[Memcompiler] WARNING: Configured MEMCOMPILER_PATH is empty. "
            f"Auto-switching to populated memory: {best_path}"
        )
        return best_path

    print(
        f"[Memcompiler] WARNING: No populated Memcompiler found. "
        f"Will continue with configured path: {configured_path}"
    )
    return configured_path


def resolve_train_eval(split: str) -> str:
    if split == "eval_out_of_distribution":
        return "eval_out_of_distribution"
    if split in {"remain", "train"}:
        return "train"
    raise ValueError(
        f"Unsupported DATASET_SPLIT '{split}'. Use remain | train | eval_out_of_distribution."
    )


def resolve_target_indices(total_games: int, split: str) -> List[int]:
    all_indices = list(range(total_games))

    if split == "remain":
        print(f"\nLoading sampled indices from: {SAMPLED_INDICES_FILE}")
        with open(SAMPLED_INDICES_FILE, "r") as f:
            sampled_data = json.load(f)
        sampled_40_indices = set(sampled_data["indices"])
        print(f"Sampled 40% indices count: {len(sampled_40_indices)}")
        target_indices = sorted([idx for idx in all_indices if idx not in sampled_40_indices])
        print(f"Target split: remain (train remaining 60%)")
        return target_indices

    if split == "train":
        print("Target split: train (all training tasks)")
        return all_indices

    if split == "eval_out_of_distribution":
        print("Target split: eval_out_of_distribution (valid_unseen tasks)")
        return all_indices

    raise ValueError(
        f"Unsupported DATASET_SPLIT '{split}'. Use remain | train | eval_out_of_distribution."
    )



# =============================================================================
# File Lock Utility
# =============================================================================
class FileLock:
    """Simple file-based lock using fcntl."""

    def __init__(self, lock_file: str):
        self.lock_file = lock_file
        self.lock_fd = None

    def acquire(self):
        self.lock_fd = open(self.lock_file, 'w')
        fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX)

    def release(self):
        if self.lock_fd:
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
            self.lock_fd.close()
            self.lock_fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


# =============================================================================
# SafeMemcompiler - Thread-safe Memcompiler with file locking for multi-process updates
# =============================================================================
def create_safe_memcompiler_class():
    """Create SafeMemcompiler class after imports are set up. Only used when UPDATE_MEMORY=True."""
    import pickle
    from mas.memory.mas_memory import Memcompiler
    from mas.memory.mas_memory.memcompiler import TaskLayer, InsightsManager
    from mas.utils import write_json

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
                # Step 1: Reload latest graph from pickle
                if os.path.exists(self._graph_save_path):
                    with open(self._graph_save_path, 'rb') as f:
                        self.graph = pickle.load(f)

                # Step 2: Check if already exists
                if task_main in self.graph:
                    return

                # Step 3: Add new node
                self.graph.add_node(task_main)

                # Step 4: Find similar tasks and add edges
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

                # Step 5: Save updated graph (still under lock)
                self._index_done()

        def retrieve_related_task(self, query_task: str, node_num: int, hop: int = 1) -> list:
            """Safe version that handles nodes not in graph (multi-process sync issues)."""
            import networkx as nx

            # Reload latest graph
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

            # Use ChromaDB HTTP client mode for multi-process concurrency
            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            self.main_memory = Chroma(
                client=chroma_client,
                collection_name="langchain",
                embedding_function=self.embedding_func,
            )

            self._hop = self.global_config.get('hop', 1)
            self._start_insights_threshold = self.global_config.get('start_insights_threshold', 5)
            self._rounds_per_insights = self.global_config.get('rounds_per_insights', 5)
            self._insights_point_num = self.global_config.get('insights_point_num', 5)

            # Use safe versions with file locking
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
                f"[SafeMemcompiler] Initialized with ChromaDB HTTP client at {CHROMA_HOST}:{CHROMA_PORT} | "
                f"persist_dir={self.persist_dir} | collection=langchain"
            )

        def add_memory(self, mas_message) -> None:
            """Add memory with locked insights update."""
            from langchain.docstore.document import Document
            from mas.memory.common import MASMessage

            # Sparsification
            mas_message = self._extract_mas_message(mas_message=mas_message)

            # Add into memory (TaskLayer uses its own lock)
            self.task_layer.add_task_node(mas_message.task_main)

            # Add to ChromaDB
            meta_data = MASMessage.to_dict(mas_message)
            memory_doc = Document(
                page_content=mas_message.task_main,
                metadata=meta_data
            )
            if mas_message.label == True or mas_message.label == False:
                self.main_memory.add_documents([memory_doc])
            else:
                raise ValueError('The mas_message must have label!')

            # Check if insights update is needed - use lock to ensure only one process updates
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


GPT_SYSTEM_PROMPT = """
You are the **High-Level Strategic Planner** for a household robot. You guide the Low-Level Executor (Qwen) using visual observation, memory, and action history. You do NOT execute actions directly.

## DECISION LOGIC
Evaluate inputs (Task, Memory, Observation, History) and select **EXACTLY ONE** response type based on these triggers:

1. **EMBODIED (Intervention):**
   - **Triggers:** Executor is stuck (loops), deviating from goal, or a sub-task is done (needs new instruction).
   - **Rules:**
     - Provide **single-step** strategic goals (No "do X then Y").
     - If looping, explicitly say "Do not [action] again".
     - If "move" fails at goal, suggest "put" (and vice versa).
     - If task item observed/closed, strictly guide to interact/open it.

2. **CONTEXT (Memory Ops):**
   - **Triggers:** New discovery (loc/state), state change (e.g., door opened), or memory cleanup needed.
   - **Rules:** `UPDATE`/`DELETE` must match existing text exactly. `FOLD` if >10 items.

3. **HYBRID:** Both Intervention and Memory Ops are needed simultaneously.

4. **NOACTION:** Executor is progressing logically; no new info to record and no need to guide.

## OUTPUT FORMAT SPECIFICATIONS
Use the **Component Definitions** below to fill the strictly required XML structure for your chosen type.

**A. [Guidance_Fields] Definition:**
<brief_reason>Why intervention is necessary</brief_reason>
<guidance>Strategic instruction (1 step only)</guidance>

**B. [Memory_Fields] Definition:**
<operation>CREATE/UPDATE/DELETE/FOLD</operation>
<!-- Add dependent tags based on operation: -->
<!-- IF UPDATE: --> <from>exact_old_substring</from><to>new_content</to>
<!-- IF DELETE: --> <content>exact_substring_to_delete</content>
<!-- IF CREATE: --> <content>new_fact</content>
<!-- IF FOLD:   --> <folded_content>summary</folded_content>

**Final XML block Structures (Choose ONE):**
Type 1: EMBODIED
<response_type>EMBODIED</response_type>
[Guidance_Fields]

Type 2: CONTEXT
<response_type>CONTEXT</response_type>
[Memory_Fields]

Type 3: HYBRID
<response_type>HYBRID</response_type>
<embodied>
  [Guidance_Fields]
</embodied>
<context>
  [Memory_Fields]
</context>

Type 4: NOACTION
<response_type>NOACTION</response_type>

## CORE CONSTRAINTS
1. Output Strictness: Only output the XML block. No markdown outside tags.
2. Memory: Read-only Task Memory; modify Working Memory only via CONTEXT ops.
3. Brevity: Keep reasons and guidance concise.
"""


# =============================================================================
# Qwen System Prompt
# =============================================================================
QWEN_SYSTEM_PROMPT = """
You are a household robot executor operating in a simulated home environment.
You are now in a household environment called Alfworld, and your tasks include locating objects, heating or cooling items, and other similar activities.

## OUTPUT FORMAT SPECIFICATIONS
- You must strictly follow the syntactic structure of the steps (where 'a' and 'b' are variables):
    - 1. take a from b.
    - 2. go to a.
    - 3. open a.
    - 4. move a to b.
    - 5. clean a with b.
    - 6. heat a with b.
    - 7. cool a with b.
    - 8. use a.

- You must check carefully whether your output command is consistent with the allowed commands above!!! Any output that is not among the commands listed above is not permitted!!!
"""

# =============================================================================
# Utility Functions
# =============================================================================
def extract_task(text: str) -> str:
    """Extract task description from observation text."""
    if not text:
        return ""
    m = re.search(r"Your task is to:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        task = m.group(1).strip()
        task = task.split("\n")[0].strip()
        return task
    return ""


def parse_action(model_text: str, admissible_commands: Optional[List[str]] = None) -> Optional[str]:
    """Extract action from <action>...</action> tags in model output.

    First tries to extract from <action> tags. If the extracted action is in
    admissible_commands, returns it directly. Otherwise tries to match against
    admissible_commands. Falls back to first line only if no admissible_commands given.
    """
    if not model_text:
        return None

    # Primary: extract from <action>...</action>
    m = re.search(r"<action>\s*(.*?)\s*</action>", model_text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        action = m.group(1).strip()
        # If we have admissible commands, verify this action is valid
        if admissible_commands:
            # Exact match first
            if action in admissible_commands:
                return action
            # Case-insensitive match
            action_lower = action.lower()
            for cmd in admissible_commands:
                if cmd.lower() == action_lower:
                    return cmd
            # Partial match: model may have slightly paraphrased, pick best candidate
            for cmd in admissible_commands:
                if action_lower in cmd.lower() or cmd.lower() in action_lower:
                    return cmd
            # Still return parsed action (env will handle invalid action gracefully)
            return action
        return action

    # Fallback: if admissible_commands provided, try to find any command mentioned in output
    if admissible_commands:
        text_lower = model_text.lower()
        for cmd in admissible_commands:
            if cmd.lower() in text_lower:
                return cmd
        # Last resort: return first admissible command (avoid crashing)
        return admissible_commands[0] if admissible_commands else None

    # No admissible_commands: take first non-empty line as last resort
    for line in model_text.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return None




def generate_runtime_key_steps(success_task: Any, llm_model: Any) -> Optional[str]:
    """Generate key steps for a retrieved successful task at runtime.

    Reuses Memcompiler's original extraction prompt so the output format stays
    aligned with the main Memcompiler+MAS workflow.
    """
    if success_task is None or llm_model is None:
        return None

    task_description = getattr(success_task, "task_description", None)
    task_main = getattr(success_task, "task_main", None)
    task_trajectory = getattr(success_task, "task_trajectory", None)

    if isinstance(success_task, dict):
        task_description = success_task.get("task_description") or success_task.get("task") or task_description
        task_main = success_task.get("task_main") or task_main
        task_trajectory = success_task.get("task_trajectory") or success_task.get("trajectory") or task_trajectory

    task_text = task_description or task_main
    if not task_text or not task_trajectory:
        return None

    from mas.llm import Message
    from mas.memory.mas_memory.prompt import MemcompilerPrompts

    clean_traj = re.sub(r'\d+', '', str(task_trajectory))
    prompt = MemcompilerPrompts.extract_true_traj_user_prompt.format(
        task=task_text,
        trajectory=clean_traj,
    )
    messages = [
        Message('system', MemcompilerPrompts.extract_true_traj_system_prompt),
        Message('user', prompt),
    ]

    try:
        response = llm_model(messages, temperature=0.1)
        if isinstance(response, str):
            response = response.strip()
        return response or None
    except Exception as e:
        print(f"[WARN] Failed to generate runtime key steps: {e}")
        return None

# =============================================================================
# Task Memory Formatting
# =============================================================================
def format_task_memory(task_memory: Optional[Dict[str, Any]]) -> str:
    """Format Task Memory for manager prompt."""
    if not task_memory:
        print(f"[DEBUG format_task_memory] ❌❌❌task_memory is empty/None, returning early")
        return "(No task memory available)"

    prompt_parts = []
    insights = task_memory.get("insights", []) if isinstance(task_memory, dict) else []

    if isinstance(task_memory, dict) and "successful_tasks" in task_memory:
        succ_tasks = task_memory.get("successful_tasks") or []
        runtime_key_steps = task_memory.get("runtime_key_steps") or []

        if succ_tasks:
            prompt_parts.append("\n### Successful Similar Tasks:")
            prompt_parts.append("\nHere are examples of successful execution processes you've previously used on similar tasks.")
            prompt_parts.append("\nPay special attention to the step-by-step procedures and strategies, especially when encountering obstacles:")
            for i, t in enumerate(list(succ_tasks), 1):
                task_description = getattr(t, "task_description", None) if t is not None else None
                task_main = getattr(t, "task_main", None) if t is not None else None
                task_traj = getattr(t, "task_trajectory", None) if t is not None else None
                if isinstance(t, dict):
                    task_description = t.get("task_description") or t.get("task") or task_description
                    task_main = t.get("task_main") or task_main
                    task_traj = t.get("task_trajectory") or t.get("trajectory") or task_traj

                display_task = task_description or task_main
                key_steps = runtime_key_steps[i - 1] if i - 1 < len(runtime_key_steps) else None

                prompt_parts.append(f"\nTask {i}:")
                prompt_parts.append("")
                prompt_parts.append("### Task description:   ")
                prompt_parts.append(str(display_task) if display_task else "(No task description available)")
                prompt_parts.append("")
                prompt_parts.append("### Key steps:")
                prompt_parts.append(str(key_steps) if key_steps else "(Key steps unavailable)")
                prompt_parts.append("")
                prompt_parts.append("### Detailed trajectory:")
                prompt_parts.append(str(task_traj) if task_traj else "(Trajectory unavailable)")

    if insights:
        prompt_parts.append("\n### Key Insights from Related Tasks:")
        prompt_parts.append("The following are insights gathered during the execution of similar tasks. You may refer to them to improve the accuracy of your decision.")
        for i, insight in enumerate(list(insights)[:10], 1):
            prompt_parts.append(f"{i}. {insight}")

    if not prompt_parts:
        return "(No relevant task memory found)"
    # print("=====================")
    # print(f"prompt_parts: {prompt_parts}")
    # print("=====================")

    return "\n".join(prompt_parts)


# =============================================================================
# GPT Module
# =============================================================================
def create_gpt_user_content(
    working_memory: str,
    task_text: str,
    history: Optional[List[Dict[str, Any]]],
    current_obs: str,
    admissible_commands: Optional[List[str]],
    task_memory: Optional[Dict[str, Any]] = None,
    initial_observation: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build GPT user message content with Task Memory and Working Memory (text-only).

    Returns:
        Tuple of (content, input_fields) where input_fields contains the intermediate variables.
    """
    prompt_parts = []

    # prompt_parts.append("## 🎯 CURRENT MISSION")
    prompt_parts.append("## 🎯 TASK DESCRIPTION")
    if initial_observation and initial_observation.strip():
        prompt_parts.append(initial_observation)
    prompt_parts.append("")
    prompt_parts.append(f"The ultimate goal is: **{task_text}**")

    # prompt_parts.append("\n## 👁️ OBSERVATION after the execution of the previous action")
    prompt_parts.append("\n## 👁️ CURRENT STATE")
    prompt_parts.append(f"{current_obs}")

    prompt_parts.append("\n## 📚 TASK MEMORY")
    formated_task_memory = format_task_memory(task_memory)
    prompt_parts.append(f"{formated_task_memory}")

    prompt_parts.append("\n## 🧠 WORKING MEMORY")
    if working_memory and working_memory.strip():
        working_memory_processed = working_memory
    else:
        working_memory_processed = "(Memory is currently empty. If you find important objects, record them.)"
    prompt_parts.append(working_memory_processed)

    prompt_parts.append("\n## 👣 RECENT TRAJECTORY (Last 15 Steps)")
    prompt_parts.append("\nPay attention to the item states contained in the observation corresponding to each executed action")

    action_history_parts = []
    if history:
        for i, item in enumerate(history[-15:], 1):
            action = item.get('action', 'Unknown')
            observation = item.get('observation', '')
            feedback = observation[:150] + "..." if len(observation) > 150 else observation
            # action_history_parts.append(f"\nStep -{16-i}: Action=[{action}]")
            action_history_parts.append(f"\nStep {i}: Action=[{action}]")
            if feedback:
                action_history_parts.append(f"c=[{feedback}]")
    else:
        action_history_parts.append("(No actions taken yet. This is the start of the episode.)")
    action_history = "\n".join(action_history_parts)
    prompt_parts.append(action_history)

    prompt_parts.append("\n## ⚡ DECISION PROTOCOL")
    prompt_parts.append("Analyze the input above explicitly:")
    prompt_parts.append("1. **Check Strategy:** Is the Executor stuck, looping, or deviating? (-> EMBODIED)")
    prompt_parts.append("2. **Check Memory:** Is there new info (locations/states) that contradicts or adds to Working Memory? (-> CONTEXT)")
    prompt_parts.append("3. **Check Progress:** If moving smoothly towards the goal with no new info? (-> NOACTION)")
    prompt_parts.append("\nOutput your decision strictly in the XML format defined in the system prompt.")

    text_content = "\n".join(prompt_parts)

    # Text-only content
    content = [{"type": "text", "text": text_content}]

    # Collect input fields for logging
    input_fields = {
        "task_text": task_text,
        "current_obs": current_obs,
        "task_memory": formated_task_memory,
        "working_memory": working_memory_processed,
        "action_history": action_history
    }

    return content, input_fields


def call_gpt(
    client: Union[OpenAI, AzureOpenAI],
    model: str,
    working_memory: str,
    task_text: str,
    history: Optional[List[Dict[str, Any]]],
    current_obs: str,
    admissible_commands: Optional[List[str]],
    task_memory: Optional[Dict[str, Any]] = None,
    initial_observation: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    """Call GPT API and return raw response along with input fields and token usage (text-only).

    Returns:
        Tuple of (raw_response, input_fields, token_usage)
        token_usage: {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    """
    user_content, input_fields = create_gpt_user_content(
        working_memory=working_memory,
        task_text=task_text,
        history=history,
        current_obs=current_obs,
        admissible_commands=admissible_commands,
        task_memory=task_memory,
        initial_observation=initial_observation,
    )

    messages = [
        {"role": "system", "content": GPT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    for attempt in range(API_MAX_RETRIES):
        try:
            # gpt-4.1
            # response = client.chat.completions.create(
            #     model=model,
            #     messages=messages,
            #     max_tokens=4096,
            #     temperature=0.0
            # )

            # gpt-5/gpt-5-mini: use larger max_completion_tokens for reasoning models
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=4096,  # Increased for GPT-5 reasoning models
                timeout=120,
                reasoning_effort="low"
            )
            # print("GPT success✅✅✅")
            print(f"👉👉👉response: {response}")
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Handle None or empty content
            if content is None or content == "":
                # Check if it's due to token limit (finish_reason='length')
                if finish_reason == "length":
                    print(f"⚠️ GPT-5 content truncated (finish_reason=length), retrying (attempt {attempt + 1}/{API_MAX_RETRIES})...")
                    time.sleep(API_RETRY_DELAY)
                    continue
                else:
                    print(f"⚠️ GPT-5 returned empty content (finish_reason={finish_reason})")
                    content = ""

            print(f"👉👉👉content extracted: {content[:200] if content else 'EMPTY'}...")

            # Extract token usage
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            return content, input_fields, token_usage
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RateLimitReached" in error_str:
                print(f"  Rate limit hit (attempt {attempt + 1}/{API_MAX_RETRIES}), waiting {API_RETRY_DELAY}s...")
                time.sleep(API_RETRY_DELAY)
                continue
            print(f"  GPT API error: {e}")
            return "<response_type>NOACTION</response_type>", input_fields, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    print(f"  GPT API failed after {API_MAX_RETRIES} retries")
    return "<response_type>NOACTION</response_type>", input_fields, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def create_manager_qwen_user_content(
    working_memory: str,
    task_text: str,
    history: Optional[List[Dict[str, Any]]],
    current_obs: str,
    admissible_commands: Optional[List[str]],
    task_memory: Optional[Dict[str, Any]] = None,
    initial_observation: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build Manager Qwen user message content with Task Memory and Working Memory (text-only).

    Returns:
        Tuple of (text_content, input_fields) where input_fields contains the intermediate variables.
    """
    prompt_parts = []

    # prompt_parts.append("## 🎯 CURRENT MISSION")
    prompt_parts.append("## 🎯 TASK DESCRIPTION")
    if initial_observation and initial_observation.strip():
        prompt_parts.append(initial_observation)
    prompt_parts.append("")
    prompt_parts.append(f"The ultimate goal is: **{task_text}**")

    prompt_parts.append("\n## 👁️ CURRENT STATE")
    prompt_parts.append(f"{current_obs}")

    prompt_parts.append("\n## 📚 TASK MEMORY")
    formated_task_memory = format_task_memory(task_memory)
    prompt_parts.append(f"{formated_task_memory}")

    prompt_parts.append("\n## 🧠 WORKING MEMORY")
    if working_memory and working_memory.strip():
        working_memory_processed = working_memory
    else:
        working_memory_processed = "(Memory is currently empty. If you find important objects, record them.)"
    prompt_parts.append(working_memory_processed)

    prompt_parts.append("\n## 👣 RECENT TRAJECTORY (Last 15 Steps)")
    prompt_parts.append("\nPay attention to the item states contained in the observation corresponding to each executed action")

    action_history_parts = []
    if history:
        for i, item in enumerate(history[-15:], 1):
            action = item.get('action', 'Unknown')
            observation = item.get('observation', '')
            feedback = observation[:150] + "..." if len(observation) > 150 else observation
            action_history_parts.append(f"\nStep {i}: Action=[{action}]")
            if feedback:
                action_history_parts.append(f"\n> Observation=[{feedback}]")
    else:
        action_history_parts.append("(No actions taken yet. This is the start of the episode.)")
    action_history = "\n".join(action_history_parts)
    prompt_parts.append(action_history)

    prompt_parts.append("\n## ⚡ DECISION PROTOCOL")
    prompt_parts.append("Analyze the input above explicitly:")
    prompt_parts.append("1. **Check Strategy:** Is the Executor stuck, looping, or deviating? (-> EMBODIED)")
    prompt_parts.append("2. **Check Memory:** Is there new info (locations/states) that contradicts or adds to Working Memory? (-> CONTEXT)")
    prompt_parts.append("3. **Check Progress:** If moving smoothly towards the goal with no new info? (-> NOACTION)")
    prompt_parts.append("\nOutput your decision strictly in the XML format defined in the system prompt.")

    text_content = "\n".join(prompt_parts)

    # Collect input fields for logging
    input_fields = {
        "task_text": task_text,
        "current_obs": current_obs,
        "task_memory": formated_task_memory,
        "working_memory": working_memory_processed,
        "action_history": action_history
    }

    return text_content, input_fields


@torch.no_grad()
def call_manager_qwen(
    model,
    tokenizer,
    working_memory: str,
    task_text: str,
    history: Optional[List[Dict[str, Any]]],
    current_obs: str,
    admissible_commands: Optional[List[str]],
    task_memory: Optional[Dict[str, Any]] = None,
    initial_observation: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Call local Qwen model as Manager and return raw response along with input fields (text-only).

    Returns:
        Tuple of (raw_response, input_fields)
    """
    text_content, input_fields = create_manager_qwen_user_content(
        working_memory=working_memory,
        task_text=task_text,
        history=history,
        current_obs=current_obs,
        admissible_commands=admissible_commands,
        task_memory=task_memory,
        initial_observation=initial_observation,
    )

    messages = [
        {"role": "system", "content": GPT_SYSTEM_PROMPT},
        {"role": "user", "content": text_content}
    ]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        return response, input_fields
    except Exception as e:
        print(f"  Manager Qwen error: {e}")
        return "<response_type>NOACTION</response_type>", input_fields


def parse_gpt_response(response_text: str) -> Dict[str, Any]:
    """Parse GPT output and return structured result."""
    result = {"raw": response_text, "type": "NOACTION"}

    if not response_text:
        return result

    type_match = re.search(r"<response_type>\s*(.*?)\s*</response_type>", response_text, re.DOTALL | re.IGNORECASE)
    if type_match:
        result["type"] = type_match.group(1).strip().upper()
    else:
        print("[DEBUG] 无有效明文部分❌")

    if result["type"] == "EMBODIED":
        reason_match = re.search(r"<brief_reason>\s*(.*?)\s*</brief_reason>", response_text, re.DOTALL | re.IGNORECASE)
        if reason_match:
            result["brief_reason"] = reason_match.group(1).strip()
        guide_match = re.search(r"<guidance>\s*(.*?)\s*</guidance>", response_text, re.DOTALL | re.IGNORECASE)
        if guide_match:
            result["guidance"] = guide_match.group(1).strip()

    elif result["type"] == "CONTEXT":
        op_match = re.search(r"<operation>\s*(.*?)\s*</operation>", response_text, re.DOTALL | re.IGNORECASE)
        if op_match:
            result["operation"] = op_match.group(1).strip().upper()
        content_match = re.search(r"<content>\s*(.*?)\s*</content>", response_text, re.DOTALL | re.IGNORECASE)
        if content_match:
            result["content"] = content_match.group(1).strip()
        fold_match = re.search(r"<folded_content>\s*(.*?)\s*</folded_content>", response_text, re.DOTALL | re.IGNORECASE)
        if fold_match:
            result["folded_content"] = fold_match.group(1).strip()
        from_match = re.search(r"<from>\s*(.*?)\s*</from>", response_text, re.DOTALL | re.IGNORECASE)
        if from_match:
            result["from"] = from_match.group(1).strip()
        to_match = re.search(r"<to>\s*(.*?)\s*</to>", response_text, re.DOTALL | re.IGNORECASE)
        if to_match:
            result["to"] = to_match.group(1).strip()

    elif result["type"] == "HYBRID":
        # Parse nested <embodied> block
        embodied_match = re.search(r"<embodied>\s*(.*?)\s*</embodied>", response_text, re.DOTALL | re.IGNORECASE)
        if embodied_match:
            embodied_content = embodied_match.group(1)
            reason_match = re.search(r"<brief_reason>\s*(.*?)\s*</brief_reason>", embodied_content, re.DOTALL | re.IGNORECASE)
            if reason_match:
                result["brief_reason"] = reason_match.group(1).strip()
            guide_match = re.search(r"<guidance>\s*(.*?)\s*</guidance>", embodied_content, re.DOTALL | re.IGNORECASE)
            if guide_match:
                result["guidance"] = guide_match.group(1).strip()

        # Parse nested <context> block
        context_match = re.search(r"<context>\s*(.*?)\s*</context>", response_text, re.DOTALL | re.IGNORECASE)
        if context_match:
            context_content = context_match.group(1)
            op_match = re.search(r"<operation>\s*(.*?)\s*</operation>", context_content, re.DOTALL | re.IGNORECASE)
            if op_match:
                result["operation"] = op_match.group(1).strip().upper()
            content_match = re.search(r"<content>\s*(.*?)\s*</content>", context_content, re.DOTALL | re.IGNORECASE)
            if content_match:
                result["content"] = content_match.group(1).strip()
            fold_match = re.search(r"<folded_content>\s*(.*?)\s*</folded_content>", context_content, re.DOTALL | re.IGNORECASE)
            if fold_match:
                result["folded_content"] = fold_match.group(1).strip()
            from_match = re.search(r"<from>\s*(.*?)\s*</from>", context_content, re.DOTALL | re.IGNORECASE)
            if from_match:
                result["from"] = from_match.group(1).strip()
            to_match = re.search(r"<to>\s*(.*?)\s*</to>", context_content, re.DOTALL | re.IGNORECASE)
            if to_match:
                result["to"] = to_match.group(1).strip()

    return result


# =============================================================================
# Working Memory Module
# =============================================================================
def apply_memory_operation(working_memory: str, parsed_gpt: Dict[str, Any]) -> str:
    """Modify Working Memory based on GPT's CONTEXT or HYBRID output."""
    response_type = parsed_gpt.get("type", "")
    if response_type not in ("CONTEXT", "HYBRID"):
        return working_memory

    operation = parsed_gpt.get("operation", "")

    if operation == "CREATE":
        content = parsed_gpt.get("content", "")
        if content:
            if working_memory:
                return working_memory + "\n" + content
            return content

    elif operation == "FOLD":
        folded = parsed_gpt.get("folded_content", "")
        if folded:
            return folded

    elif operation == "UPDATE":
        from_text = parsed_gpt.get("from", "")
        to_text = parsed_gpt.get("to", "")
        if from_text and from_text in working_memory:
            return working_memory.replace(from_text, to_text)

    elif operation == "DELETE":
        content = parsed_gpt.get("content", "")
        if content and content in working_memory:
            new_memory = working_memory.replace(content, "")
            new_memory = re.sub(r'\n{3,}', '\n\n', new_memory)
            return new_memory.strip()

    return working_memory


# =============================================================================
# Qwen Module
# =============================================================================
def create_qwen_user_content(
    working_memory: str,
    task_text: str,
    admissible_commands: Optional[List[str]],
    history: Optional[List[Dict[str, Any]]],
    current_obs: str,
    gpt_guidance: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """Build Qwen user message content (text-only).

    Returns:
        Tuple of (text_content, input_fields) where input_fields contains the intermediate variables.
    """
    prompt_parts = []

    prompt_parts.append("## Working Memory")
    if working_memory:
        working_memory_processed = working_memory
    else:
        working_memory_processed = "(Empty for the time being)"
    prompt_parts.append(working_memory_processed)

    prompt_parts.append("\n## ATTENTION: Extremely useful guidance for your next action")
    if gpt_guidance:
        gpt_guidance_processed = gpt_guidance
    else:
        gpt_guidance_processed = "(Empty for the time being)"
    prompt_parts.append(gpt_guidance_processed)

    prompt_parts.append("\n## Task Description")
    prompt_parts.append(f"Your task is to: {task_text}")

    prompt_parts.append("\n## Executed Action History")
    action_history_parts = []
    if history:
        for i, item in enumerate(history[-10:], 1):
            action = item.get('action', '')
            observation = item.get('observation', '')
            action_history_parts.append(f"Step {i}: {action}")
            action_history_parts.append(f"   Obs: {observation[:150]}..." if len(observation) > 150 else f"  Feedback: {observation}")
    else:
        action_history_parts.append("(Empty for the time being)")
    action_history = "\n".join(action_history_parts)
    prompt_parts.append(action_history)

    prompt_parts.append("\n## Current State")
    prompt_parts.append(current_obs)

    prompt_parts.append("\n## Current Executable Commands")
    admissible_commands_parts = []
    if admissible_commands:
        unique_cmds = list(dict.fromkeys(admissible_commands))[:120]
        for cmd in unique_cmds:
            admissible_commands_parts.append(f"- {cmd}")
    else:
        admissible_commands_parts.append("- (no specific commands)")
    admissible_commands_str = "\n".join(admissible_commands_parts)
    prompt_parts.append(admissible_commands_str)


    prompt_parts.append("\n## Output Requirements (Strictly Follow!)")
    prompt_parts.append("Please select one from the executable commands above and output exactly.")
    # prompt_parts.append("<action>your selected command</action>")
    # prompt_parts.append("\nNote: Only output the <action> tag, do not output any other content!")
    prompt_parts.append("\n## Your Turn: Take Action!")
    prompt_parts.append("Use the above guidance and insights as a foundation, and now work on the following task:")
    prompt_parts.append(f"Your task is to: {task_text}")
    # prompt_parts.append("\nPlease select one from the executable commands above.")

    text_content = "\n".join(prompt_parts)

    # Collect input fields for logging
    input_fields = {
        "working_memory": working_memory_processed,
        "gpt_guidance": gpt_guidance_processed,
        "task_text": task_text,
        "action_history": action_history,
        "current_obs": current_obs,
        "admissible_commands": admissible_commands_str
    }

    return text_content, input_fields


@torch.no_grad()
def qwen_choose_action(
    model,
    tokenizer,
    working_memory: str,
    task_text: str,
    admissible: Optional[List[str]],
    history: Optional[List[Dict[str, Any]]],
    current_obs: str,
    gpt_guidance: Optional[str] = None,
    max_history_turns: int = 12
) -> Tuple[str, str, Dict[str, Any]]:
    """Use Qwen to generate next action (text-only).

    Returns:
        Tuple of (action, raw_response, input_fields)
    """
    if history and len(history) > max_history_turns:
        history = history[-max_history_turns:]

    text_content, input_fields = create_qwen_user_content(
        working_memory=working_memory,
        task_text=task_text,
        admissible_commands=admissible,
        history=history,
        current_obs=current_obs,
        gpt_guidance=gpt_guidance
    )

    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": text_content},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(f"👉👉👉admissible commands: {admissible}")
    print(f"👉👉👉qwen response: {response}")

    action = parse_action(response, admissible_commands=admissible)
    print(f"👉👉👉qwen action: {action}")
    action = action.strip() if action else (admissible[0] if admissible else "look")
    return action, response, input_fields


# =============================================================================
# GPT LLM Wrapper for Memcompiler
# =============================================================================
class GPTLLMWrapper:
    """Wrapper to make GPT API compatible with Memcompiler's LLMCallable interface."""

    def __init__(self, client: Union[OpenAI, AzureOpenAI], model: str):
        self.client = client
        self.model = model
        # Token usage tracking
        self.accumulated_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def reset_token_counter(self):
        """Reset accumulated token counter (call at start of each task)."""
        self.accumulated_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_accumulated_tokens(self) -> Dict[str, int]:
        """Get accumulated token usage."""
        return self.accumulated_tokens.copy()

    def __call__(self, messages, temperature: float = 0.0, max_tokens: int = 512, **kwargs) -> str:
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        for attempt in range(API_MAX_RETRIES):
            try:
                # gpt-4.1
                # response = self.client.chat.completions.create(
                #     model=self.model,
                #     messages=openai_messages,
                #     max_tokens=4096,
                #     temperature=temperature,
                #     timeout=120,
                # )

                # gpt-5/gpt-5-mini: use larger max_completion_tokens for reasoning models
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    max_completion_tokens=4096,  # Increased for GPT-5 reasoning models
                    timeout=120,
                    reasoning_effort="low"
                )
                # print("GPT success✅✅✅")
                print(f"👉👉👉response in Memcompiler: {response}")

                # Accumulate token usage
                if response.usage:
                    self.accumulated_tokens["prompt_tokens"] += response.usage.prompt_tokens
                    self.accumulated_tokens["completion_tokens"] += response.usage.completion_tokens
                    self.accumulated_tokens["total_tokens"] += response.usage.total_tokens

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                # Handle None or empty content
                if content is None or content == "":
                    if finish_reason == "length":
                        print(f"⚠️ Memcompiler GPT-5 content truncated (finish_reason=length), retrying (attempt {attempt + 1}/{API_MAX_RETRIES})...")
                        time.sleep(API_RETRY_DELAY)
                        continue
                    else:
                        print(f"⚠️ Memcompiler GPT-5 returned empty content (finish_reason={finish_reason})")
                        return ""

                return content
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RateLimitReached" in error_str:
                    print(f"  Memcompiler rate limit hit (attempt {attempt + 1}/{API_MAX_RETRIES}), waiting {API_RETRY_DELAY}s...")
                    time.sleep(API_RETRY_DELAY)
                    continue
                print(f"GPT API error in Memcompiler: {e}")
                return ""

        print(f"  Memcompiler GPT API failed after {API_MAX_RETRIES} retries")
        return ""


class QwenLLMWrapper:
    """Wrapper to make local Qwen model compatible with Memcompiler's LLMCallable interface."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def __call__(self, messages, temperature: float = 0.0, max_tokens: int = 512, **kwargs) -> str:
        try:
            # Convert Message objects to dict format
            qwen_messages = [{"role": m.role, "content": m.content} for m in messages]

            # Use tokenizer's chat template for text-only model
            text = self.tokenizer.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=(temperature > 0))

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

            return response
        except Exception as e:
            print(f"Qwen LLM error in Memcompiler: {e}")
            return ""


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
    print(f"[Process {process_id}] DATASET_SPLIT: {DATASET_SPLIT}")

    # Add project root to sys.path to ensure 'mas' module can be imported
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Set GPU visibility
    use_multi_gpu_manager = NUM_PROCESSES == 1 and NUM_GPUS > 1 and MANAGER_MODEL == "qwen"
    manager_device_map: Union[str, Dict[str, int]] = {"": 0}
    manager_max_memory: Optional[Dict[int, str]] = None
    executor_device_index = 0

    if use_multi_gpu_manager:
        visible_gpus = list(GPU_ASSIGNMENTS[:NUM_GPUS])
        if not visible_gpus:
            raise RuntimeError("GPU_ASSIGNMENTS is empty; cannot enable multi-GPU manager mode.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in visible_gpus)
        manager_device_map = "auto"

        # Build max_memory to exclude GPU0 (reserved for executor)
        manager_max_memory = {0: "0GiB"}
        device_count = torch.cuda.device_count()
        for idx in range(1, device_count):
            total_gb = int(torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3))
            cap_gb = max(total_gb - 2, 1)
            manager_max_memory[idx] = f"{cap_gb}GiB"

        print(
            f"[Process {process_id}] Multi-GPU manager mode enabled. "
            f"Visible GPUs={visible_gpus}, executor_device=0, manager_device_map=auto"
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[Process {process_id}] Single-GPU mode. Visible GPU={gpu_id}")

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
                        elif "id" in data:
                            completed_indices.add(data["id"])
                    except:
                        pass

    remaining_tasks = [idx for idx in task_indices if idx not in completed_indices]
    print(f"[Process {process_id}] Completed: {len(task_indices) - len(remaining_tasks)}, Remaining: {len(remaining_tasks)}")

    if not remaining_tasks:
        print(f"[Process {process_id}] All tasks completed!")
        return

    # Setup Qwen models FIRST (needed for Memcompiler llm_model when MANAGER_MODEL="qwen")
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # Load Executor Qwen model (text-only, 14B)
    print(
        f"[Process {process_id}] Loading Executor Qwen model on device {executor_device_index} "
        f"(visible CUDA devices)."
    )
    executor_tokenizer = AutoTokenizer.from_pretrained(EXECUTOR_QWEN_MODEL_NAME)
    # 配置 8-bit 量化
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    executor_model = AutoModelForCausalLM.from_pretrained(
        EXECUTOR_QWEN_MODEL_NAME,
        # quantization_config=quantization_config,
        device_map={"": executor_device_index},
    ).eval()
    print(f"[Process {process_id}] Executor Qwen model loaded (text-only).")

    # Load Manager Qwen model if MANAGER_MODEL is "qwen"
    manager_model = None
    manager_tokenizer = None
    if MANAGER_MODEL == "qwen":
        # Check if this is a VL model based on model name
        is_vl_model = "VL" in MANAGER_QWEN_MODEL_NAME or "vl" in MANAGER_QWEN_MODEL_NAME.lower()
        
        if is_vl_model:
            print(
                f"[Process {process_id}] Loading Manager Qwen VL model with device_map={manager_device_map} "
                f"(visible CUDA devices)."
            )
            manager_processor = AutoProcessor.from_pretrained(MANAGER_QWEN_MODEL_NAME, use_fast=False)
            manager_tokenizer = AutoTokenizer.from_pretrained(MANAGER_QWEN_MODEL_NAME)
            manager_model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": manager_device_map,
            }
            if manager_max_memory:
                manager_model_kwargs["max_memory"] = manager_max_memory
            manager_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MANAGER_QWEN_MODEL_NAME,
                **manager_model_kwargs,
            ).eval()
            print(f"[Process {process_id}] Manager Qwen VL model loaded.")
        else:
            print(
                f"[Process {process_id}] Loading Manager Qwen model with device_map={manager_device_map} "
                f"(visible CUDA devices)."
            )
            manager_tokenizer = AutoTokenizer.from_pretrained(MANAGER_QWEN_MODEL_NAME)
            manager_model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": manager_device_map,
            }
            if manager_max_memory:
                manager_model_kwargs["max_memory"] = manager_max_memory
            manager_model = AutoModelForCausalLM.from_pretrained(
                MANAGER_QWEN_MODEL_NAME,
                **manager_model_kwargs,
            ).eval()
            print(f"[Process {process_id}] Manager Qwen model loaded (text-only).")

    # Setup GPT client (only needed when MANAGER_MODEL != "qwen")
    gpt_client = None
    if MANAGER_MODEL != "qwen":
        if API_TYPE == "azure":
            # Use Azure OpenAI with hardcoded config
            endpoint_config = AZURE_OPENAI_CONFIG["endpoints"].get(GPT_MODEL)
            if not endpoint_config:
                print(f"[Process {process_id}] ERROR: No Azure endpoint configured for model {GPT_MODEL}!")
                return
            gpt_client = AzureOpenAI(
                api_key=AZURE_OPENAI_CONFIG["api_key"],
                api_version=endpoint_config["api_version"],
                azure_endpoint=endpoint_config["azure_endpoint"]
            )
            print(f"[Process {process_id}] Using Azure OpenAI API with model: {GPT_MODEL}")
        else:
            # Use standard OpenAI
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL", None)
            if not api_key:
                print(f"[Process {process_id}] ERROR: OPENAI_API_KEY not set!")
                return
            gpt_client = OpenAI(api_key=api_key, base_url=base_url)
            print(f"[Process {process_id}] Using standard OpenAI API")

    # Setup Memcompiler (read-only or read-write based on UPDATE_MEMORY)
    resolved_memcompiler_path = resolve_memcompiler_path(MEMCOMPILER_PATH, allow_fallback=not UPDATE_MEMORY)
    print(
        f"[Process {process_id}] Loading Memcompiler from: {resolved_memcompiler_path} "
        f"(configured={MEMCOMPILER_PATH}, UPDATE_MEMORY={UPDATE_MEMORY})"
    )
    from mas.memory.mas_memory.memcompiler import Memcompiler
    from mas.utils import EmbeddingFunc
    from mas.memory.common import MASMessage

    # Create llm_model wrapper based on MANAGER_MODEL
    if MANAGER_MODEL == "qwen":
        # Use Manager Qwen model for Memcompiler's internal LLM calls
        llm_model = QwenLLMWrapper(model=manager_model, tokenizer=manager_tokenizer)
        print(f"[Process {process_id}] Using Qwen as Memcompiler LLM model")
    else:
        # Use GPT for Memcompiler's internal LLM calls
        llm_model = GPTLLMWrapper(client=gpt_client, model=GPT_MODEL)
        print(f"[Process {process_id}] Using GPT as Memcompiler LLM model")

    # Use embedding model from configs.yaml (sentence-transformers/all-MiniLM-L6-v2)
    embedding_func = EmbeddingFunc(model_type="sentence-transformers/all-MiniLM-L6-v2")

    global_config = {
        "working_dir": resolved_memcompiler_path,
        "hop": 1,
        "start_insights_threshold": 5,
        "rounds_per_insights": 5,
        "insights_point_num": 5,
    }

    # Use SafeMemcompiler with ChromaDB HTTP client when UPDATE_MEMORY is True
    if UPDATE_MEMORY:
        SafeMemcompiler = create_safe_memcompiler_class()
        memcompiler_mem = SafeMemcompiler(
            namespace="memcompiler",
            global_config=global_config,
            llm_model=llm_model,
            embedding_func=embedding_func,
        )
        print(f"[Process {process_id}] SafeMemcompiler loaded (ChromaDB mode). Memory size: {memcompiler_mem.memory_size}")
    else:
        memcompiler_mem = Memcompiler(
            namespace="memcompiler",
            global_config=global_config,
            llm_model=llm_model,
            embedding_func=embedding_func,
        )
        print(f"[Process {process_id}] Memcompiler loaded (read-only). Memory size: {memcompiler_mem.memory_size}")

    # Setup ALFWorld environment
    print(f"[Process {process_id}] Loading ALFWorld environment...")

    import alfworld.agents.modules.generic as generic
    import alfworld.agents.environment.alfred_tw_env as alfred_tw_env

    generic.ALFWORLD_CONFIG = CONFIG_FILE
    config = generic.load_config()

    # Directly instantiate the text environment (AlfredTWEnv)
    train_eval_split = resolve_train_eval(DATASET_SPLIT)
    thor_env = alfred_tw_env.AlfredTWEnv(config, train_eval=train_eval_split)
    env_type = 'AlfredTWEnv'  # Fixed to text environment
    print(f"[Process {process_id}] AlfredTWEnv train_eval={train_eval_split}")

    # Different initialization for different env types
    is_text_env = (env_type == 'AlfredTWEnv')
    print(f"[Process {process_id}] Environment type: {env_type}, is_text_env: {is_text_env}")

    if is_text_env:
        # For AlfredTWEnv, we'll create env per game (textworld.gym doesn't support switching games easily)
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        def reset_to_game(game_index: int):
            """Reset to specific game for text environment."""
            task_file = game_file_list[game_index]
            # Create a new gym environment for this specific game
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
            # TextWorld returns obs as tuple with batch_size=1, e.g., ('observation text',)
            # Extract the actual string observation
            if isinstance(obs, (list, tuple)) and len(obs) > 0:
                obs_str = obs[0]
            else:
                obs_str = obs
            return [obs_str], infos, env

        env = None  # Will be created per game
    else:
        # For AlfredThorEnv, use the original mechanism
        env = thor_env.init_env(batch_size=1)

        def reset_to_game(game_index: int):
            """Reset to specific game for Thor environment."""
            task_file = game_file_list[game_index]
            thor_env.action_queues[0].put((None, True, task_file))
            obs, dones, infos = thor_env.wait_and_get_info()
            return obs, infos, None  # Return None for env (use existing)

    # Statistics
    stats = {'total': 0, 'success': 0, 'failed': 0}

    # Main loop
    for task_num, target_idx in enumerate(remaining_tasks):
        try:
            print(f"\n[Process {process_id}] Task {task_num + 1}/{len(remaining_tasks)} (env index: {target_idx})")

            # Reset to target game
            obs, info, new_env = reset_to_game(target_idx)
            if new_env is not None:
                env = new_env  # For text env, use the newly created env

            initial_obs = obs[0] if isinstance(obs, (list, tuple)) else obs
            gamefile = info.get('extra.gamefile', ['unknown'])
            if isinstance(gamefile, list):
                gamefile = gamefile[0]

            task_text = extract_task(initial_obs)
            print(f"[Process {process_id}] Task: {task_text[:80]}...")

            # Initialize token usage counter for this task
            task_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            # Reset Memcompiler LLM wrapper token counter (if using GPT)
            if MANAGER_MODEL != "qwen" and hasattr(llm_model, 'reset_token_counter'):
                llm_model.reset_token_counter()

            # Retrieve Task Memory (once per task, read-only)
            # print(f"[DEBUG Process {process_id}] Calling memcompiler_mem.retrieve_memory with query_task='{task_text}'")
            successful_tasks, failed_tasks, insights = memcompiler_mem.retrieve_memory(
                query_task=task_text,
                successful_topk=1,
                failed_topk=0,
                insight_topk=5,
                threshold=0.0
            )

            runtime_key_steps = []
            for success_task in list(successful_tasks or [])[:1]:
                key_steps = generate_runtime_key_steps(success_task=success_task, llm_model=llm_model)
                runtime_key_steps.append(key_steps)

            task_memory = {
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "insights": insights,
                "runtime_key_steps": runtime_key_steps,
            }
            print(f"[Process {process_id}] Task Memory: {len(successful_tasks or [])} success, {len(failed_tasks or [])} fail, {len(insights or [])} insights, {len([k for k in runtime_key_steps if k])} key_steps")

            # Initialize per-episode state
            history = []
            detailed_steps = []
            working_memory = ""
            step_count = 0
            episode_done = False
            episode_won = False
            failure_reason = None

            # Episode execution loop
            while not episode_done:
                step_count += 1

                if step_count > MAX_STEPS:
                    print(f"[Process {process_id}] Max steps reached. Episode failed.")
                    failure_reason = "max_steps_exceeded"
                    break

                # Get admissible commands (different format for different envs)
                admissible = None
                if isinstance(info, dict) and "admissible_commands" in info:
                    try:
                        ac = info["admissible_commands"]
                        # Both envs may return nested lists with batch_size=1
                        # Need to extract the actual list of command strings
                        if ac:
                            # If it's a list of lists, get the first inner list
                            if isinstance(ac, (list, tuple)) and len(ac) > 0:
                                if isinstance(ac[0], (list, tuple)):
                                    # Nested: [['cmd1', 'cmd2', ...]]
                                    admissible = list(ac[0])
                                else:
                                    # Flat: ['cmd1', 'cmd2', ...]
                                    admissible = list(ac)
                    except Exception as e:
                        print(f"[Process {process_id}] Warning: Failed to extract admissible_commands: {e}")
                        pass

                current_obs = obs[0] if isinstance(obs, (list, tuple)) else obs

                # Step 1: Call Manager (GPT or Qwen)
                if MANAGER_MODEL == "qwen":
                    manager_raw_response, manager_input_fields = call_manager_qwen(
                        model=manager_model,
                        tokenizer=manager_tokenizer,
                        working_memory=working_memory,
                        task_text=task_text,
                        history=history,
                        current_obs=current_obs,
                        admissible_commands=admissible,
                        task_memory=task_memory,
                        initial_observation=initial_obs,
                    )
                else:
                    manager_raw_response, manager_input_fields, step_token_usage = call_gpt(
                        client=gpt_client,
                        model=GPT_MODEL,
                        working_memory=working_memory,
                        task_text=task_text,
                        history=history,
                        current_obs=current_obs,
                        admissible_commands=admissible,
                        task_memory=task_memory,
                        initial_observation=initial_obs,
                    )
                    # Accumulate token usage for this task
                    task_token_usage["prompt_tokens"] += step_token_usage["prompt_tokens"]
                    task_token_usage["completion_tokens"] += step_token_usage["completion_tokens"]
                    task_token_usage["total_tokens"] += step_token_usage["total_tokens"]

                parsed_manager = parse_gpt_response(manager_raw_response)
                manager_type = parsed_manager.get("type", "NOACTION")

                # Step 2: Process Manager output
                manager_guidance = None

                # Build enhanced guidance: {guidance} REASON: {brief_reason}
                def build_enhanced_guidance(parsed: Dict[str, Any]) -> Optional[str]:
                    guidance = parsed.get("guidance", "")
                    brief_reason = parsed.get("brief_reason", "")
                    if brief_reason:
                        # 使用正则表达式匹配 executor（忽略大小写）
                        # \b 确保是完整单词匹配，避免匹配到 executioner 等词
                        brief_reason = re.sub(r'\bexecutor\b', r'\g<0>(You)', brief_reason, flags=re.IGNORECASE)
                    if guidance:
                        if brief_reason:
                            return f"{guidance} REASON: {brief_reason}"
                        return guidance
                    return None

                if manager_type == "EMBODIED":
                    manager_guidance = build_enhanced_guidance(parsed_manager)
                elif manager_type == "CONTEXT":
                    working_memory = apply_memory_operation(working_memory, parsed_manager)
                elif manager_type == "HYBRID":
                    # HYBRID: both guidance and memory edit
                    manager_guidance = build_enhanced_guidance(parsed_manager)
                    working_memory = apply_memory_operation(working_memory, parsed_manager)

                # Step 3: Call Qwen Executor
                action, qwen_raw_response, qwen_input_fields = qwen_choose_action(
                    model=executor_model,
                    tokenizer=executor_tokenizer,
                    working_memory=working_memory,
                    task_text=task_text,
                    admissible=admissible,
                    history=history,
                    current_obs=current_obs,
                    gpt_guidance=manager_guidance,
                    max_history_turns=12
                )

                # Step 4: Execute action
                if is_text_env:
                    # TextWorld gym: step takes list of actions even with batch_size=1
                    # Returns: obs (tuple), scores (tuple), dones (tuple), infos (dict)
                    obs_result, scores, dones, info = env.step([action])
                    # Extract string from tuple/list (batch format)
                    if isinstance(obs_result, (list, tuple)) and len(obs_result) > 0:
                        observation = obs_result[0]
                    else:
                        observation = obs_result
                    obs = [observation]  # Update obs for next iteration
                    dones = list(dones) if isinstance(dones, (list, tuple)) else [dones]
                else:
                    # Thor env: step takes list of actions
                    actions = [action]
                    obs_result, scores, dones, info = env.step(actions)
                    observation = obs_result[0]
                    obs = obs_result  # Update obs for next iteration

                history.append({"action": action, "observation": observation})

                # Record detailed step with all input/output fields
                step_record = {
                    "step": step_count,
                    "manager": {
                        "model": MANAGER_MODEL,
                        "input": manager_input_fields,
                        "output": {"raw": manager_raw_response}
                    },
                    "executor": {
                        "input": qwen_input_fields,
                        "output": {"raw": qwen_raw_response, "action": action}
                    },
                    "thor": {"action": action, "observation": observation}
                }
                detailed_steps.append(step_record)

                # Step 5: Check if done
                if dones[0]:
                    episode_done = True
                    won_value = info.get('won', [False])[0] if isinstance(info.get('won'), list) else info.get('won', False)
                    # Convert numpy.bool_ to Python bool for JSON serialization
                    episode_won = bool(won_value)
                    print(f"[Process {process_id}] Episode finished! Won: {episode_won}, Steps: {len(history)}")

            # Episode ended - update stats
            stats['total'] += 1
            if episode_won:
                stats['success'] += 1
            else:
                stats['failed'] += 1

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

            # Append to overview (with file lock)
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
                try:
                    mas_message = MASMessage(
                        task_main=task_text,
                        task_description=task_text,
                        label=episode_won,
                    )
                    mas_message.add_extra_field("gamefile", gamefile)
                    for h in history:
                        mas_message.move_state(h["action"], h["observation"])

                    # Add to memory
                    memcompiler_mem.add_memory(mas_message)
                    print(f"[Process {process_id}] Added task to Memcompiler. New size: {memcompiler_mem.memory_size}")
                except Exception as e:
                    print(f"[Process {process_id}] Failed to add memory: {e}")

            # Print progress and token usage
            acc = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            print(f"[Process {process_id}] Progress: {stats['total']}/{len(remaining_tasks)} | Success: {stats['success']} | Acc: {acc:.2%}")

            # Add Memcompiler LLM token usage to task total (if using GPT)
            memcompiler_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if MANAGER_MODEL != "qwen" and hasattr(llm_model, 'get_accumulated_tokens'):
                memcompiler_tokens = llm_model.get_accumulated_tokens()

            total_prompt = task_token_usage['prompt_tokens'] + memcompiler_tokens['prompt_tokens']
            total_completion = task_token_usage['completion_tokens'] + memcompiler_tokens['completion_tokens']
            total_all = task_token_usage['total_tokens'] + memcompiler_tokens['total_tokens']

            print(f"[Process {process_id}] 📊 Task {target_idx} Token Usage:")
            print(f"    Manager API: prompt={task_token_usage['prompt_tokens']}, completion={task_token_usage['completion_tokens']}, total={task_token_usage['total_tokens']}")
            print(f"    Memcompiler API: prompt={memcompiler_tokens['prompt_tokens']}, completion={memcompiler_tokens['completion_tokens']}, total={memcompiler_tokens['total_tokens']}")
            print(f"    TOTAL: prompt={total_prompt}, completion={total_completion}, total={total_all}")

            # Close text env after each task to avoid resource leaks
            if is_text_env and env is not None:
                try:
                    env.close()
                except:
                    pass

        except Exception as e:
            print(f"[Process {process_id}] Error on task {target_idx}: {e}")
            traceback.print_exc()
            stats['total'] += 1
            stats['failed'] += 1

    # Final summary
    print(f"\n[Process {process_id}] COMPLETED")
    print(f"[Process {process_id}] Total: {stats['total']} | Success: {stats['success']} | Failed: {stats['failed']}")


# =============================================================================
# Main Function
# =============================================================================
def main():
    print("=" * 60)
    print("Thor GPT+Qwen Multi-Process")
    print("=" * 60)
    print(f"NUM_PROCESSES: {NUM_PROCESSES}")
    print(f"NUM_GPUS: {NUM_GPUS}")
    print(f"OUTPUT_PATH: {OUTPUT_PATH}")
    print(f"MEMCOMPILER_PATH: {MEMCOMPILER_PATH}")
    print(f"DATASET_SPLIT: {DATASET_SPLIT}")
    print(f"UPDATE_MEMORY: {UPDATE_MEMORY}")

    # Setup directories
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "trajectory"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "detailed_trajectory"), exist_ok=True)

    # Get json_file_list from ALFWorld environment (in main process)
    print("\nInitializing ALFWorld environment to get game list...")

    import alfworld.agents.modules.generic as generic
    import alfworld.agents.environment.alfred_tw_env as alfred_tw_env

    generic.ALFWORLD_CONFIG = CONFIG_FILE
    config = generic.load_config()
    env_type = config["env"]["type"]
    train_eval_split = resolve_train_eval(DATASET_SPLIT)

    # Directly instantiate the text environment
    thor_env = alfred_tw_env.AlfredTWEnv(config, train_eval=train_eval_split)

    # Get game file list (different attribute names for different env types)
    game_file_list = None
    total_games = None
    if hasattr(thor_env, "json_file_list"):
        game_file_list = thor_env.json_file_list
        total_games = len(game_file_list)
    elif hasattr(thor_env, "game_files"):
        game_file_list = thor_env.game_files
        total_games = len(game_file_list)
    elif hasattr(thor_env, "num_games"):
        total_games = int(thor_env.num_games)
    if not total_games:
        total_games = 3000
    print(f"Total games in split '{train_eval_split}': {total_games}")
    print(f"Environment type: {env_type}")

    target_indices = resolve_target_indices(total_games, DATASET_SPLIT)
    # 当重新运行失败的任务时，可临时手动覆盖 target_indices
    print(f"Target task indices count: {len(target_indices)}")

    # Filter out already completed tasks from overview.jsonl
    overview_path = os.path.join(OUTPUT_PATH, "overview.jsonl")
    completed_indices = set()
    if os.path.exists(overview_path):
        print(f"Checking completed tasks from: {overview_path}")
        with open(overview_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    env_index = record.get("env_index")
                    if env_index is not None:
                        completed_indices.add(env_index)
                except json.JSONDecodeError:
                    continue
        print(f"Already completed tasks: {len(completed_indices)}")
    else:
        print("No existing overview.jsonl found, starting fresh.")

    # Remove completed tasks from the target list
    uncompleted_indices = sorted([idx for idx in target_indices if idx not in completed_indices])
    print(f"Uncompleted tasks to run: {len(uncompleted_indices)}")

    if len(uncompleted_indices) == 0:
        print("All tasks already completed! Exiting.")
        return

    # Static task pre-allocation (only uncompleted tasks)
    task_assignments = [[] for _ in range(NUM_PROCESSES)]
    for i, idx in enumerate(uncompleted_indices):
        task_assignments[i % NUM_PROCESSES].append(idx)

    for i, tasks in enumerate(task_assignments):
        print(f"Process {i}: {len(tasks)} tasks")

    # GPU assignments (4 processes × 4 GPUs)
    gpu_assignments = GPU_ASSIGNMENTS #[0,1,3,4]

    # Start worker processes
    print("\n" + "=" * 60)
    print("Starting worker processes...")
    print("=" * 60)

    processes = []
    for i in range(NUM_PROCESSES):
        p = mp.Process(
            target=worker_process,
            args=(
                i,
                gpu_assignments[i],
                task_assignments[i],
                game_file_list,
            )
        )
        p.start()
        processes.append(p)
        print(f"Started process {i} on GPU {gpu_assignments[i]}")

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("\n" + "=" * 60)
    print("ALL PROCESSES COMPLETED")
    print("=" * 60)

    # Final summary from overview file
    overview_file = os.path.join(OUTPUT_PATH, "overview.jsonl")
    if os.path.exists(overview_file):
        total = 0
        success = 0
        with open(overview_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    total += 1
                    if data.get("won"):
                        success += 1
                except:
                    pass
        print(f"Total: {total} | Success: {success} | Acc: {success/total:.2%}" if total > 0 else "No results")

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
