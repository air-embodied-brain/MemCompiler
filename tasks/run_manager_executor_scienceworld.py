#!/usr/bin/env python
"""
ScienceWorld Manager-Executor script with GPT + Qwen collaboration - Multi-process version.
Uses ScienceWorldEnv for science experiment tasks with Memcompiler integration.

Usage:
    conda activate scienceworld && python tasks/run_manager_executor_scienceworld.py
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

# Ensure project root is on sys.path so that 'mas' and 'tasks/' modules can be found
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_tasks_dir = os.path.abspath(os.path.dirname(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _tasks_dir not in sys.path:
    sys.path.insert(0, _tasks_dir)

# Import reusable functions from old script
from run_manager_executor import (
    FileLock,
    parse_gpt_response,
    apply_memory_operation,
    generate_runtime_key_steps,
    QwenLLMWrapper,
    get_memcompiler_embedding_count,
)

# Import ScienceWorld prompts
from prompts.scienceworld_prompt import (
    GPT_SYSTEM_PROMPT_SCIENCEWORLD,
    QWEN_SYSTEM_PROMPT_SCIENCEWORLD,
    score_to_progress,
    create_gpt_user_content_scienceworld,
    create_qwen_user_content_scienceworld,
    format_task_memory,
    apply_scienceworld_insights_prompts,
)
from prompts.scienceworld_mas_prompt import scienceworld_few_shots

# Import ScienceWorld environment
from envs.scienceworld_env.scienceworld_env import (
    ScienceWorldEnvWrapper,
    get_scienceworld_task_configs,
)

# =============================================================================
# Configuration
# =============================================================================
NUM_PROCESSES = 4  # ScienceWorld JVM is heavy; start conservative
NUM_GPUS = 4

OUTPUT_PATH = "./output/scienceworld"
MEMCOMPILER_PATH = "./output/memcompiler"

# ScienceWorld-specific configuration
TASK_SPLIT = "train"
TASK_NAMES = None  # None = all 30 tasks; or specify e.g. ["boil", "melt"]
SIMPLIFICATION_STR = ""  # Empty = auto (easy for non-electrical, reduced for electrical)
MAX_STEPS = 30  # ScienceWorld default envStepLimit

# Task loading mode: "env" (from ScienceWorld JVM) or "jsonl" (from file)
TASK_LOAD_MODE = "env"
TEST_JSONL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sciworld", "test.jsonl")

# SentenceTransformer for action matching (same model as CLIN)
SENT_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ACTION_MATCH_THRESHOLD = 0.9
ACTION_MATCH_MAX_RETRIES = 3

# Executor model (low-level action) - Text-only Qwen
EXECUTOR_QWEN_MODEL_NAME = "YOUR_EXECUTOR_MODEL_PATH"

# Manager model configuration: "gpt", "qwen", or "gemini"
MANAGER_MODEL = "gemini"   # 可选gemini，注意的是目前设置是memcompiler内置LLM和manager用的是一样的
MANAGER_QWEN_MODEL_NAME = "YOUR_MANAGER_MODEL_PATH"
GPT_MODEL = "mgg-7"   #可填gemini系列

# API type configuration: "openai" or "azure"
API_TYPE = "openai"  # 只对gpt有用

# Azure OpenAI configuration
AZURE_OPENAI_CONFIG = {
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY", "YOUR_AZURE_API_KEY"),
    "endpoints": {
        "gpt-4o": {
            "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", "https://YOUR_ENDPOINT.openai.azure.com/"),
            "api_version": "2025-01-01-preview",
        },
    }
}

# Gemini API mode: "official" (Google official) or "ksyun" (Ksyun proxy)
GEMINI_API_MODE = "official"

# Ksyun Gemini proxy configuration (example)
KSYUN_GEMINI_CONFIG = {
    "api_key": os.environ.get("KSYUN_GEMINI_API_KEY", "YOUR_KSYUN_API_KEY"),
    "base_url": os.environ.get("KSYUN_GEMINI_BASE_URL", "http://YOUR_PROXY_HOST"),
    "api_version": "v1",
}

# API retry configuration
API_MAX_RETRIES = 5
API_RETRY_DELAY = 5  # seconds

GPU_ASSIGNMENTS = [1,2,3,0]

# Memory update configuration
UPDATE_MEMORY = True
CHROMA_HOST = "localhost"
CHROMA_PORT = 8058

DEFAULT_MEMCOMPILER_FALLBACK_PATHS = [
    # Add fallback paths if needed
]


# =============================================================================
# JSONL Task Loading
# =============================================================================
def load_scienceworld_tasks_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load ScienceWorld tasks from a JSONL file.

    Each line: {"task": "scienceworld", "id": N, "goal": "...",
                "subgoals": [...], "difficulty": "easy"|"hard",
                "additional_info": {"var": N, "env_name": "..."}}

    Returns list of dicts with keys matching Memcompiler-main's sciworld_tasks schema:
    task_name, variation_idx, simplification_str, id, modified_goal, subgoals, difficulty
    """
    # Unified simplification for all tasks (aligned with Memcompiler-main)
    simplification = 'selfWateringFlowerPots,openContainers,openDoors,noElectricalAction'
    tasks = []
    with open(path, 'r') as f:
        for line in f:
            row = json.loads(line)
            info = row["additional_info"]
            tasks.append({
                "id": row["id"],
                "task_name": info["env_name"],
                "variation_idx": info["var"],
                "modified_goal": row["goal"],
                "subgoals": row["subgoals"],
                "difficulty": row.get("difficulty", "easy"),
                "simplification_str": simplification,
            })
    print(f"[ScienceWorld] Loaded {len(tasks)} tasks from {path}")
    return tasks


# =============================================================================
# resolve_memcompiler_path (same as PDDL)
# =============================================================================
def resolve_memcompiler_path(configured_path: str, allow_fallback: bool = True) -> str:
    """Check configured and fallback Memcompiler paths, return the best one."""
    if not allow_fallback:
        return configured_path

    candidate_paths = [configured_path] + DEFAULT_MEMCOMPILER_FALLBACK_PATHS
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


# =============================================================================
# SafeMemcompiler (same as PDDL)
# =============================================================================
def create_safe_memcompiler_class():
    """Create SafeMemcompiler class after imports are set up. Only used when UPDATE_MEMORY=True."""
    import pickle
    from mas.memory.mas_memory import Memcompiler
    from mas.memory.mas_memory.memcompiler import TaskLayer, InsightsManager
    from mas.utils import write_json

    class SafeTaskLayer(TaskLayer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._lock_file = self._graph_save_path + ".lock"

        def _index_done(self) -> None:
            with open(self._graph_save_path, "wb") as f:
                pickle.dump(self.graph, f)

        def add_task_node(self, task_main: str) -> None:
            import networkx as nx
            with FileLock(self._lock_file):
                if os.path.exists(self._graph_save_path):
                    with open(self._graph_save_path, 'rb') as f:
                        self.graph = pickle.load(f)
                if task_main in self.graph:
                    return
                self.graph.add_node(task_main)
                results = self.task_storage.similarity_search_with_score(query=task_main, k=10)
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
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._lock_file = self.persist_file + ".lock"

        def _index_done(self):
            with FileLock(self._lock_file):
                write_json(self.insights_memory, self.persist_file)

    class SafeMemcompiler(Memcompiler):
        def __post_init__(self):
            import chromadb
            from langchain_chroma import Chroma

            persist_base_dir = self.global_config["working_dir"]
            os.makedirs(persist_base_dir, exist_ok=True)
            self.persist_dir = persist_base_dir

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

            self.task_layer = SafeTaskLayer(
                working_dir=self.persist_dir, namespace='task_layer', task_storage=self.main_memory
            )
            self.insights_layer = SafeInsightsManager(
                working_dir=self.persist_dir, namespace='insights',
                llm_model=self.llm_model, task_storage=self.main_memory, task_layer=self.task_layer
            )
            self.insights_cache = []
            self._insights_lock_file = os.path.join(self.persist_dir, "insights_update.lock")
            print(
                f"[SafeMemcompiler] Initialized with ChromaDB HTTP client at {CHROMA_HOST}:{CHROMA_PORT} | "
                f"persist_dir={self.persist_dir} | collection=langchain"
            )

        def add_memory(self, mas_message) -> None:
            from langchain.docstore.document import Document
            from mas.memory.common import MASMessage

            mas_message = self._extract_mas_message(mas_message=mas_message)
            self.task_layer.add_task_node(mas_message.task_main)
            meta_data = MASMessage.to_dict(mas_message)
            memory_doc = Document(page_content=mas_message.task_main, metadata=meta_data)
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
            pass

    return SafeMemcompiler


# =============================================================================
# GPTLLMWrapper (same as PDDL)
# =============================================================================
class GPTLLMWrapper:
    """Wrapper to make GPT API compatible with Memcompiler's LLMCallable interface."""

    def __init__(self, client: Union[OpenAI, AzureOpenAI], model: str):
        self.client = client
        self.model = model
        self.accumulated_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def reset_token_counter(self):
        self.accumulated_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_accumulated_tokens(self) -> Dict[str, int]:
        return self.accumulated_tokens.copy()

    def __call__(self, messages, temperature: float = 0.0, max_tokens: int = 512, **kwargs) -> str:
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]
        for attempt in range(API_MAX_RETRIES):
            # try:
            #     response = self.client.chat.completions.create(
            #         model=self.model, messages=openai_messages,
            #         max_completion_tokens=4096, timeout=120, reasoning_effort="low"
            #     )
            #     if response.usage:
            #         self.accumulated_tokens["prompt_tokens"] += response.usage.prompt_tokens
            #         self.accumulated_tokens["completion_tokens"] += response.usage.completion_tokens
            #         self.accumulated_tokens["total_tokens"] += response.usage.total_tokens
            #     content = response.choices[0].message.content
            #     finish_reason = response.choices[0].finish_reason
            #     if content is None or content == "":
            #         if finish_reason == "length":
            #             time.sleep(API_RETRY_DELAY)
            #             continue
            #         return ""
            #     return content
            # except Exception as e:
            #     error_str = str(e)
            #     if "429" in error_str or "RateLimitReached" in error_str:
            #         time.sleep(API_RETRY_DELAY)
            #         continue
            #     print(f"GPT API error in Memcompiler: {e}")
            #     return ""

            response = self.client.chat.completions.create(
                model=self.model, messages=openai_messages,
                max_completion_tokens=4096, timeout=120, reasoning_effort="low"
            )
            if response.usage:
                self.accumulated_tokens["prompt_tokens"] += response.usage.prompt_tokens
                self.accumulated_tokens["completion_tokens"] += response.usage.completion_tokens
                self.accumulated_tokens["total_tokens"] += response.usage.total_tokens
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            if content is None or content == "":
                if finish_reason == "length":
                    time.sleep(API_RETRY_DELAY)
                    continue
                return ""
            return content
        return ""


# =============================================================================
# GeminiLLMWrapper
# =============================================================================
class GeminiLLMWrapper:
    """Wrapper to make Gemini API compatible with Memcompiler's LLMCallable interface."""

    def __init__(self, client, model: str):
        self.client = client  # google.genai.Client
        self.model = model
        self.accumulated_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        from google.genai import types as genai_types
        self._genai_types = genai_types

    def reset_token_counter(self):
        self.accumulated_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_accumulated_tokens(self) -> Dict[str, int]:
        return self.accumulated_tokens.copy()

    def __call__(self, messages, temperature: float = 0.0, max_tokens: int = 512, **kwargs) -> str:
        system_parts = []
        contents = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "user":
                contents.append(self._genai_types.Content(
                    role="user", parts=[self._genai_types.Part(text=m.content)]
                ))
            elif m.role == "assistant":
                contents.append(self._genai_types.Content(
                    role="model", parts=[self._genai_types.Part(text=m.content)]
                ))

        config_kwargs = {"temperature": temperature, "max_output_tokens": max_tokens}
        if system_parts:
            config_kwargs["system_instruction"] = "\n".join(system_parts)
        config = self._genai_types.GenerateContentConfig(**config_kwargs)

        for attempt in range(API_MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config,
                )
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    pt = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                    ct = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
                    self.accumulated_tokens["prompt_tokens"] += pt
                    self.accumulated_tokens["completion_tokens"] += ct
                    self.accumulated_tokens["total_tokens"] += pt + ct
                content = response.text
                if content is None or content == "":
                    return ""
                return content
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    time.sleep(API_RETRY_DELAY)
                    continue
                print(f"Gemini API error in Memcompiler: {e}")
                return ""
        return ""


# =============================================================================
# ScienceWorld Action Matching (NEW)
# =============================================================================
def match_action_scienceworld(
    raw_action: str,
    valid_actions: List[str],
    sent_transformer_model,
    threshold: float = ACTION_MATCH_THRESHOLD,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Match LLM-generated free-text action to valid environment actions.

    Uses exact match → case-insensitive match → SentenceTransformer fuzzy match.

    Returns:
        (best_match_action, best_score, top5_matches)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if not valid_actions:
        return raw_action, 0.0, []

    raw_lower = raw_action.lower().strip()

    # --- CLIN-style focus guard (applied BEFORE all matching steps) ---
    # "focus on" is a critical answer-submission action in ScienceWorld.
    # Focusing on the wrong object instantly fails the task (score = -100).
    # To prevent accidental triggering via fuzzy match:
    #   - If LLM output contains "focus" → only consider focus actions
    #   - If LLM output does NOT contain "focus" → exclude ALL focus actions
    llm_wants_focus = 'focus' in raw_lower
    candidate_actions = [a for a in valid_actions if 'reset' not in a.lower()]
    if llm_wants_focus:
        candidate_actions = [a for a in candidate_actions if 'focus' in a.lower()]
    else:
        candidate_actions = [a for a in candidate_actions if 'focus' not in a.lower()]

    # Fallback: if filtering leaves nothing, use all non-reset actions
    if not candidate_actions:
        candidate_actions = [a for a in valid_actions if 'reset' not in a.lower()]
        if not candidate_actions:
            candidate_actions = valid_actions

    # 1. Exact match (within filtered candidates)
    if raw_action in candidate_actions:
        return raw_action, 1.0, [(raw_action, 1.0)]

    # 2. Case-insensitive match (within filtered candidates)
    for va in candidate_actions:
        if va.lower().strip() == raw_lower:
            return va, 1.0, [(va, 1.0)]

    # 3. SentenceTransformer fuzzy match (within filtered candidates)
    action_embeddings = sent_transformer_model.encode(candidate_actions)
    query_embedding = sent_transformer_model.encode([raw_action])
    similarities = cosine_similarity(query_embedding, action_embeddings)[0]

    top_indices = similarities.argsort()[::-1][:5]
    top5 = [(candidate_actions[i], float(similarities[i])) for i in top_indices]
    best_idx = top_indices[0]

    return candidate_actions[best_idx], float(similarities[best_idx]), top5


def parse_action_scienceworld(raw_response: str) -> str:
    """Extract and clean the action from the Executor's raw text output.

    Since ScienceWorld Executor outputs plain text (no <action> tags),
    we take the last non-empty line and strip numbering/quotes.
    """
    if not raw_response or not raw_response.strip():
        return "look around"

    # Take the last non-empty line (most likely the action)
    lines = [line.strip() for line in raw_response.strip().split('\n') if line.strip()]
    if not lines:
        return "look around"

    action = lines[-1]

    # Remove possible prefix numbering (e.g., "1. pick up thermometer", "1) move to kitchen")
    action = re.sub(r'^\d+[\.\)]\s*', '', action)
    # Remove possible quote wrapping
    action = action.strip('"').strip("'")
    # Remove leading/trailing whitespace
    action = action.strip()

    return action if action else "look around"


# =============================================================================
# call_gpt (adapted for ScienceWorld)
# =============================================================================
def call_gpt(
    client: Union[OpenAI, AzureOpenAI],
    model: str,
    working_memory: str,
    task_text: str,
    history: Optional[List[Dict[str, Any]]],
    current_obs: str,
    current_score: int,
    task_memory: Optional[Dict[str, Any]] = None,
    initial_observation: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    user_content, input_fields = create_gpt_user_content_scienceworld(
        working_memory=working_memory, task_text=task_text, history=history,
        current_obs=current_obs, current_score=current_score,
        task_memory=task_memory, initial_observation=initial_observation,
        few_shots=[],
    )
    messages = [
        {"role": "system", "content": GPT_SYSTEM_PROMPT_SCIENCEWORLD},
        {"role": "user", "content": user_content}
    ]
    for attempt in range(API_MAX_RETRIES):
        # try:
        #     response = client.chat.completions.create(
        #         model=model, messages=messages,
        #         max_completion_tokens=4096, timeout=120, reasoning_effort="low"
        #     )
        #     print(response)
        #     content = response.choices[0].message.content
        #     finish_reason = response.choices[0].finish_reason
        #     if content is None or content == "":
        #         if finish_reason == "length":
        #             time.sleep(API_RETRY_DELAY)
        #             continue
        #         content = ""
        #     token_usage = {
        #         "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        #         "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        #         "total_tokens": response.usage.total_tokens if response.usage else 0
        #     }
        #     return content, input_fields, token_usage
        # except Exception as e:
        #     error_str = str(e)
        #     if "429" in error_str or "RateLimitReached" in error_str:
        #         time.sleep(API_RETRY_DELAY)
        #         continue
        #     print(f"  GPT API error: {e}")
        #     return "<response_type>NOACTION</response_type>", input_fields, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        print(client.base_url,55555555555555555555555555555555555)
        response = client.chat.completions.create(
            model=model, messages=messages,
            max_completion_tokens=4096, timeout=120, reasoning_effort="low"
        )
        print(response,666)
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        if content is None or content == "":
            if finish_reason == "length":
                time.sleep(API_RETRY_DELAY)
                continue
            content = ""
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0
        }
        return content, input_fields, token_usage
    return "<response_type>NOACTION</response_type>", input_fields, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# =============================================================================
# call_gemini (adapted for ScienceWorld Manager)
# =============================================================================
def call_gemini(
    client,  # google.genai.Client
    model: str,
    working_memory: str,
    task_text: str,
    history: Optional[List[Dict[str, Any]]],
    current_obs: str,
    current_score: int,
    task_memory: Optional[Dict[str, Any]] = None,
    initial_observation: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    from google.genai import types as genai_types

    user_content, input_fields = create_gpt_user_content_scienceworld(
        working_memory=working_memory, task_text=task_text, history=history,
        current_obs=current_obs, current_score=current_score,
        task_memory=task_memory, initial_observation=initial_observation,
        few_shots=[],
    )
    # Convert content list to text for Gemini
    if isinstance(user_content, list):
        text_content = user_content[0]["text"] if user_content else ""
    else:
        text_content = user_content

    contents = [genai_types.Content(
        role="user", parts=[genai_types.Part(text=text_content)]
    )]
    config = genai_types.GenerateContentConfig(
        system_instruction=GPT_SYSTEM_PROMPT_SCIENCEWORLD,
        temperature=0.0,
        max_output_tokens=4096,
    )

    for attempt in range(API_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config,
            )
            # print(response)
            # print(222222222222222222222222222222)
            content = response.text
            if content is None or content == "":
                content = ""
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_usage["prompt_tokens"] = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                token_usage["completion_tokens"] = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
                token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
            return content, input_fields, token_usage
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                time.sleep(API_RETRY_DELAY)
                continue
            print(f"  Gemini API error: {e}")
            return "<response_type>NOACTION</response_type>", input_fields, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    # for attempt in range(API_MAX_RETRIES):
        
    #     response = client.models.generate_content(
    #         model=model, contents=contents, config=config,
    #     )
    #     print(response)
    #     print(222222222222222222222222222222)
    #     content = response.text
    #     if content is None or content == "":
    #         content = ""
    #     token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    #     if hasattr(response, 'usage_metadata') and response.usage_metadata:
    #         token_usage["prompt_tokens"] = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
    #         token_usage["completion_tokens"] = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
    #         token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
    #     return content, input_fields, token_usage
    
    # return "<response_type>NOACTION</response_type>", input_fields, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# =============================================================================
# call_manager_qwen (adapted for ScienceWorld)
# =============================================================================
@torch.no_grad()
def call_manager_qwen(
    model, tokenizer,
    working_memory: str, task_text: str,
    history: Optional[List[Dict[str, Any]]], current_obs: str,
    current_score: int,
    task_memory: Optional[Dict[str, Any]] = None,
    initial_observation: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    text_content, input_fields = create_gpt_user_content_scienceworld(
        working_memory=working_memory, task_text=task_text, history=history,
        current_obs=current_obs, current_score=current_score,
        task_memory=task_memory, initial_observation=initial_observation,
        few_shots=[],
    )
    # create_gpt_user_content_scienceworld returns (content_list, input_fields)
    # Extract the text from content_list for Qwen
    if isinstance(text_content, list):
        text_for_qwen = text_content[0]["text"] if text_content else ""
    else:
        text_for_qwen = text_content

    messages = [
        {"role": "system", "content": GPT_SYSTEM_PROMPT_SCIENCEWORLD},
        {"role": "user", "content": text_for_qwen}
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return response, input_fields
    except Exception as e:
        print(f"  Manager Qwen error: {e}")
        return "<response_type>NOACTION</response_type>", input_fields


# =============================================================================
# qwen_choose_action (adapted for ScienceWorld)
# =============================================================================
@torch.no_grad()
def qwen_choose_action(
    model, tokenizer,
    working_memory: str, task_text: str,
    action_templates: List[str],
    available_objects: List[str],
    history: Optional[List[Dict[str, Any]]], current_obs: str,
    current_inventory: str,
    gpt_guidance: Optional[str] = None,
    max_history_turns: int = 12,
) -> Tuple[str, str, Dict[str, Any]]:
    """Use Qwen to generate next action for ScienceWorld.

    Returns:
        (raw_action, raw_response, input_fields)
        Note: raw_action is the cleaned text, NOT yet matched to valid actions.
    """
    if history and len(history) > max_history_turns:
        history = history[-max_history_turns:]

    text_content, input_fields = create_qwen_user_content_scienceworld(
        working_memory=working_memory, task_text=task_text,
        action_templates=action_templates, available_objects=available_objects,
        history=history, current_obs=current_obs,
        current_inventory=current_inventory, gpt_guidance=gpt_guidance,
    )
    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT_SCIENCEWORLD},
        {"role": "user", "content": text_content},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    print(f"[SW Executor] raw response: {response}")
    raw_action = parse_action_scienceworld(response)
    print(f"[SW Executor] parsed action: {raw_action}")

    return raw_action, response, input_fields


# =============================================================================
# Worker Process (adapted for ScienceWorld)
# =============================================================================
def worker_process(
    process_id: int,
    gpu_id: int,
    task_indices: List[int],
    all_task_configs: List[Dict[str, Any]],
):
    """Worker process that handles a subset of ScienceWorld tasks."""
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
            raise RuntimeError("GPU_ASSIGNMENTS is empty.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in visible_gpus)
        manager_device_map = "auto"
        manager_max_memory = {0: "0GiB"}
        device_count = torch.cuda.device_count()
        for idx in range(1, device_count):
            total_gb = int(torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3))
            cap_gb = max(total_gb - 2, 1)
            manager_max_memory[idx] = f"{cap_gb}GiB"
        print(f"[Process {process_id}] Multi-GPU manager mode. Visible GPUs={visible_gpus}")
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
                    except:
                        pass

    remaining_tasks = [idx for idx in task_indices if idx not in completed_indices]
    print(f"[Process {process_id}] Completed: {len(task_indices) - len(remaining_tasks)}, Remaining: {len(remaining_tasks)}")
    if not remaining_tasks:
        print(f"[Process {process_id}] All tasks completed!")
        return

    # Load Executor Qwen model
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print(f"[Process {process_id}] Loading Executor Qwen model...")
    executor_tokenizer = AutoTokenizer.from_pretrained(EXECUTOR_QWEN_MODEL_NAME)
    executor_model = AutoModelForCausalLM.from_pretrained(
        EXECUTOR_QWEN_MODEL_NAME, device_map={"": executor_device_index},
    ).eval()
    print(f"[Process {process_id}] Executor loaded.")

    # Load Manager Qwen model if needed
    manager_model = None
    manager_tokenizer = None
    if MANAGER_MODEL == "qwen":
        is_vl_model = "VL" in MANAGER_QWEN_MODEL_NAME or "vl" in MANAGER_QWEN_MODEL_NAME.lower()
        if is_vl_model:
            manager_processor = AutoProcessor.from_pretrained(MANAGER_QWEN_MODEL_NAME, use_fast=False)
            manager_tokenizer = AutoTokenizer.from_pretrained(MANAGER_QWEN_MODEL_NAME)
            manager_model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": manager_device_map}
            if manager_max_memory:
                manager_model_kwargs["max_memory"] = manager_max_memory
            manager_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MANAGER_QWEN_MODEL_NAME, **manager_model_kwargs
            ).eval()
        else:
            manager_tokenizer = AutoTokenizer.from_pretrained(MANAGER_QWEN_MODEL_NAME)
            manager_model_kwargs = {"torch_dtype": torch.float16, "device_map": manager_device_map}
            if manager_max_memory:
                manager_model_kwargs["max_memory"] = manager_max_memory
            manager_model = AutoModelForCausalLM.from_pretrained(
                MANAGER_QWEN_MODEL_NAME, **manager_model_kwargs
            ).eval()
        print(f"[Process {process_id}] Manager Qwen loaded.")

    # Load SentenceTransformer model for action matching
    from sentence_transformers import SentenceTransformer
    print(f"[Process {process_id}] Loading SentenceTransformer ({SENT_TRANSFORMER_MODEL})...")
    sent_model = SentenceTransformer(SENT_TRANSFORMER_MODEL)
    print(f"[Process {process_id}] SentenceTransformer loaded.")

    # Setup Manager API client
    gpt_client = None
    gemini_client = None
    if MANAGER_MODEL == "gemini":
        from google import genai
        if GEMINI_API_MODE == "ksyun":
            api_key = KSYUN_GEMINI_CONFIG["api_key"]
            gemini_client = genai.Client(
                api_key=api_key,
                http_options={
                    "base_url": KSYUN_GEMINI_CONFIG["base_url"],
                    "api_version": KSYUN_GEMINI_CONFIG["api_version"],
                    "headers": {"Authorization": f"Bearer {api_key}"},
                },
            )
            print(f"[Process {process_id}] Gemini client initialized via Ksyun proxy (model={GPT_MODEL}).")
        else:
            # Official Google Gemini API (via environment variables)
            gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
            gemini_base_url = os.environ.get("GOOGLE_GEMINI_BASE_URL", "")
            if not gemini_api_key:
                print(f"[Process {process_id}] ERROR: GEMINI_API_KEY not set!")
                return
            client_kwargs = {"api_key": gemini_api_key}
            if gemini_base_url:
                client_kwargs["http_options"] = {"base_url": gemini_base_url}
            gemini_client = genai.Client(**client_kwargs)
            print(f"[Process {process_id}] Gemini client initialized via official API (model={GPT_MODEL}).")
    elif MANAGER_MODEL != "qwen":
        if API_TYPE == "azure":
            endpoint_config = AZURE_OPENAI_CONFIG["endpoints"].get(GPT_MODEL)
            if not endpoint_config:
                print(f"[Process {process_id}] ERROR: No Azure endpoint for {GPT_MODEL}!")
                return
            gpt_client = AzureOpenAI(
                api_key=AZURE_OPENAI_CONFIG["api_key"],
                api_version=endpoint_config["api_version"],
                azure_endpoint=endpoint_config["azure_endpoint"]
            )
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL", None)
            if not api_key:
                print(f"[Process {process_id}] ERROR: OPENAI_API_KEY not set!")
                return
            gpt_client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"[Process {process_id}] GPT client initialized ({API_TYPE}).")

    # Apply ScienceWorld-specific insights prompts (monkey-patch MemcompilerPrompts)
    # apply_scienceworld_insights_prompts()

    # Setup Memcompiler
    resolved_memcompiler_path = resolve_memcompiler_path(MEMCOMPILER_PATH, allow_fallback=not UPDATE_MEMORY)
    print(f"[Process {process_id}] Loading Memcompiler from: {resolved_memcompiler_path}")
    from mas.memory.mas_memory.memcompiler import Memcompiler
    from mas.utils import EmbeddingFunc
    from mas.memory.common import MASMessage

    if MANAGER_MODEL == "qwen":
        llm_model = QwenLLMWrapper(model=manager_model, tokenizer=manager_tokenizer)
    elif MANAGER_MODEL == "gemini":
        llm_model = GeminiLLMWrapper(client=gemini_client, model=GPT_MODEL)
    else:
        llm_model = GPTLLMWrapper(client=gpt_client, model=GPT_MODEL)

    embedding_func = EmbeddingFunc(model_type="sentence-transformers/all-MiniLM-L6-v2")
    global_config = {
        "working_dir": resolved_memcompiler_path,
        "hop": 1, "start_insights_threshold": 5,
        "rounds_per_insights": 5, "insights_point_num": 5,
    }

    if UPDATE_MEMORY:
        SafeMemcompiler = create_safe_memcompiler_class()
        memcompiler_mem = SafeMemcompiler(
            namespace="memcompiler", global_config=global_config,
            llm_model=llm_model, embedding_func=embedding_func,
        )
        print(f"[Process {process_id}] SafeMemcompiler loaded. Size: {memcompiler_mem.memory_size}")
    else:
        memcompiler_mem = Memcompiler(
            namespace="memcompiler", global_config=global_config,
            llm_model=llm_model, embedding_func=embedding_func,
        )
        print(f"[Process {process_id}] Memcompiler loaded (read-only). Size: {memcompiler_mem.memory_size}")

    # Initialize ScienceWorld environment
    print(f"[Process {process_id}] Initializing ScienceWorldEnvWrapper...")
    sw_env = ScienceWorldEnvWrapper(env_config={}, max_trials=MAX_STEPS)
    print(f"[Process {process_id}] ScienceWorldEnvWrapper initialized.")

    # Statistics
    stats = {'total': 0, 'success': 0, 'failed': 0, 'total_jvm_score': 0, 'total_progress_rate': 0.0}

    # Main loop
    for task_num, target_idx in enumerate(remaining_tasks):
        # try:
        task_config = all_task_configs[target_idx]
        task_name = task_config["task_name"]
        variation_idx = task_config["variation_idx"]
        print(f"\n[Process {process_id}] Task {task_num + 1}/{len(remaining_tasks)} "
                f"(idx={target_idx}, task={task_name}, var={variation_idx})")

        # Setup ScienceWorld environment for this task
        task_main, task_description = sw_env.set_env(task_config)
        task_text = task_main
        print(f"[Process {process_id}] Task: {task_text[:100]}...")

        # Token usage tracking
        task_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if MANAGER_MODEL != "qwen" and hasattr(llm_model, 'reset_token_counter'):
            llm_model.reset_token_counter()

        # Retrieve Task Memory
        successful_tasks, failed_tasks, insights = memcompiler_mem.retrieve_memory(
            query_task=task_text, successful_topk=1, failed_topk=0,
            insight_topk=5, threshold=0.0
        )
        runtime_key_steps = []
        for success_task in list(successful_tasks or [])[:1]:
            key_steps = generate_runtime_key_steps(success_task=success_task, llm_model=llm_model)
            runtime_key_steps.append(key_steps)

        task_memory = {
            "successful_tasks": successful_tasks, "failed_tasks": failed_tasks,
            "insights": insights, "runtime_key_steps": runtime_key_steps,
        }
        print(f"[Process {process_id}] Task Memory: {len(successful_tasks or [])} success, "
                f"{len(failed_tasks or [])} fail, {len(insights or [])} insights")

        # Initialize per-episode state
        history = []
        detailed_steps = []
        working_memory = ""
        step_count = 0
        episode_done = False
        episode_won = False
        failure_reason = None
        final_score = 0

        # Episode execution loop
        while not episode_done:
            step_count += 1
            if step_count > MAX_STEPS:
                print(f"[Process {process_id}] Max steps reached.")
                failure_reason = "max_steps_exceeded"
                break

            # Get environment info
            action_templates = sw_env.get_action_templates()
            available_objects = sw_env.get_available_objects()
            valid_actions = sw_env.get_valid_actions()
            current_obs = sw_env.get_observation()
            current_inventory = sw_env.get_inventory()
            current_score = sw_env.get_score()

            # Step 1: Call Manager
            if MANAGER_MODEL == "qwen":
                manager_raw_response, manager_input_fields = call_manager_qwen(
                    model=manager_model, tokenizer=manager_tokenizer,
                    working_memory=working_memory, task_text=task_text,
                    history=history, current_obs=current_obs,
                    current_score=current_score, task_memory=task_memory,
                    initial_observation=task_description,
                )
            elif MANAGER_MODEL == "gemini":
                print(11111111111111111111111111111111111111111)
                manager_raw_response, manager_input_fields, step_token_usage = call_gemini(
                    client=gemini_client, model=GPT_MODEL,
                    working_memory=working_memory, task_text=task_text,
                    history=history, current_obs=current_obs,
                    current_score=current_score, task_memory=task_memory,
                    initial_observation=task_description,
                )
                task_token_usage["prompt_tokens"] += step_token_usage["prompt_tokens"]
                task_token_usage["completion_tokens"] += step_token_usage["completion_tokens"]
                task_token_usage["total_tokens"] += step_token_usage["total_tokens"]
            else:
                manager_raw_response, manager_input_fields, step_token_usage = call_gpt(
                    client=gpt_client, model=GPT_MODEL,
                    working_memory=working_memory, task_text=task_text,
                    history=history, current_obs=current_obs,
                    current_score=current_score, task_memory=task_memory,
                    initial_observation=task_description,
                )
                task_token_usage["prompt_tokens"] += step_token_usage["prompt_tokens"]
                task_token_usage["completion_tokens"] += step_token_usage["completion_tokens"]
                task_token_usage["total_tokens"] += step_token_usage["total_tokens"]

            parsed_manager = parse_gpt_response(manager_raw_response)
            manager_type = parsed_manager.get("type", "NOACTION")

            # Step 2: Process Manager output
            manager_guidance = None

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

            if manager_type == "EMBODIED":
                manager_guidance = build_enhanced_guidance(parsed_manager)
            elif manager_type == "CONTEXT":
                working_memory = apply_memory_operation(working_memory, parsed_manager)
            elif manager_type == "HYBRID":
                manager_guidance = build_enhanced_guidance(parsed_manager)
                working_memory = apply_memory_operation(working_memory, parsed_manager)

            # Step 3: Call Executor
            raw_action, qwen_raw_response, qwen_input_fields = qwen_choose_action(
                model=executor_model, tokenizer=executor_tokenizer,
                working_memory=working_memory, task_text=task_text,
                action_templates=action_templates, available_objects=available_objects,
                history=history, current_obs=current_obs,
                current_inventory=current_inventory,
                gpt_guidance=manager_guidance, max_history_turns=12,
            )

            # Step 3.5: Action matching with SentenceTransformer
            matched_action, match_score, top5 = match_action_scienceworld(
                raw_action=raw_action,
                valid_actions=valid_actions,
                sent_transformer_model=sent_model,
                threshold=ACTION_MATCH_THRESHOLD,
            )
            print(f"[SW Match] raw='{raw_action}' → matched='{matched_action}' (score={match_score:.3f})")
            if top5:
                print(f"[SW Match] top5: {[(a, f'{s:.3f}') for a, s in top5[:3]]}")

            final_action = matched_action

            # Step 4: Execute action in ScienceWorld environment
            observation, reward, done = sw_env.step(final_action)
            history.append({"action": final_action, "observation": observation})

            # Record detailed step
            step_record = {
                "step": step_count,
                "manager": {
                    "model": MANAGER_MODEL,
                    "input": manager_input_fields,
                    "output": {"raw": manager_raw_response}
                },
                "executor": {
                    "input": qwen_input_fields,
                    "output": {"raw": qwen_raw_response, "parsed_action": raw_action, "matched_action": final_action}
                },
                "match_info": {
                    "score": match_score,
                    "top5": top5,
                },
                "env": {
                    "action": final_action, "observation": observation,
                    "reward": reward, "done": done,
                    "jvm_score": sw_env.get_score(),
                }
            }
            detailed_steps.append(step_record)

            # Step 5: Check if done
            final_jvm_score = sw_env.get_score()
            if done or final_jvm_score >= 100:
                episode_done = True
                episode_won = (final_jvm_score >= 100)
                print(f"[Process {process_id}] Episode finished! Won: {episode_won}, "
                        f"JVM Score: {final_jvm_score}, Steps: {len(history)}")

        # Episode ended - get progress_rate via feedback()
        progress_rate, subgoal_done, feedback_msg = sw_env.feedback()
        final_jvm_score = sw_env.get_score()
        if final_jvm_score >= 100:   # progress_rate 存在 subgoal 质量问题——有些JVM score=100 的任务，其 progress_rate 可能不到 1.0，所以最后强制赋值为1.0
            progress_rate = 1.0
        print(f"[Process {process_id}] Feedback: progress_rate={progress_rate:.2f}, "
                f"subgoal_done={subgoal_done}, jvm_score={final_jvm_score}")

        # Update stats
        stats['total'] += 1
        stats['total_jvm_score'] += final_jvm_score
        stats['total_progress_rate'] += progress_rate
        if episode_won:
            stats['success'] += 1
        else:
            stats['failed'] += 1

        # Save trajectory
        traj_file = os.path.join(traj_path, f"{target_idx}.json")
        out_traj = {
            "env_index": target_idx, "task_name": task_name,
            "variation_idx": variation_idx, "task": task_text,
            "won": episode_won, "jvm_score": final_jvm_score,
            "progress_rate": progress_rate,
            "failure_reason": failure_reason,
            "steps": len(history), "trajectory": history
        }
        with open(traj_file, "w") as f:
            json.dump(out_traj, f, indent=4)

        # Save detailed trajectory
        detailed_file = os.path.join(detailed_traj_path, f"{target_idx}.json")
        detailed_out = {
            "env_index": target_idx, "task_name": task_name,
            "variation_idx": variation_idx, "task": task_text,
            "won": episode_won, "jvm_score": final_jvm_score,
            "progress_rate": progress_rate,
            "failure_reason": failure_reason,
            "task_memory_summary": f"{len(successful_tasks or [])} success, {len(failed_tasks or [])} fail, {len(insights or [])} insights",
            "steps": detailed_steps
        }
        with open(detailed_file, "w") as f:
            json.dump(detailed_out, f, indent=4)

        # Append to overview
        out_overview = {
            "env_index": target_idx, "task_name": task_name,
            "variation_idx": variation_idx, "task": task_text,
            "won": episode_won, "jvm_score": final_jvm_score,
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
                    task_main=task_text, task_description=task_description,
                    label=episode_won,
                )
                mas_message.add_extra_field("task_name", task_name)
                mas_message.add_extra_field("variation_idx", variation_idx)
                mas_message.add_extra_field("jvm_score", final_jvm_score)
                mas_message.add_extra_field("progress_rate", progress_rate)
                for h in history:
                    mas_message.move_state(h["action"], h["observation"])
                memcompiler_mem.add_memory(mas_message)
                print(f"[Process {process_id}] Added task to Memcompiler. Size: {memcompiler_mem.memory_size}")
            except Exception as e:
                print(f"[Process {process_id}] Failed to add memory: {e}")

        # Print progress
        acc = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        avg_jvm_score = stats['total_jvm_score'] / stats['total'] if stats['total'] > 0 else 0
        avg_progress = stats['total_progress_rate'] / stats['total'] if stats['total'] > 0 else 0
        print(f"[Process {process_id}] Progress: {stats['total']}/{len(remaining_tasks)} | "
                f"Success: {stats['success']} | Acc: {acc:.2%} | "
                f"Avg Progress Rate: {avg_progress:.2%} | Avg JVM Score: {avg_jvm_score:.1f}")

        # Token usage summary
        memcompiler_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if MANAGER_MODEL != "qwen" and hasattr(llm_model, 'get_accumulated_tokens'):
            memcompiler_tokens = llm_model.get_accumulated_tokens()
        total_prompt = task_token_usage['prompt_tokens'] + memcompiler_tokens['prompt_tokens']
        total_completion = task_token_usage['completion_tokens'] + memcompiler_tokens['completion_tokens']
        total_all = task_token_usage['total_tokens'] + memcompiler_tokens['total_tokens']
        print(f"[Process {process_id}] Task {target_idx} Token Usage: prompt={total_prompt}, completion={total_completion}, total={total_all}")

        # except Exception as e:
        #     print(f"[Process {process_id}] Error on task {target_idx}: {e}")
        #     traceback.print_exc()
        #     stats['total'] += 1
        #     stats['failed'] += 1

    # Cleanup: close the ScienceWorld JVM
    sw_env.close()
    print(f"[Process {process_id}] ScienceWorld environment closed.")

    # Final summary
    print(f"\n[Process {process_id}] COMPLETED")
    print(f"[Process {process_id}] Total: {stats['total']} | Success: {stats['success']} | Failed: {stats['failed']}")
    if stats['total'] > 0:
        avg_jvm = stats['total_jvm_score'] / stats['total']
        avg_pr = stats['total_progress_rate'] / stats['total']
        print(f"[Process {process_id}] Avg JVM Score: {avg_jvm:.1f} | Avg Progress Rate: {avg_pr:.2%}")


# =============================================================================
# Main Function (adapted for ScienceWorld task loading)
# =============================================================================
def main():
    print("=" * 60)
    print("ScienceWorld Manager-Executor Multi-Process")
    print("=" * 60)
    print(f"NUM_PROCESSES: {NUM_PROCESSES}")
    print(f"NUM_GPUS: {NUM_GPUS}")
    print(f"OUTPUT_PATH: {OUTPUT_PATH}")
    print(f"MEMCOMPILER_PATH: {MEMCOMPILER_PATH}")
    print(f"UPDATE_MEMORY: {UPDATE_MEMORY}")
    print(f"TASK_SPLIT: {TASK_SPLIT}")
    print(f"TASK_NAMES: {TASK_NAMES or 'All 30 tasks'}")
    print(f"SIMPLIFICATION: {SIMPLIFICATION_STR or 'Auto (easy / reduced for electrical)'}")
    print(f"MAX_STEPS: {MAX_STEPS}")
    print(f"SENT_TRANSFORMER: {SENT_TRANSFORMER_MODEL}")

    # Setup directories
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "trajectory"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "detailed_trajectory"), exist_ok=True)

    # Load ScienceWorld task configs
    if TASK_LOAD_MODE == "jsonl":
        print(f"\nLoading ScienceWorld task configs from JSONL: {TEST_JSONL_PATH}")
        all_task_configs = load_scienceworld_tasks_from_jsonl(TEST_JSONL_PATH)
    else:
        print(f"\nLoading ScienceWorld task configs from env (split={TASK_SPLIT})...")
        all_task_configs = get_scienceworld_task_configs(
            task_names=TASK_NAMES,
            split=TASK_SPLIT,
            simplification_str=SIMPLIFICATION_STR,
        )
    total_tasks = len(all_task_configs)
    print(f"Total ScienceWorld tasks: {total_tasks}")
    for tc in all_task_configs[:5]:
        print(f"  Sample: task={tc['task_name']}, var={tc['variation_idx']}, simp={tc['simplification_str'][:30]}...")

    # All task indices
    target_indices = list(range(total_tasks))

    # =========================================================================
    # Mode A: Run uncompleted tasks (original logic)
    # =========================================================================
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
    uncompleted_indices = sorted([idx for idx in target_indices if idx not in completed_indices])
    print(f"Uncompleted tasks to run: {len(uncompleted_indices)}")

    # =========================================================================
    # Mode B: Retry failed tasks (reuse memory from previous epoch)
    # =========================================================================
    # overview_path = os.path.join(OUTPUT_PATH, "overview.jsonl")
    # failed_indices = set()
    # kept_lines = []
    # if os.path.exists(overview_path):
    #     print(f"Checking failed tasks from: {overview_path}")
    #     with open(overview_path, "r") as f:
    #         for line in f:
    #             line = line.strip()
    #             if not line:
    #                 continue
    #             try:
    #                 record = json.loads(line)
    #                 env_index = record.get("env_index")
    #                 won = record.get("won", False)
    #                 if env_index is not None and not won:
    #                     failed_indices.add(env_index)
    #                 else:
    #                     kept_lines.append(line)
    #             except json.JSONDecodeError:
    #                 kept_lines.append(line)
    #                 continue
    #     print(f"Failed tasks to retry: {len(failed_indices)}")

    #     # Purge failed task records from overview.jsonl (keep only successful ones)
    #     with FileLock(overview_path + ".lock"):
    #         with open(overview_path, "w") as f:
    #             for kl in kept_lines:
    #                 f.write(kl + "\n")
    #     print(f"Purged {len(failed_indices)} failed records from overview.jsonl, "
    #           f"kept {len(kept_lines)} successful records.")
    # else:
    #     print(f"No overview file found at {overview_path}, cannot determine failed tasks.")
    #     return

    # uncompleted_indices = sorted([idx for idx in target_indices if idx in failed_indices])
    # print(f"Tasks to retry (failed): {len(uncompleted_indices)}")
    # =========================================================================
    if len(uncompleted_indices) == 0:
        print("All tasks already completed! Exiting.")
        return

    # Static task pre-allocation
    task_assignments = [[] for _ in range(NUM_PROCESSES)]
    for i, idx in enumerate(uncompleted_indices):
        task_assignments[i % NUM_PROCESSES].append(idx)

    for i, tasks in enumerate(task_assignments):
        print(f"Process {i}: {len(tasks)} tasks")

    gpu_assignments = GPU_ASSIGNMENTS

    # Start worker processes
    print("\n" + "=" * 60)
    print("Starting worker processes...")
    print("=" * 60)

    processes = []
    for i in range(NUM_PROCESSES):
        p = mp.Process(
            target=worker_process,
            args=(i, gpu_assignments[i], task_assignments[i], all_task_configs)
        )
        p.start()
        processes.append(p)
        print(f"Started process {i} on GPU {gpu_assignments[i]}")

    for p in processes:
        p.join()

    print("\n" + "=" * 60)
    print("ALL PROCESSES COMPLETED")
    print("=" * 60)

    # Final summary
    overview_file = os.path.join(OUTPUT_PATH, "overview.jsonl")
    if os.path.exists(overview_file):
        total = 0
        success = 0
        total_jvm_score = 0
        total_progress_rate = 0.0
        with open(overview_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    total += 1
                    if data.get("won"):
                        success += 1
                    total_jvm_score += data.get("jvm_score", data.get("score", 0))
                    total_progress_rate += data.get("progress_rate", 0.0)
                except:
                    pass
        if total > 0:
            print(f"Total: {total} | Success: {success} | Acc: {success/total:.2%} | "
                  f"Avg Progress Rate: {total_progress_rate/total:.2%} | Avg JVM Score: {total_jvm_score/total:.1f}")
        else:
            print("No results")

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
