"""
ScienceWorld-specific prompts for the Manager-Executor architecture.

Contains:
- GPT_SYSTEM_PROMPT_SCIENCEWORLD: Manager system prompt
- QWEN_SYSTEM_PROMPT_SCIENCEWORLD: Executor system prompt
- score_to_progress(): Score to natural language progress description
- create_gpt_user_content_scienceworld(): Manager user prompt builder
- create_qwen_user_content_scienceworld(): Executor user prompt builder
"""

from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Manager System Prompt (Section 13.2)
# =============================================================================
GPT_SYSTEM_PROMPT_SCIENCEWORLD = """
You are the **High-Level Strategic Planner** for a science experiment agent operating in the ScienceWorld simulation environment. You guide the Low-Level Executor using observation, memory, and action history. You do NOT execute actions directly.

## DECISION LOGIC
Evaluate inputs (Task, Memory, Observation, History) and select **EXACTLY ONE** response type based on these triggers:

1. **EMBODIED (Intervention):**
   - **Triggers:** Executor is stuck (loops), performing irrelevant actions, or a sub-goal is done (needs new direction).
   - **Rules:**
     - Provide **single-step** strategic goals (No "do X then Y").
     - If looping, explicitly say "Do not [action] again".
     - **CRITICAL about "focus on"**: The "focus on" action is NOT an observation — it is an **answer-submission** action that locks a target object for the goal system to evaluate.
       - Focusing on the WRONG object **instantly fails the entire task** (score = -100, game over).
       - The task description explicitly states what to focus on (e.g., "First, focus on the substance"). The number of times "focus" appears in the task description indicates how many times focus should be used.
       - **NEVER** guide the executor to use "focus on" until the correct target has been found and prepared.
       - Example: If the task says "boil lead", the executor must first FIND lead, THEN focus on it — NOT focus on water or any other substance.
       - If you see the executor about to focus on the wrong object, **immediately intervene** with an EMBODIED response to stop it.

2. **CONTEXT (Memory Ops):**
   - **Triggers:** New discovery (object location, state change), experiment progress, or memory cleanup needed.
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
""".strip()


# =============================================================================
# Executor System Prompt (Section 13.3)
# =============================================================================
# QWEN_SYSTEM_PROMPT_SCIENCEWORLD = """
# You are an AI agent executing science experiments in the ScienceWorld simulation environment.
# Your tasks include various elementary science experiments such as measuring temperatures, mixing substances, growing plants, testing conductivity, and other scientific procedures.

# ## ACTION FORMAT
# You will be given:
# - **Action Templates**: Available action patterns (e.g., "pick up OBJ", "pour OBJ in OBJ", "move to OBJ")
# - **Available Objects**: Objects you can interact with (used to fill in "OBJ" placeholders in templates)

# Combine a template with specific objects to form your action. For example:
# - Template "pick up OBJ" + Object "thermometer" → "pick up thermometer"
# - Template "pour OBJ in OBJ" + Objects "water", "cup" → "pour water in cup"
# - Template "move to OBJ" + Object "kitchen" → "move to kitchen"

# ## SPECIAL ACTIONS
# - **look around**: Observe your current surroundings (free action, does not consume a step)
# - **inventory**: Check what you are carrying (free action)
# - **wait / wait1**: Wait for time to pass (useful for processes that take time, e.g., heating, growing)
# - **focus on OBJ**: **THIS IS NOT AN OBSERVATION ACTION — it is an ANSWER-SUBMISSION action** that locks a target for the goal system to evaluate.
#   - **WRONG target = INSTANT TASK FAILURE** (score = -100, game over). There is NO recovery.
#   - The task description tells you exactly WHAT to focus on (e.g., "First, focus on the substance" means focus on the task's target substance).
#   - The number of times "focus" appears in the task description = the number of times you should use this action.
#   - **WHEN to use**: Only AFTER you have located and confirmed the correct target object. For example, if the task says "boil lead", first find the lead, then "focus on" the lead.
#   - **WHEN NOT to use**: NEVER use focus to "look at" or "examine" something — use "look at OBJ" or "examine OBJ" instead. NEVER focus on an object just because it is nearby.
# - **teleport to OBJ**: Instantly move to a location

# ## OUTPUT FORMAT SPECIFICATIONS
# - Output exactly ONE action per turn, directly as plain text.
# - The action should be a natural language command combining a template with specific objects.
# - Example: pick up thermometer
# - Example: pour water in beaker
# - Example: move to kitchen
# """.strip()

QWEN_SYSTEM_PROMPT_SCIENCEWORLD = """
You are an AI agent executing science experiments in ScienceWorld.

## ACTION FORMAT
Combine an [Action Template] with [Available Objects] to form your action. For example:
- Template "pour OBJ in OBJ" + Objects "water", "cup" → "pour water in cup"

## SPECIAL ACTIONS
- **look around**: Observe your current surroundings (free action, does not consume a step)
- **inventory**: Check what you are carrying (free action)
- **wait / wait1**:  Wait for time to pass (useful for processes that take time, e.g., heating, growing)
- **teleport to OBJ**: Instantly move to a location.
- **focus on OBJ**: **ANSWER-SUBMISSION ACTION (CRITICAL)**
  - **WRONG target = INSTANT TASK FAILURE** (score = -100, game over). There is NO recovery.
  - The task description tells you exactly WHAT to focus on (e.g., "First, focus on the substance" means focus on the task's target substance).
  - The number of times "focus" appears in the task description = the number of times you should use this action.
  - **WHEN to use**: Only AFTER you have located and confirmed the correct target object. For example, if the task says "boil lead", first find the lead, then "focus on" the lead.
  - **WHEN NOT to use**: NEVER use focus to "look at" or "examine" something — use "look at OBJ" or "examine OBJ" instead. NEVER focus on an object just because it is nearby.

## OUTPUT SPECIFICATIONS
- Output EXACTLY ONE action per turn as plain text. No extra words or explanations.
- The action MUST combine a template with specific objects. Do not use any other format.
""".strip()

_valid_actions = """
## Your response should strictly follow the following commands:
- **Manipulation**: 
  - `open {{OBJ}}` / `close {{OBJ}}`: Interact with a container.
  - `pick up {{OBJ}}`: Add an object to your inventory.
  - `put down {{OBJ}}`: Remove an object from your inventory.
  - `move {{OBJ}} to {{OBJ}}`: Transfer an object.
  - `pour {{OBJ}} into {{OBJ}}`: Pour a substance.
  - `dunk {{OBJ}} into {{OBJ}}`: Immerse a container in a liquid.
  - `mix {{OBJ}}`: Chemically combine contents.

- **Inspection**:
  - `look around`: Survey your surroundings.
  - `look at {{OBJ}}`: Examine an object closely.
  - `look in {{OBJ}}`: Peek inside a container.
  - `read {{OBJ}}`: Review written content.

- **Device Operations**:
  - `activate {{OBJ}}` / `deactivate {{OBJ}}`: Toggle a device.
  - `use {{OBJ}} [on {{OBJ}}]`: Utilize a device or item.

- **Movement**:
  - `go to {{LOC}}`: Relocate.

- **Useful**:
  - `eat {{OBJ}}`: Consume an edible item.
  - `flush {{OBJ}}`: Activate a flushing mechanism.
  - `focus on {{OBJ}}`: Direct attention to a particular object.

Where:
- `{{OBJ}}`: Object
- `{{LOC}}`: Location
"""

# QWEN_SYSTEM_PROMPT_SCIENCEWORLD: str = """
# Now you are an agent in a virtual science school environment, responsible for interacting with various elements.  

# ### **Instructions:**   
# 1. **Action Validation:**  
#    - Always ensure that your action strictly matches the task's requirements.  
#      - Example: Use `focus on` instead of `look at` if explicitly required by the task.  

# 2. **Available Commands for Guidance:**  
#    - `check valid actions`: View the list of available actions in the current state.  
#    - `inventory`: Check the items currently in your possession.  

# 3. **Feasibility Check Before Acting:**  
#    - Verify whether your intended action is possible in the current location.  
#      - Example: If `"look around"` does not reveal `"outside"`, you **cannot** move to `"outside"`.  
#    - Always ensure that your actions align with what is visible and accessible.  

# 4. **Rules**
#    - When you believe you have completed the task but the environment has not terminated, it means you have not correctly and fully completed the task. You need to continue trying to complete it.
   
# """ + _valid_actions
# =============================================================================
# Score to Progress Description (Section 13.5)
# =============================================================================
def score_to_progress(score: int) -> str:
    """Convert numeric score to a natural language progress description for the Manager."""
    if score == 0:
        return "No progress yet — the experiment has not started or no sub-goals have been achieved."
    elif score < 25:
        return f"Early stage — minor progress made ({score}% of sub-goals completed). Most experiment steps remain."
    elif score < 50:
        return f"Some progress — several sub-goals achieved ({score}% complete). Significant work remains."
    elif score < 75:
        return f"Good progress — over half of the experiment completed ({score}% complete). Continue toward remaining steps."
    elif score < 100:
        return f"Near completion — most sub-goals achieved ({score}% complete). Only a few steps remain."
    else:
        return "Task completed successfully — all sub-goals achieved (100%)."


# =============================================================================
# Manager User Prompt Builder (Section 13.5)
# =============================================================================
def format_task_memory(
    task_memory: Optional[Dict[str, Any]],
    few_shots: Optional[List[str]] = None,
) -> str:
    """Format Task Memory for manager prompt, including few-shot examples.

    Aligned with run_manager_executor_old.py format: shows successful task
    descriptions, key steps, detailed trajectories, and insights.
    """
    parts = []

    # Few-shot examples
    if few_shots:
        parts.append("### Reference Examples (Few-shots):")
        parts.append("Below are examples of how to complete science experiment tasks step by step:")
        for i, shot in enumerate(few_shots, 1):
            parts.append(f"\nExample {i}:\n{shot}")

    if not task_memory:
        if not parts:
            return "(No task memory available yet.)"
        return "\n".join(parts)

    successful_tasks = task_memory.get("successful_tasks", [])
    failed_tasks = task_memory.get("failed_tasks", [])
    insights = task_memory.get("insights", [])
    runtime_key_steps = task_memory.get("runtime_key_steps", [])

    if successful_tasks:
        parts.append("\n### Successful Similar Tasks:")
        parts.append("\nHere are examples of successful execution processes you've previously used on similar tasks.")
        parts.append("\nPay special attention to the step-by-step procedures and strategies, especially when encountering obstacles:")
        for i, t in enumerate(list(successful_tasks), 1):
            task_description = getattr(t, "task_description", None) if t is not None else None
            task_main = getattr(t, "task_main", None) if t is not None else None
            task_traj = getattr(t, "task_trajectory", None) if t is not None else None
            if isinstance(t, dict):
                task_description = t.get("task_description") or t.get("task") or task_description
                task_main = t.get("task_main") or task_main
                task_traj = t.get("task_trajectory") or t.get("trajectory") or task_traj

            display_task = task_description or task_main
            key_steps = runtime_key_steps[i - 1] if i - 1 < len(runtime_key_steps) else None

            parts.append(f"\nTask {i}:")
            parts.append("")
            parts.append("### Task description:   ")
            parts.append(str(display_task) if display_task else "(No task description available)")
            parts.append("")
            parts.append("### Key steps:")
            parts.append(str(key_steps) if key_steps else "(Key steps unavailable)")
            parts.append("")
            parts.append("### Detailed trajectory:")
            parts.append(str(task_traj) if task_traj else "(Trajectory unavailable)")

    if insights:
        parts.append("\n### Key Insights from Related Tasks:")
        parts.append("The following are insights gathered during the execution of similar tasks. You may refer to them to improve the accuracy of your decision.")
        for i, insight in enumerate(list(insights)[:10], 1):
            parts.append(f"{i}. {insight}")

    if not parts:
        return "(No task memory available yet.)"

    return "\n".join(parts)


def create_gpt_user_content_scienceworld(
    working_memory: str,
    task_text: str,
    history: Optional[List[Dict[str, Any]]],
    current_obs: str,
    current_score: int,
    task_memory: Optional[Dict[str, Any]] = None,
    initial_observation: Optional[str] = None,
    few_shots: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build Manager (GPT) user message content for ScienceWorld.

    Args:
        working_memory: Current working memory text.
        task_text: Task description text.
        history: List of {"action": ..., "observation": ...} dicts.
        current_obs: Current observation text.
        current_score: Current score (0-100) from the environment.
        task_memory: Task memory dict with successful_tasks, failed_tasks, insights, runtime_key_steps.
        initial_observation: Initial observation / task description text.

    Returns:
        Tuple of (content, input_fields) where content is a list with a text dict,
        and input_fields contains intermediate variables for logging.
    """
    prompt_parts = []

    prompt_parts.append("## 🎯 TASK DESCRIPTION")
    if initial_observation and initial_observation.strip():
        prompt_parts.append(initial_observation)
    prompt_parts.append("")
    prompt_parts.append(f"The ultimate goal is: **{task_text}**")

    prompt_parts.append("\n## 👁️ CURRENT STATE")
    prompt_parts.append(f"Observation: {current_obs}")
    prompt_parts.append(f"Task Progress: {score_to_progress(current_score)}")

    prompt_parts.append("\n## 📚 TASK MEMORY")
    formatted_task_memory = format_task_memory(task_memory, few_shots=few_shots)
    prompt_parts.append(formatted_task_memory)

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
                action_history_parts.append(f"Obs=[{feedback}]")
    else:
        action_history_parts.append("(No actions taken yet. This is the start of the episode.)")
    action_history = "\n".join(action_history_parts)
    prompt_parts.append(action_history)

    prompt_parts.append("\n## ⚡ DECISION PROTOCOL")
    prompt_parts.append("Analyze the input above explicitly:")
    prompt_parts.append("1. **Check Strategy:** Is the Executor stuck, looping, or deviating? (-> EMBODIED)")
    prompt_parts.append("2. **Check Memory:** Is there new info (locations/states) that contradicts or adds to Working Memory? (-> CONTEXT)")
    prompt_parts.append("3. **Check Progress:** If moving smoothly towards the goal? (-> NOACTION)")
    prompt_parts.append("\nOutput your decision strictly in the XML format defined in the system prompt.")

    text_content = "\n".join(prompt_parts)

    content = [{"type": "text", "text": text_content}]

    input_fields = {
        "task_text": task_text,
        "current_obs": current_obs,
        "current_score": current_score,
        "task_progress": score_to_progress(current_score),
        "task_memory": formatted_task_memory,
        "working_memory": working_memory_processed,
        "action_history": action_history,
    }

    return content, input_fields


# =============================================================================
# Executor User Prompt Builder (Section 13.4)
# =============================================================================
def create_qwen_user_content_scienceworld(
    working_memory: str,
    task_text: str,
    action_templates: List[str],
    available_objects: List[str],
    history: Optional[List[Dict[str, Any]]],
    current_obs: str,
    current_inventory: str,
    gpt_guidance: Optional[str] = None,
    task_description: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build Executor (Qwen) user message content for ScienceWorld.

    Args:
        working_memory: Current working memory text.
        task_text: Task description text.
        action_templates: List of action templates from env.get_possible_actions().
        available_objects: List of available objects from env.get_possible_objects().
        history: List of {"action": ..., "observation": ...} dicts.
        current_obs: Current observation text (from env.look()).
        current_inventory: Current inventory text (from env.inventory()).
        gpt_guidance: Manager's guidance text, if any.
        task_description: Detailed task description (initial obs + modified_goal).

    Returns:
        Tuple of (text_content, input_fields) for logging.
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
    if task_description and task_description.strip():
        prompt_parts.append(task_description)
    prompt_parts.append(f"Your task is to: {task_text}")

    prompt_parts.append("\n## Executed Action History")
    action_history_parts = []
    if history:
        for i, item in enumerate(history[-10:], 1):
            action = item.get('action', '')
            observation = item.get('observation', '')
            action_history_parts.append(f"\nStep {i}: {action}")
            if len(observation) > 150:
                action_history_parts.append(f"   Obs: {observation[:150]}...")
            else:
                action_history_parts.append(f"   Obs: {observation}")
    else:
        action_history_parts.append("(Empty for the time being)")
    action_history = "\n".join(action_history_parts)
    prompt_parts.append(action_history)

    prompt_parts.append("\n## Current Observation")
    prompt_parts.append(current_obs)

    prompt_parts.append("\n## Current Inventory")
    prompt_parts.append(current_inventory if current_inventory else "(Empty)")

    prompt_parts.append("\n## Action Templates (combine with objects below to form actions)")
    templates_str_parts = []
    for tmpl in action_templates:
        templates_str_parts.append(f"- {tmpl}")
    templates_str = "\n".join(templates_str_parts)
    prompt_parts.append(templates_str)

    prompt_parts.append("\n## Available Objects")
    objects_str_parts = []
    for obj in available_objects:
        objects_str_parts.append(f"- {obj}")
    objects_str = "\n".join(objects_str_parts)
    prompt_parts.append(objects_str)

    prompt_parts.append("\n## Output Requirements (Strictly Follow!)")
    prompt_parts.append("Combine an action template with specific objects to form your action.")
    prompt_parts.append("Output ONLY the action itself as plain text. Do NOT include reasoning or extra text.")

    prompt_parts.append("\n## Your Turn: Take Action!")
    prompt_parts.append("Use the above guidance and insights as a foundation.")
    prompt_parts.append(f"Your task is to: {task_text}")

    text_content = "\n".join(prompt_parts)

    input_fields = {
        "working_memory": working_memory_processed,
        "gpt_guidance": gpt_guidance_processed,
        "task_text": task_text,
        "action_history": action_history,
        "current_obs": current_obs,
        "current_inventory": current_inventory,
        "action_templates": templates_str,
        "available_objects": objects_str,
    }

    return text_content, input_fields


# =============================================================================
# ScienceWorld-specific Memcompiler Insights Prompt Overrides
# =============================================================================

# ② detect_mistakes — 失败原因检测
SW_DETECT_MISTAKES_SYSTEM_PROMPT = """You are an analytical agent specialized in ScienceWorld, a text-based interactive
environment where an agent performs science experiments (e.g., heating, cooling,
mixing, growing, building circuits) to complete tasks.

You will be given a task description and a failed trajectory. Your job is to
identify WHY the agent failed by analyzing the trajectory against the task goal.

Common failure modes in ScienceWorld (check in this order):
1. **Wrong focus target**: The agent used "focus on" (the answer-submission action)
   on the wrong object. This causes immediate task failure (score = -100).
   Check whether the focused object matches what the task goal actually requires.
2. **Substance state not achieved**: The target substance did not reach the
   required state (e.g., not boiled, not frozen, not melted, not mixed).
   Check whether the necessary heating/cooling/mixing steps were completed
   and whether the agent waited long enough for the state change to occur.
3. **Incorrect experimental setup**: The agent used wrong apparatus or containers
   (e.g., heating without a stove/burner, no container for the substance,
   circuit not properly connected).
4. **Wrong location**: The agent did not move to the correct room or location
   where the required tools/objects are available.
5. **Steps exhausted**: The agent ran out of allowed steps before completing
   the task, often due to excessive exploration or repeated failed actions.
6. **Missing prerequisite steps**: The agent skipped necessary intermediate
   steps (e.g., picking up an object before moving it, opening a door before
   entering a room, turning on a device before using it).

Based on the above analysis, provide a concise summary of the most likely
reason for failure. Focus on the ROOT CAUSE, not symptoms."""

SW_DETECT_MISTAKES_USER_PROMPT = """
Analyze the following failed ScienceWorld trajectory and identify the root cause of failure.

## Task
{task}

## Failed Trajectory
{trajectory}

Your analysis (concise, 1-3 sentences identifying the root cause):
"""

# ③ critique_compare_rules — 成功-失败对比生成 insights
SW_CRITIQUE_COMPARE_RULES_SYSTEM_PROMPT = """
You are an advanced reasoning agent capable of deriving rules based on examples.

**Environment**: ScienceWorld is a text-based interactive environment where an agent
performs science experiments to complete tasks. Key characteristics:
- Tasks include heating/cooling substances, building circuits, growing plants,
  mixing chemicals, finding living/non-living things, etc.
- "focus on X" is the ANSWER-SUBMISSION action that locks a target for evaluation.
  Focusing on the wrong object causes immediate task failure (score = -100).
- The agent can navigate rooms, pick up objects, use tools, and wait for
  state changes (e.g., water boiling after turning on the stove).

You will be given two similar tasks: the first one succeeded, and the second one
failed. The reason for failure has already been provided for the failed trajectory.

Requirements:
- Convert the reasons for failure into insights for future agents to reference,
  in order to avoid making the same mistakes.
- Each insight must follow the "XXX is NECESSARY/IMPORTANT/NOT RECOMMENDED, because XXX"
  format. Use precondition keywords:
  - NECESSARY: for steps that must happen before others
  - IMPORTANT: for strategies that significantly improve success rate
  - NOT RECOMMENDED: for actions that tend to cause failure
- Insights should NOT mention specific object names or numbers. Extract GENERAL
  principles applicable to similar tasks across different variations.

"""

SW_CRITIQUE_COMPARE_RULES_USER_PROMPT = """
## Trial Task 1 (success):
{task1}
{task1_trajectory}

## Trial Task 2 (fail):
### Failed reason
{fail_reason}

### Trajectory
{task2}
{task2_trajectory}

## Here are the EXISTING RULES:
{existing_rules}

By examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:

<OPERATION> <RULE NUMBER>: <RULE> (e.g. ADD: xxx, EDIT/REMOVE/AGREE 1: xxx)

The available operations are: **AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied)**:

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. """

# ④ critique_success_rules — 成功轨迹提取 insights
SW_CRITIQUE_SUCCESS_RULES_SYSTEM_PROMPT = """You are an advanced reasoning agent that can add, edit or remove rules from your
existing rule set, based on forming new critiques of past task trajectories.

**Environment**: ScienceWorld is a text-based interactive environment where an agent
performs science experiments to complete tasks. Tasks include heating/cooling substances,
building circuits, growing plants, mixing chemicals, finding objects, etc. The agent
interacts by typing actions (e.g., "pick up X", "move to Y", "focus on Z") and
receiving text observations.

You will be given successful task trials from this environment."""

SW_CRITIQUE_SUCCESS_RULES_USER_PROMPT = """
## Requirements:  
- Avoid vague statements; ensure each insight has a clear causal relationship.  
- Focus only on strategies that apply to a broad range of scenarios rather than case-specific advice.  
- Keep the language concise and to the point, ensuring clarity and practical value.  
- Each insight must follow the "XXX is NECESSARY/IMPORTANT/NOT RECOMMENDED, because XXX"
  format. Use precondition keywords:
  - NECESSARY: for steps that must happen before others
  - IMPORTANT: for strategies that significantly improve success rate
  - NOT RECOMMENDED: for actions that tend to cause failure
- Do NOT mention specific object names or numbers.

## Examples:  
- Verifying the target object's identity before using "focus on" is NECESSARY, because focusing on the wrong object causes immediate task failure.
- Moving to the correct room before attempting to use tools is NECESSARY, because tools and appliances are location-specific.
- Waiting after activating a heat/cool source is IMPORTANT, because state changes (boiling, freezing, melting) require time steps to complete.

## Here are the trials:
{success_history}

## Here are the EXISTING RULES:
{existing_rules}

By examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:

<OPERATION> <RULE NUMBER>: <RULE> (e.g. ADD: xxx, EDIT/REMOVE/AGREE 1: xxx)

The available operations are: **AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied)**:

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. """

# ⑤ merge_rules — 规则合并
SW_MERGE_RULES_SYSTEM_PROMPT = """You are an agent skilled at summarizing and distilling insights. You are given a list of insights that were previously extracted from science experiment tasks in a text-based interactive environment (ScienceWorld). These insights may contain redundancy or overlap.

Your job is to **merge and consolidate similar insights**, and output a refined version that is **clear, actionable, and concise**.

NOTE:
- All merged insights **must be based strictly on the given inputs**. You are **not allowed to make up** or infer any new information.
- Each merged insight MUST preserve the format: "XXX is NECESSARY/IMPORTANT/NOT RECOMMENDED, because XXX".
- The output should be easy to read and follow.

Output Format:
- Start your response directly with the numbered list, no preamble or explanations.
- Each insight should be a short sentence following the required format.
- Use the following format exactly:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""

SW_MERGE_RULES_USER_PROMPT = """
## Here are the current insights that need to be merged:
{current_rules}

## Please consolidate and rewrite them into **no more than {limited_number} refined insights**.

As the summarizing agent, remove redundancies, combine similar ideas, and ensure clarity.
Each output insight must follow the "XXX is NECESSARY/IMPORTANT/NOT RECOMMENDED, because XXX" format.

Your output:
"""


def apply_scienceworld_insights_prompts():
    """Monkey-patch MemcompilerPrompts with ScienceWorld-specific prompts.
    Call this BEFORE Memcompiler initialization."""
    from core.memory.core_memory.prompt import MemcompilerPrompts
    MemcompilerPrompts.detect_mistakes_system_prompt = SW_DETECT_MISTAKES_SYSTEM_PROMPT
    MemcompilerPrompts.detect_mistakes_user_prompt = SW_DETECT_MISTAKES_USER_PROMPT
    MemcompilerPrompts.critique_compare_rules_system_prompt = SW_CRITIQUE_COMPARE_RULES_SYSTEM_PROMPT
    MemcompilerPrompts.critique_compare_rules_user_prompt = SW_CRITIQUE_COMPARE_RULES_USER_PROMPT
    MemcompilerPrompts.critique_success_rules_system_prompt = SW_CRITIQUE_SUCCESS_RULES_SYSTEM_PROMPT
    MemcompilerPrompts.critique_success_rules_user_prompt = SW_CRITIQUE_SUCCESS_RULES_USER_PROMPT
    MemcompilerPrompts.merge_rules_system_prompt = SW_MERGE_RULES_SYSTEM_PROMPT
    MemcompilerPrompts.merge_rules_user_prompt = SW_MERGE_RULES_USER_PROMPT
