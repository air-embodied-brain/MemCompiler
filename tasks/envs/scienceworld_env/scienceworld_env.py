"""
ScienceWorld Environment Wrapper for the Manager-Executor architecture.

Wraps the ScienceWorldEnv from the scienceworld package and implements
the BaseEnv interface used by the Memcompiler Manager-Executor framework.
"""

import re
from typing import List, Optional, Tuple

from scienceworld import ScienceWorldEnv

from envs.base_env import BaseEnv


class ScienceWorldEnvWrapper(BaseEnv):
    """ScienceWorld environment adapter implementing the BaseEnv interface."""

    def __init__(self, env_config: dict, max_trials: int = 100):
        """
        Args:
            env_config: Optional configuration (e.g., serverPath).
            max_trials: Maximum steps per episode (maps to envStepLimit).
        """
        server_path = env_config.get("serverPath", None)
        if server_path:
            self.env = ScienceWorldEnv(envStepLimit=max_trials, serverPath=server_path)
        else:
            self.env = ScienceWorldEnv(envStepLimit=max_trials)
        self.max_trials = max_trials
        self.current_score = 0
        self.current_task_name = ""
        self.current_variation = 0
        self.simplification_str = ""
        self._initial_obs = ""
        self.last_matched_action = ""
        self.selected_obs = []
        self.modified_goal = ""
        self.difficulty = ""
        self.finished_sub_goal = []

    def set_env(self, task_config: dict) -> Tuple[str, str]:
        """Load a specific task variation.

        Args:
            task_config: Dict with keys:
                - task_name (str): Task name (e.g., "boil")
                - variation_idx (int): Variation index
                - simplification_str (str): Simplification options string
                - subgoals (list[str]): Regex patterns for subgoal matching
                - modified_goal (str): Goal text from test.jsonl
                - id (int): Task ID
                - difficulty (str): Task difficulty level

        Returns:
            (task_main, task_description):
                task_main: Task description text (used as Memcompiler retrieval key)
                task_description: Full task description text
        """
        task_name = task_config["task_name"]
        variation_idx = task_config["variation_idx"]
        simplification_str = task_config.get("simplification_str", "")
        subgoals = task_config.get("subgoals", [])
        modified_goal = task_config.get("modified_goal", "")
        task_id = task_config.get("id", 0)
        difficulty = task_config.get("difficulty", "easy")

        self.env.load(task_name, variation_idx, simplification_str)
        obs, info = self.env.reset()

        self.current_score = 0
        self.current_task_name = task_name
        self.current_variation = variation_idx
        self.simplification_str = simplification_str
        self._initial_obs = obs

        # Subgoal tracking (aligned with Memcompiler-main)
        self.selected_obs = subgoals
        self.modified_goal = modified_goal
        self.difficulty = difficulty
        self.finished_sub_goal = [0 for _ in range(len(self.selected_obs))]

        task_main = self.env.get_task_description() + f"___{task_id}"
        task_description = (f"- Here is your start point(you should check it carefully "
                           f"before you start to action):\n{obs}\n"
                           f"- {modified_goal}"
                           f"* Hint: You should use `look around` command to get some clues. *")

        return task_main, task_description

    def step(self, action: str) -> Tuple[str, float, bool]:
        """Execute an action in the environment.

        Args:
            action: The action string to execute.

        Returns:
            (observation, reward, done):
                observation: Text observation after the action
                reward: Subgoal-based reward (aligned with Memcompiler-main)
                done: Whether all subgoals are completed
        """
        obs, reward_raw, done_env, info = self.env.step(action)
        self.current_score = info.get('score', 0)

        # Subgoal regex matching (aligned with Memcompiler-main)
        self._complete_sub_goals(obs)
        done = self._check_is_done()

        if obs == "No known action matches that input.":
            reward = -1
        else:
            reward = 1 if done else 0

        return obs, reward, done

    def get_observation(self) -> str:
        """Get current observation (look around). Free action."""
        return self.env.look()

    def get_inventory(self) -> str:
        """Get current inventory. Free action."""
        return self.env.inventory()

    def get_task_description(self) -> str:
        """Get the task description text."""
        return self.env.taskdescription()

    def get_action_templates(self) -> List[str]:
        """Get currently available action templates."""
        return self.env.get_possible_actions()

    def get_available_objects(self) -> List[str]:
        """Get currently interactable objects."""
        return self.env.get_possible_objects()

    def get_valid_actions(self) -> List[str]:
        """Get all valid action-object combinations for SentenceTransformer matching."""
        return self.env.get_valid_action_object_combinations()

    def get_score(self) -> int:
        """Get current score (0-100)."""
        return self.current_score

    def get_initial_obs(self) -> str:
        """Get the initial observation from the last reset."""
        return self._initial_obs

    @classmethod
    def process_action(cls, action: str) -> str:
        """Clean up LLM-generated action text."""
        action = action.strip()
        # Remove possible prefix numbering (e.g., "1. pick up thermometer")
        action = re.sub(r'^\d+[\.\)]\s*', '', action)
        # Remove possible quote wrapping
        action = action.strip('"').strip("'")
        return action

    def feedback(self) -> Tuple[float, bool, str]:
        """Return (progress_rate, done, message) aligned with Memcompiler-main."""
        progress_rate: float = self._get_progress_rate()
        done: bool = self._check_is_done()

        if done:
            message = "You successfully finished this task!"
        else:
            finished_subgoals = '\n'.join(
                [f'{i + 1}. {goal}' for i, goal in enumerate(self._get_finished_subgoals())])
            unfinished_subgoals = '\n'.join(
                [f'{i + 1}. {goal}' for i, goal in enumerate(self._get_unfinished_subgoals())])
            message = (f"\nIn this task, you successfully finished these subgoals:\n"
                       f"{finished_subgoals}.\n"
                       f"But you failed in the following subgoals:\n{unfinished_subgoals}")
        return progress_rate, done, message

    def _complete_sub_goals(self, obs: str):
        """Check observation against subgoal regex patterns."""
        for i, pattern in enumerate(self.selected_obs):
            match = re.search(pattern, obs)
            if match:
                self.finished_sub_goal[i] = 1.

    def _check_is_done(self) -> bool:
        """All subgoals completed."""
        if not self.selected_obs:
            return self.current_score >= 100
        return sum(self.finished_sub_goal) >= len(self.selected_obs)

    def _get_progress_rate(self) -> float:
        """Fraction of subgoals completed."""
        if not self.selected_obs:
            return 1.0 if self.current_score >= 100 else 0.0
        return sum(self.finished_sub_goal) * 1.0 / len(self.finished_sub_goal)

    def _get_finished_subgoals(self) -> List[str]:
        return [self.selected_obs[i] for i in range(len(self.finished_sub_goal))
                if self.finished_sub_goal[i] == 1]

    def _get_unfinished_subgoals(self) -> List[str]:
        return [self.selected_obs[i] for i in range(len(self.finished_sub_goal))
                if self.finished_sub_goal[i] == 0]

    def close(self):
        """Close the environment and release the JVM process."""
        try:
            self.env.close()
        except Exception as e:
            print(f"[ScienceWorldEnvWrapper] Error closing env: {e}")


def get_scienceworld_task_configs(
    task_names: Optional[List[str]] = None,
    split: str = "test",
    simplification_str: str = "",
) -> List[dict]:
    """Generate ScienceWorld task configuration list.

    Args:
        task_names: List of task names to include. None = all 30 tasks.
        split: Dataset split ("test", "dev", or "train").
        simplification_str: Override simplification for all tasks.
            If empty, uses "easy" for non-electrical tasks and
            "teleportAction,selfWateringFlowerPots,openContainers,openDoors"
            for electrical tasks (matching ScienceWorld source logic).

    Returns:
        List of task config dicts, each with keys:
            task_name, task_id, variation_idx, simplification_str
    """
    env = ScienceWorldEnv()

    all_task_names = task_names if task_names else env.get_task_names()
    configs = []

    for tn in all_task_names:
        env.load(tn, 0)

        if split == "test":
            variations = env.get_variations_test()
        elif split == "dev":
            variations = env.get_variations_dev()
        else:
            variations = env.get_variations_train()

        # Determine simplification per task type (confirmed in Q3)
        # Logic matches ScienceWorld source (scienceworld.py line 119):
        #   is_electrical_task = "power-component" in taskName or "conductivity" in taskName
        # Affected tasks: power-component, power-component-renewable-vs-nonrenewable-energy,
        #   test-conductivity, test-conductivity-of-unknown-substances
        effective_simplification = simplification_str
        if not effective_simplification:
            is_electrical = "power-component" in tn or "conductivity" in tn
            if is_electrical:
                effective_simplification = (
                    "teleportAction,selfWateringFlowerPots,openContainers,openDoors"
                )
            else:
                effective_simplification = "easy"

        task_description = env.taskdescription()

        for var_idx in variations:
            configs.append({
                "task_name": tn,
                "task_description": task_description,
                "variation_idx": var_idx,
                "simplification_str": effective_simplification,
            })

    env.close()
    return configs
