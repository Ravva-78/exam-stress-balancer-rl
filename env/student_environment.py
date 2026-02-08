"""
Student Environment for Exam Stress Balancer (Reinforcement Learning)

This environment simulates a student's cognitive state during exam preparation.
It is responsible for:
- Maintaining the current state (fatigue, stress, retention, etc.)
- Applying actions (study, revise, rest, sleep, ignore)
- Updating the state based on predefined transition rules
- Computing rewards for reinforcement learning agents

NOTE:
This file only defines the environment structure.
No reinforcement learning logic is implemented here.
"""

from typing import Dict, Tuple


class StudentEnvironment:
    """
    Simulated environment representing a student's exam preparation process.
    """

    def __init__(self, total_days: int = 30):
        """
        Initialize the environment with default values.

        Args:
            total_days (int): Number of days before the exam.
        """
        self.total_days = total_days
        self.current_day = 0

        # Initialize the student state
        self.state: Dict[str, float | int | str] = {}

        # Define possible actions
        self.actions = {
            0: "study",
            1: "revise",
            2: "break",
            3: "sleep",
            4: "ignore"
        }

        # Reset environment to initial state
        self.reset()

    def reset(self) -> Dict[str, float | int | str]:
        """
        Reset the environment to the initial state.

        Returns:
            dict: Initial state of the environment.
        """
        self.current_day = 0

        self.state = {
            "fatigue": 30,        # 0 to 100
            "stress": 20,         # 0 to 100
            "retention": 0.2,     # 0.0 to 1.0
            "days_left": self.total_days,
            "difficulty": "medium"
        }

        return self.state

    def step(self, action: int):
        reward = 0.0

        fatigue = self.state["fatigue"]
        stress = self.state["stress"]
        retention = self.state["retention"]

        # ---------- STUDY ----------
        if action == 0:
            learning_gain = 0.1 * (1 - fatigue / 100)

            self.state["retention"] = min(1.0, retention + learning_gain)
            self.state["fatigue"] = min(100, fatigue + 10)
            self.state["stress"] = min(100, stress + 5)

            reward = learning_gain * 10 - (self.state["stress"] - stress) * 0.2

        # ---------- REVISE ----------
        elif action == 1:
            revision_gain = 0.05

            self.state["retention"] = min(1.0, retention + revision_gain)
            self.state["fatigue"] = max(0, fatigue - 5)
            self.state["stress"] = max(0, stress - 3)

            reward = revision_gain * 8 + 1

        # ---------- TIME ----------
        self.state["days_left"] -= 1
        self.current_day += 1
        done = self.state["days_left"] <= 0

        return self.state, reward, done

    



    def get_state(self) -> Dict[str, float | int | str]:
        """
        Get the current state of the environment.

        Returns:
            dict: Current state.
        """
        return self.state

    def get_action_meaning(self, action: int) -> str:
        """
        Get human-readable meaning of an action.

        Args:
            action (int): Action index.

        Returns:
            str: Action description.
        """
        return self.actions.get(action, "unknown")

    