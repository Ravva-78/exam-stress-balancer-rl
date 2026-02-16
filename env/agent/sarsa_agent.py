import random
from collections import defaultdict


class SarsaAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0):
        """
        SARSA Agent

        actions: list of action indices
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: state -> action values
        self.q_table = {}

    def choose_action(self, state):
        """
        ε-greedy action selection
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward, next_state, next_action):

        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}

        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]

        # Proper SARSA update (ONLY ONCE)
        self.q_table[state][action] = current_q + self.alpha * (
            reward + self.gamma * next_q - current_q
        )

