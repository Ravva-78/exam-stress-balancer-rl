import pickle
from env.student_environment import StudentEnvironment
from env.agent.sarsa_agent import SarsaAgent
from env.agent.state_discretizer import discretize_state
import random

# Load trained policy
with open("trained_policy.pkl", "rb") as f:
    q_table = pickle.load(f)

env = StudentEnvironment(total_days=5)
actions = [0, 1, 2]

agent = SarsaAgent(actions)
agent.q_table = q_table
agent.epsilon = 0.0  # pure greedy (no exploration)

state = env.reset()
state = discretize_state(state)

done = False
total_reward = 0
action_counts = {0: 0, 1: 0, 2: 0}

print("=== Running Trained Policy ===")

while not done:
    action = agent.choose_action(state)
    action = agent.choose_action(state)

    next_state, reward, done = env.step(action)
    next_state = discretize_state(next_state)

    print(f"Action: {action}, Reward: {reward:.2f}")

    state = next_state
    total_reward += reward

print("Total Reward:", total_reward)
print("Action Distribution:", action_counts)
print("Final State:", env.state)




def run_trained_policy(difficulty):

    env = StudentEnvironment(total_days=5, difficulty=difficulty)

    state = env.reset()
    state = discretize_state(state)

    done = False
    total_reward = 0
    action_counts = {0: 0, 1: 0, 2: 0}

    while not done:
        action = agent.choose_action(state)
        action_counts[action] += 1

        next_state, reward, done = env.step(action)
        next_state = discretize_state(next_state)

        state = next_state
        total_reward += reward

    print("Difficulty:", difficulty)
    print("Total Reward:", total_reward)
    print("Action Distribution:", action_counts)
    print("Final State:", env.state)
    print("-" * 40)

print("=== Testing Trained Policy on All Difficulties ===")

for diff in ["easy", "medium", "hard"]:
    run_trained_policy(diff)
