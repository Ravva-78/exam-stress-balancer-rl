from env.student_environment import StudentEnvironment
from env.agent.sarsa_agent import SarsaAgent
from env.agent.state_discretizer import discretize_state


import numpy as np
import pickle

EPISODES = 3000

env = StudentEnvironment(total_days=15)
actions = [0, 1, 2]  # study, revise, sleep

agent = SarsaAgent(actions)

episode_rewards = []

for episode in range(EPISODES):

    state = env.reset()
    # Randomize initial conditions
    env.state["fatigue"] = np.random.randint(0, 101)
    env.state["stress"] = np.random.randint(0, 101)
    env.state["retention"] = np.random.uniform(0, 1)
    env.state["difficulty"] = np.random.choice(["easy", "medium", "hard"])
    state = discretize_state(state)
    print("Initial state:", state)


    action = agent.choose_action(state)

    total_reward = 0
    done = False
    action_counts = {0: 0, 1: 0, 2: 0}


    while not done:

        next_state, reward, done = env.step(action)
        action_counts[action] += 1

        next_state = discretize_state(next_state)
        print("Next state:", next_state)

        next_action = agent.choose_action(next_state)

        agent.update(state, action, reward, next_state, next_action)

        state = next_state
        action = next_action
        total_reward += reward

        if done:
            break

    agent.epsilon = max(0.05, agent.epsilon * 0.997)  # Decay epsilon
    episode_rewards.append(total_reward)
    print(f"Episode {episode}: Study={action_counts[0]}, Revise={action_counts[1]}, Sleep={action_counts[2]}")


print("Training Complete")
print("Average Reward:", sum(episode_rewards) / len(episode_rewards))

with open("trained_policy.pkl", "wb") as f:
    pickle.dump(agent.q_table, f)

print("Policy saved as trained_policy.pkl")

# print("Training complete.")
# print("Average reward (last 50 episodes):",
#       np.mean(episode_rewards[-50:]))

print("Total learned states:", len(agent.q_table))


import matplotlib.pyplot as plt

window = 20
smoothed = np.convolve(episode_rewards, 
                       np.ones(window)/window, 
                       mode='valid')

plt.plot(smoothed)
plt.title("SARSA Training Rewards (Smoothed)")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.show()

