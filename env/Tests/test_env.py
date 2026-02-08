from env.student_environment import StudentEnvironment

env = StudentEnvironment(total_days=3)
state = env.reset()

print("Initial:", state)

actions = [0, 0, 0]  # study continuously → burnout

for day in range(3):
    action = actions[day]
    state, reward, done = env.step(action)
    print(f"Day {day+1}: Action={action}, State={state}, Reward={reward:.2f}")


