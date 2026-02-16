from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from env.student_environment import StudentEnvironment
from env.agent.state_discretizer import discretize_state
from env.agent.sarsa_agent import SarsaAgent
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates



import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    fatigue: str = Form(...),
    stress: str = Form(...),
    retention: str = Form(...),
    days_left: int = Form(...),
    difficulty: str = Form(...)
):
    
    def level_to_number(level, type_):
        if type_ in ["fatigue", "stress"]:
            mapping = {
                "low": 20,
                "medium": 50,
                "high": 85
            }
        elif type_ == "retention":
            mapping = {
                "low": 0.2,
                "medium": 0.5,
                "high": 0.85
            }

        return mapping.get(level.lower(), 50)

    fatigue_value = level_to_number(fatigue, "fatigue")
    stress_value = level_to_number(stress, "stress")
    retention_value = level_to_number(retention, "retention")

    state = {
    "fatigue": fatigue_value,
    "stress": stress_value,
    "retention": retention_value,
    "days_left": days_left,
    "difficulty": difficulty
}


    # Discretize state
    discrete_state = discretize_state(state)

    # Get action from trained agent
    action = agent.choose_action(discrete_state)

    # Convert action number to readable label
    action_map = {
        0: "📚 Study",
        1: "🔁 Revise",
        2: "Break ☕",
        3:"sleep 🛌"
    }
    warning = None
    if fatigue_value > 80 or stress_value > 80:
        warning = "⚠ High burnout risk detected!"

    action_name = action_map.get(action, "Unknown")

    #-------------temporary fix prints----------------
    print("INPUT STATE:", fatigue, stress, retention, difficulty)
    print("DISCRETIZED:", discrete_state)
    print("CHOSEN ACTION:", action)



    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": action_name,
        "state": state
    })


from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load trained policy
with open("trained_policy.pkl", "rb") as f:
    q_table = pickle.load(f)

agent = SarsaAgent(actions=[0, 1, 2])

agent.q_table = q_table


class SimulationRequest(BaseModel):
    difficulty: str
    days: int



@app.post("/simulate")
def simulate_exam(request: SimulationRequest):
    env = StudentEnvironment(total_days=request.days)

    state = env.reset()
    env.state["difficulty"] = request.difficulty

    state = discretize_state(state)

    total_reward = 0
    actions_taken = []

    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        next_state = discretize_state(next_state)

        state = next_state
        total_reward += reward
        actions_taken.append(action)

    return {
        "total_reward": total_reward,
        "final_state": env.state,
        "actions": actions_taken
    }
