# 🎓 Exam Stress Balancer
### Reinforcement Learning Based Adaptive Study Planning System

Exam Stress Balancer is an AI-driven adaptive study planning system that models a student's cognitive state and recommends optimal actions (Study / Revise / Break) using Reinforcement Learning (SARSA).

This project simulates real-world academic pressure and dynamically balances:

- 📚 Learning gain
- 🧠 Memory retention
- 😓 Stress levels
- 💤 Fatigue accumulation
- ⏳ Time urgency before exams

---

## 🚀 Project Overview

Traditional study planning is static. This system adapts decisions based on:

- Current fatigue
- Current stress
- Memory retention
- Days left before exam
- Exam difficulty

The agent learns an optimal strategy through experience using SARSA (State-Action-Reward-State-Action) learning.

---

## 🧠 Reinforcement Learning Design

### 🔹 State Space (5D Discretized)
(fatigue_level, stress_level, retention_level, urgency, difficulty)

- Fatigue: LOW / MEDIUM / HIGH  
- Stress: LOW / MEDIUM / HIGH  
- Retention: LOW / MEDIUM / HIGH  
- Urgency: LOW / MEDIUM / HIGH  
- Difficulty: EASY / MEDIUM / HARD  

---

### 🔹 Actions
| Action ID | Meaning |
|-----------|---------|
| 0 | Study |
| 1 | Revise |
| 2 | Break / Sleep |

---

### 🔹 Reward Engineering Includes

- Learning gain bonus
- Burnout penalties
- Overstudying penalty
- Stress sensitivity
- Retention zone bonus
- Urgency-based reward adjustment (when days_left ≤ 3)

---

## 📊 Training Details

- Algorithm: **SARSA**
- Episodes: 2000+
- Epsilon-greedy exploration
- Learned Q-table size: ~160+ states
- Final Average Reward: Positive (~47+)
- Policy saved as: `trained_policy.pkl`

---

## 🌐 Web Application

Built using:

- **FastAPI** (Backend API)
- **Uvicorn** (ASGI Server)
- **Custom HTML + CSS UI**
- Glassmorphism UI Design
- Real-time prediction endpoint (`/predict`)

Users input:

- Fatigue
- Stress
- Retention
- Days until exam
- Exam difficulty

System outputs recommended action.

---

## 📁 Project Structure


exam-stress-balancer-rl/
│
├── env/
│ ├── student_environment.py
│ ├── train_sarsa.py
│ ├── agent/
│ │ ├── sarsa_agent.py
│ │ └── state_discretizer.py
│
├── templates/
│ └── index.html
│
├── static/
│ └── style.css
│
├── api.py
├── trained_policy.pkl
├── requirements.txt
└── README.md


---

## ⚙️ Installation & Run
### 1️⃣ Clone Repository

git clone https://github.com/YOUR\_USERNAME/exam-stress-balancer.git
cd exam-stress-balancer


### 2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate # Windows


### 3️⃣ Install Dependencies
pip install -r requirements.txt


### 4️⃣ Train Model (Optional)
python -m env.train_sarsa


### 5️⃣ Run Web App
python -m uvicorn api:app --reload
open:http://127.0.0.1:8000


---

## 🧪 Test Scenarios

The system was tested on 11 structured academic stress scenarios including:

- High stress & low retention
- Low stress & high retention
- High urgency exam
- Burnout conditions
- Easy vs Hard exam comparison

Model performance stabilized with positive reward trends.

---

## 🎯 Academic Objective

This project demonstrates:

- Custom RL environment modeling
- Reward shaping design
- Cognitive state simulation
- Policy learning via SARSA
- Full-stack AI deployment

---

## 🔮 Future Roadmap

### Phase 2
- Dashboard analytics
- Stress & fatigue trend graphs
- Burnout warning system
- Decision explanation layer

### Phase 3
- LLM integration for syllabus parsing
- Chapter upload planning
- AI-generated study schedules
- Mobile app version

---

## 👨‍💻 Author

Developed as part of academic research in AI-driven adaptive learning systems.

---

## 📜 License

Academic & Educational Use


