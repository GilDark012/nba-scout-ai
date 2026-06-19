# 🏀 NBA Scout AI

# 🏀 NBA Scout AI

> A production-ready end-to-end ML project: NBA shot efficiency prediction, player scouting dashboard, REST API, and model monitoring.

[![CI/CD](https://github.com/GilDark012/nba-scout-ai/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/GilDark012/nba-scout-ai/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)

---

## 🎯 Product Goal

NBA Scout AI is a basketball scouting product for **coaches, scouts, journalists, and fans**.

> "Search a player → get a visual scouting report about shot tendencies, recent form, and shot efficiency — no machine learning knowledge required."

---

## 🏗️ Architecture
User (Streamlit Dashboard)
│
▼
FastAPI Backend (REST API)
│
├── nba_api / Kaggle dataset
├── XGBoost Model (joblib)
├── MLflow experiment tracking
└── Evidently AI monitoring

---

## 📁 Project Structure

nba-scout-ai/
├── data/ # Raw and processed shot data
├── src/
│ ├── pipeline/ # fetch → preprocess → features → train → evaluate
│ ├── api/ # FastAPI backend
│ ├── monitoring/ # Evidently AI drift reports
│ └── dashboard/ # Streamlit product UI
├── models/ # Saved best model
├── reports/ # Evaluation figures + monitoring HTML
├── tests/ # Pytest test suite
├── Dockerfile # Backend container
├── render.yaml # Render deployment config
└── requirements.txt # Pinned dependencies


---

## ⚡ Local Setup

```bash
# 1. Clone
git clone https://github.com/GilDark012/nba-scout-ai.git
cd nba-scout-ai

# 2. Create conda environment
conda create -n nba-scout-ai python=3.11 -y
conda activate nba-scout-ai

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy env file
cp .env.example .env
```

---

## 🚀 Run the Full Pipeline

```bash
# Fetch / load data (Kaggle CSV)
python src/pipeline/fetch_data.py --player "LeBron James"

# Preprocess
python src/pipeline/preprocess.py --input data/raw/lebron_james_shots.csv

# Feature engineering
python src/pipeline/feature_engineering.py

# Train models (logs to MLflow)
python src/pipeline/train.py

# Evaluate
python src/pipeline/evaluate.py

# Monitoring report
python src/monitoring/monitor.py
```

---

## 🌐 Run Locally

```bash
# Terminal 1 — Start API backend
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — Start Streamlit dashboard
streamlit run src/dashboard/app.py

# Terminal 3 — View MLflow experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## 🐳 Docker

```bash
docker build -t nba-scout-ai:latest .
docker run -p 8000:8000 nba-scout-ai:latest
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/players?query=` | Search players by name |
| `POST` | `/predict` | Predict shot outcome |
| `GET` | `/player-report` | Full scouting report |
| `GET` | `/monitoring` | Model health summary |
| `GET` | `/monitoring/report` | Full Evidently HTML report |
| `GET` | `/docs` | Interactive API docs |

---

## 🚢 Deployment

| Component | Platform | URL |
|---|---|---|
| FastAPI Backend | Render (Docker) | `https://nba-scout-ai-api.onrender.com` |
| Streamlit Dashboard | Streamlit Cloud | `https://nba-scout-ai-dashboard.streamlit.app` |

---

## 📸 Screenshots

> *(Add screenshots of the Streamlit dashboard tabs here)*

---

## 💼 CV-Ready Highlights

- **End-to-end ML pipeline**: data ingestion → preprocessing → feature engineering → training → evaluation
- **Binary classification** with XGBoost and Logistic Regression baseline; tracked with **MLflow**
- **Production REST API** with FastAPI, Pydantic validation, CORS, and automatic OpenAPI docs
- **Model monitoring** with **Evidently AI**: data drift detection and performance tracking
- **Non-technical product UI** with Streamlit and Plotly — scouting dashboard for coaches and fans
- **Dockerised backend** deployed on **Render** with health checks and CI/CD via **GitHub Actions**
- **Reproducible environment** with conda, pinned `requirements.txt`, and `.env` config

---

## 📊 Dataset

NBA shot logs from [Kaggle — dansbecker/nba-shot-logs](https://www.kaggle.com/datasets/dansbecker/nba-shot-logs)
covering the 2014–2015 NBA regular season (~126,000 shots).