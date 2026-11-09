# Netflix Top 10 Survival — MLOps Pipeline

Predicting whether a title in this week's Netflix global Top 10 will still be in the Top 10 next week, built as a fully automated, reproducible MLOps pipeline: data ingestion and versioning, automated training with experiment tracking, a containerized prediction API, and drift monitoring.

> **Status:** work in progress. Sections marked _TODO_ are filled in as each phase is completed.

---

## Problem statement

Each week, Netflix publishes its global Top 10 lists, but a title's current rank says little about whether its audience will last. For every title in this week's Top 10, we predict the probability that it remains in the Top 10 next week. A streaming programming team could use these predictions to decide which titles keep premium promotional slots.

- **Target:** `1` if the title appears on the same Top 10 list the following week, else `0`
- **Metric:** PR-AUC, compared against a rank-based baseline (titles ranked 1–5 survive)
- **Data:** [Netflix Top 10](https://www.netflix.com/tudum/top10) weekly lists (June 2021 onward), optionally enriched with metadata from the [TMDB API](https://developer.themoviedb.org/)

## Architecture

_TODO: add the architecture diagram image here._

| Stage | Tool |
| --- | --- |
| Ingestion and validation | Python, Pandera |
| Data versioning | DVC |
| Orchestration | Prefect (weekly flow) |
| AutoML and experiment tracking | FLAML, MLflow |
| Model registry | MLflow Model Registry (semantic versions, `champion` / `challenger` aliases) |
| Serving | FastAPI in Docker |
| Monitoring and drift detection | Evidently |
| CI | GitHub Actions (ruff, pytest, Docker build) |

## Repository structure

```
.
├── api/             # FastAPI prediction service and Dockerfile
├── data/            # Raw and processed data (tracked by DVC, not Git)
├── flows/           # Prefect flow that runs the weekly pipeline
├── monitoring/      # Evidently reports, dashboard, drift simulation (corrupt.py)
├── notebooks/       # EDA and experiments
├── src/             # Ingestion, validation, feature building, training code
├── tests/           # Unit, leakage, and API tests
├── .github/         # GitHub Actions workflows
├── requirements.txt
└── README.md
```

## Getting started

### Prerequisites

- Python 3.12
- Git
- Docker Desktop
- Access to the team's DVC remote (ask a teammate for credentials)

### 1. Clone and set up the environment

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the data

```bash
dvc pull
```

### 3. Set environment variables

Create a `.env` file in the project root (it is ignored by Git, so never commit it):

```
TMDB_API_KEY=your_key_here
MLFLOW_TRACKING_URI=http://localhost:5000
```

### 4. Start the services

_TODO: confirm once `docker-compose.yml` is complete._

```bash
docker compose up
```

| Service | URL |
| --- | --- |
| MLflow UI | http://localhost:5000 |
| Prefect UI | http://localhost:4200 |
| Prediction API docs | http://localhost:8000/docs |
| Monitoring dashboard | _TODO_ |

## Running the pipeline

_TODO: confirm the command once the flow is built._

```bash
python flows/weekly_flow.py
```

This runs ingest → validate → build features → train → evaluate → register → monitor. A model is registered only if it beats the baseline PR-AUC.

## Using the API

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | Survival probability for one title |
| `POST /predict/batch` | Survival probabilities for many titles |
| `GET /health` | Service health check |
| `GET /model` | Which model version is serving |

_TODO: add an example request and response._

## Monitoring and drift testing

_TODO: describe the clean baseline validation, the drift scenarios in `monitoring/corrupt.py`, and what caught each one._

## Running tests

```bash
pytest
ruff check .
```

## Contributing

- Each team member works on their own branch.
- All changes reach `main` through a pull request with at least one approving review.
- CI (lint, tests, Docker build) must pass before merging.

## Team

| Name | Role |
| --- | --- |
| _TODO_ | Data lead |
| _TODO_ | Modeling lead |
| _TODO_ | Serving lead |
| _TODO_ | Monitoring lead |
