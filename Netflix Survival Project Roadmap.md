# Netflix Survival Project Roadmap

Nov 9, 2026 · @Shannon

## Overview

Seven phases take the project from an empty repo to the Week 10 presentation, with every part built by the end of Week 9. Skip anything your team has already finished.

&#91;embedded content: Roadmap · 7 phases over Weeks 7–10, 3 gates\]

Phases 4 and 5 start a week early on stand-ins (a dummy model, the feature table), so nobody sits waiting on another phase.

### Team roles

| Role | Owns | Phases |
| --- | --- | --- |
| Data lead | Ingestion, DVC, features, holdout, baseline | 2 |
| Modeling lead | MLflow, FLAML, registry, Prefect flow | 3 |
| Serving lead | FastAPI, Docker, API tests, Compose | 4 |
| Monitoring lead | Evidently, dashboard, alerts, drift tests | 5–6 |
| Everyone | Repo setup, PR reviews, README, slides | 1, 7 |

With three people, the serving lead also helps with drift testing; with five, give drift testing its own owner.

## Phase 1: Repo and CI foundations

**Week 7 · Owner: everyone.** Set up the shared repo and the automatic checks first, because every later piece of work arrives through them.

- [ ] **Create the GitHub repo and add everyone**

  One branch per person, merged into `main` only through pull requests. The PR history is graded, and a PR means someone else checks each change.
- [ ] **Protect the `main` branch**

  Require one review and a green check before anything merges. This stops broken code from landing on the branch everyone builds from.
- [ ] **Lay out the folders**

  For example `src/`, `flows/`, `api/`, `monitoring/`, `tests/`, `notebooks/`, and `data/` (tracked by DVC, not Git). Everyone knows where their piece goes from day one.
- [ ] **Write `requirements.txt` with pinned versions**

  Teammates and the grader then install exactly the same library versions, so "it works on my machine" problems show up early.
- [ ] **Set up GitHub Actions**

  Run ruff (linting) and pytest on every pull request; the Docker build gets added in Phase 4. Tests are added as each phase writes code, so mistakes are caught automatically.
- [ ] **Start the README and a shared "challenges" list**

  Both are much easier to fill in as you go than to reconstruct in Week 10, and challenges are their own graded slide.

**Done when:** a teammate opens a small PR and the checks run green before it merges.

## Phase 2: Data, features, and the holdout

**Week 7 · Owner: data lead.** Turn Netflix's weekly files into a clean, versioned table of labeled title-weeks, and lock away the test weeks before anyone trains a model. Every later phase reads this table, so it comes first.

- [ ] **Write `ingest.py`**

  It downloads the global (and countries) TSV and saves a dated copy. A dated snapshot means you can always rebuild exactly the data a model was trained on.
- [ ] **Add Pandera checks**

  Check that the expected columns exist with the right types, rank is between 1 and 10, and hours viewed is above 0. Bad data stops here, before it reaches training, and the same rules later catch some of your corrupted data.
- [ ] **Version the data with DVC**

  Track `data/raw` and `data/processed` and push them to a shared remote (Google Drive or S3). Git can't hold large data files; DVC lets every teammate pull identical data with one command.
- [ ] **Build the feature table**

  One row per title per week. The label is 1 if the title appears on the same list the following week. Features: rank, rank change from last week, lagged hours viewed, weeks in the Top 10 so far, and week of year. This table is what the model learns from.
- [ ] **Handle views carefully**

  Compute views ≈ hours ÷ runtime only where runtime exists. Don't fill missing early weeks with 0: the model would learn "views is missing" as a stand-in for the date.
- [ ] **Optional: add TMDB details**

  Look up each new title once, cache the result, and record the match rate. This adds genre, language, and release date, and the cache keeps reruns fast.
- [ ] **Split by time**

  Hold out the last 20 weeks, and drop the very last week (its label isn't known yet). Splitting by time mimics predicting the future, and the holdout stays untouched until Phase 5.
- [ ] **Score the baseline**

  The rule "rank 5 or better survives" is your baseline; record its PR-AUC. A model only earns deployment if it beats this.
- [ ] **Write the data tests**

  Test the feature builder, add a label-leakage test (no feature may use next week's data), and test the title lookup if you use TMDB. CI then catches broken features automatically.

**Done when:** a teammate runs `dvc pull`, gets identical data, and sees the baseline PR-AUC.

## Phase 3: Automated training pipeline

**Week 8 · Owner: modeling lead.** Automate training so one run builds features, searches for the best model, logs every attempt, and registers the winner only if it beats the baseline. It needs Phase 2's table, and serving needs the model it registers.

- [ ] **Start the MLflow tracking server**

  Use a SQLite backend and a local artifact store. Every experiment's settings and scores end up in one place, and its dashboard becomes your experiment-tracking slide.
- [ ] **Run the FLAML AutoML search**

  Try LightGBM, logistic regression, and random forest, logging every trial as an MLflow run. Tune with cross-validation on the training weeks only, so the holdout stays untouched.
- [ ] **Save preprocessing and model as one sklearn Pipeline**

  The API then applies exactly the same feature steps as training did. This prevents training-serving skew, where the live model sees data prepared differently than in training.
- [ ] **Register the model behind a gate**

  Register only if PR-AUC beats the baseline. The first version is `v1.0.0` with the alias `champion`; newer runs get `challenger`. Semantic versioning is required, and the gate keeps a worse model from ever reaching serving.
- [ ] **Wire the Prefect flow**

  Chain ingest → validate → build\_features → train → evaluate → register, scheduled weekly on Tuesday evening, when Netflix publishes new lists. The monitor step gets added in Phase 5. Pipeline automation is its own graded item.

**Done when:** one Prefect run ends with a `champion` model visible in the MLflow registry.

## Phase 4: Serving the model

**Weeks 7–8 · Owner: serving lead.** Package the champion model in a Docker container behind a FastAPI service that returns survival probabilities and logs every request. Monitoring reads that log, so serving comes before monitoring. Start in Week 7 with a dummy model, then swap in the real champion once Phase 3 registers it.

- [ ] **Build the FastAPI app**

  Endpoints: `/predict` (one title), `/predict/batch` (many), `/health`, and `/model` (which version is serving). The brief requires real-time predictions, and the batch endpoint is what you use for the validation and drift tests.
- [ ] **Define the request schema with Pydantic**

  Requests with missing or wrongly typed fields are rejected immediately. This is your first line of defense against schema changes.
- [ ] **Load the champion at startup**

  The API asks the registry for whatever model holds the `champion` alias, so promoting a new model needs no code change.
- [ ] **Log every request**

  Save the inputs, the prediction, and the model version to a prediction log (parquet). Evidently compares this log against the training data.
- [ ] **Write the Dockerfile and add the build to CI**

  Use a `python:3.12-slim` base image. Everyone runs the same environment, and CI proves the image still builds after every change.
- [ ] **Add the API tests**

  A contract test checks that bad requests are rejected. A regression test checks that the API's predictions match the registered model's on a small fixed sample, proving the deployed model is the one you registered.
- [ ] **Start `docker-compose.yml`**

  Begin with MLflow and the API; add Prefect and the dashboard in Phase 5. The goal is one `docker compose up` that brings up everything, which is how graders reproduce your system.

**Done when:** `docker compose up` starts the API and a request to `/predict` returns a probability.

## Phase 5: Monitoring and the clean baseline

**Weeks 8–9 · Owner: monitoring lead.** Set up Evidently against the training data, then send the clean holdout through the API to prove nothing alarms when the data is good. This needs the API's prediction log, and it must happen before the drift tests so their alerts mean something. Prototype the reports on the feature table in Week 8, before the API is ready.

- [ ] **Set the reference data**

  The training data is "what normal looks like". All drift is measured against it.
- [ ] **Build the Evidently reports**

  Data drift (have the inputs changed?), target drift (has the survival rate changed?), and classification performance (is the model still accurate?). Run them weekly and for each validation batch. Together they cover both input changes and accuracy decay.
- [ ] **Set up the dashboard**

  The Evidently UI, or a small Streamlit page that reads the report JSON. The brief requires a dashboard, and it supplies your monitoring screenshots.
- [ ] **Write the alert rules before testing**

  Alert on the share of drifted features, the rate of zero runtimes, and a TMDB match rate below 80%; a breach opens a GitHub issue. Fixing the thresholds first keeps the drift test honest.
- [ ] **Run the clean baseline**

  Send the 20 holdout weeks to `/predict/batch`. Confirm no alerts fire and the PR-AUC is close to what MLflow recorded. The brief requires this check: it proves monitoring stays quiet on good data.
- [ ] **Add the monitor step to the Prefect flow**

  Monitoring then runs automatically every week after new predictions arrive.

**Done when:** the clean holdout passes with no alerts, and a dashboard screenshot is saved.

## Phase 6: Drift and stress testing

**Week 9 · Owner: monitoring lead plus one helper.** Corrupt the holdout in several ways, send each version to the API, and record which part of the system catches it. It needs the clean baseline from Phase 5, and it is the last graded stage of the pipeline.

| Variant in `corrupt.py` | What it simulates | Expected to be caught by |
| --- | --- | --- |
| Rank set to 11 | Out-of-bounds value | Pydantic rejects the request |
| Hours viewed below zero | Impossible value | Pydantic or Pandera |
| Renamed column | Schema change | Pydantic rejects the request |
| Runtime set to 0 | Broken upstream field | Runtime-zero alert |
| Unseen category | New, unknown value | Evidently data drift |
| Rank and weeks-in-Top-10 swapped | Values valid but wrong | Evidently data drift only |

Use "hours below zero" rather than "views greater than hours": a title shorter than one hour legitimately has more views than hours.

- [ ] **Write `corrupt.py` with one function per variant**

  Each variant tests a different layer of the system, so together they show defense in layers.
- [ ] **Send each variant to `/predict/batch`, one at a time**

  One at a time, so each alert is clearly tied to one cause.
- [ ] **Record what caught each variant**

  Note whether Pydantic, Pandera, or an Evidently alert caught it, and how quickly. The finished table becomes your drift-analysis slide.
- [ ] **Screenshot every alert and report**

  The brief asks for visual evidence that monitoring responded.
- [ ] **Optional: replay the June 2023 metric switch**

  Train a model on weeks before June 2023 and replay the weeks after it. This shows real-world drift next to the synthetic kind.

**Done when:** every variant has been caught, recorded, and screenshotted.

## Phase 7: Wrap-up

**Week 10 · Owner: everyone.** Make the repo reproducible and turn the screenshots you've collected into the presentation. This comes last because it documents a finished system, but it only works if screenshots and challenges were saved along the way.

- [ ] **Finish the README**

  Cover setup, `dvc pull`, `docker compose up`, running the Prefect flow, sending test data, and opening the dashboard. Reproducibility is graded directly.
- [ ] **Run a clean-machine test**

  A teammate who didn't build a part follows the README from scratch. Missing steps show up here instead of in front of the grader.
- [ ] **Check code comments and PR history**

  The brief asks for fully commented code and visible check-in and PR logs.
- [ ] **Build the slides**

  Problem and EDA; the metric (PR-AUC versus the rank baseline) and why; the architecture diagram; the MLflow dashboard; the API and monitoring dashboard; the drift results table; challenges and future work (for example data lineage with Marquez, API latency with Prometheus and Grafana, automatic retraining); and the repo link.
- [ ] **Do a timed dry run**

  Rehearse the whole presentation, including any live demo, and have a screenshot backup for each demo step.

**Done when:** the README passes the clean-machine test and the rehearsal fits the time limit.
