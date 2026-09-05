# UCB Project

Adaptive model selection layer for the stock forecasting project.

This module reads already trained forecasts from MLflow and treats each selected
`ticker + tier + family + model` configuration as a bandit arm. By default the
arms are PECNet configurations, matching the adaptive PECNet paper framing.
MLForecast and StatsForecast can still be included explicitly for baseline
selection experiments.

## Why UCB1-Tuned

Standard UCB mostly relies on historical average reward. That is weak for market
data because model performance can change across regimes. This project therefore
uses the UCB1-Tuned variance term plus either:

- `sliding_window`: use only the latest `L` rewards.
- `discounted`: keep all rewards but downweight older observations.

Replay is executed through a Gymnasium environment. Each environment step
reveals exactly one forecast timestamp, starting from `2026-02-03` by default.
The selector chooses an arm, the environment returns the realized forecast loss,
and the UCB statistics are updated before moving to the next timestamp.

The default reward follows the paper-style negative loss form:

```text
reward = -RMSE(y_t, prediction_t)
```

For a single forecast step this is equivalent to:

```text
reward = -absolute_error
```

Higher reward means the model forecast was closer to the realized close. For a
bounded reward, the CLI also supports:

```text
reward = 1 / (1 + absolute_percentage_error)
```

through `--reward-mode inverse_mape`. Other direct `r = -L` modes are
`negative_mape`, `negative_mae`, and `negative_squared_error`.

## Install

From the repository root:

```bash
cd /Users/batuhansaylam/Documents/Codex/2026-07-04/a/outputs/yahooquery_lakehouse_revamp
source .venv/bin/activate
pip install -e ucb_project
```

If FastAPI is not installed:

```bash
pip install -r ucb_project/requirements.txt
```

## Run A Backtest

```bash
ucb-select backtest \
  --ticker AAPL \
  --tiers tier1,tier2,tier3,tier4,tier5,tier6,tier7,tier8 \
  --recency-mode discounted \
  --discount-factor 0.97 \
  --start-date 2026-02-03 \
  --reward-mode negative_rmse \
  --window-size 60
```

To compare PECNet against the statistical and ML baselines in the same bandit
run, pass all families explicitly:

```bash
ucb-select backtest \
  --ticker AAPL \
  --tiers tier1,tier2,tier3,tier4,tier5,tier6,tier7,tier8 \
  --families mlforecast,statsforecast,pecnet \
  --recency-mode discounted \
  --discount-factor 0.97 \
  --start-date 2026-02-03 \
  --reward-mode negative_rmse \
  --window-size 60
```

## Debug The Selector

This prints each stage: environment, MLflow candidates, evaluation dates, UCB
scores, selected arm, realized reward, and final summary.

```bash
ucb-debug \
  --ticker AAPL \
  --tiers tier1,tier2,tier3,tier4,tier5,tier6,tier7,tier8 \
  --recency-mode discounted \
  --discount-factor 0.97 \
  --start-date 2026-02-03 \
  --reward-mode negative_rmse \
  --window-size 60 \
  --print-every 1
```

## Start API

```bash
ucb-select serve --host 0.0.0.0 --port 8088
```

Endpoints:

- `GET /health`
- `POST /backtest`
- `POST /select`
- `POST /register`
- `POST /pecnet/predict-from-artifact`

## Environment

Defaults match the local project:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
export AWS_ACCESS_KEY_ID=admin
export AWS_SECRET_ACCESS_KEY=admin1234
export AWS_DEFAULT_REGION=us-east-1
```

Inside Docker, `127.0.0.1` is automatically rewritten to
`host.docker.internal`.
