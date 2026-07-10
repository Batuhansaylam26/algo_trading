import os

import optuna
from mlforecast.auto import (
    AutoCatboost,
    AutoLightGBM,
    AutoModel,
    AutoRandomForest,
    AutoRidge,
    AutoXGBoost,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from .runtime import cpu_count_from_env


def _worker_count() -> int:
    return cpu_count_from_env("MODEL_N_JOBS")


def _max_estimators() -> int:
    return max(50, int(os.getenv("MODEL_MAX_ESTIMATORS", "300")))


def _verbose() -> bool:
    return os.getenv("MLFORECAST_VERBOSE", "1").lower() in {"1", "true", "yes"}


def ridge_config(trial: optuna.Trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
    }


def random_forest_config(trial: optuna.Trial) -> dict:
    max_estimators = _max_estimators()
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, max_estimators),
        "max_depth": trial.suggest_int("max_depth", 2, 24),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 25),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        "random_state": 26,
        "n_jobs": _worker_count(),
        "verbose": 1 if _verbose() else 0,
    }


def lightgbm_config(trial: optuna.Trial) -> dict:
    max_estimators = _max_estimators()
    max_depth = trial.suggest_int("max_depth", 2, 10)
    max_leaves = min(64, (2**max_depth) - 1)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, max_estimators),
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 2, max_leaves),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "objective": "regression",
        "force_col_wise": True,
        "random_state": 26,
        "n_jobs": _worker_count(),
        "verbosity": -1,
    }


def xgboost_config(trial: optuna.Trial) -> dict:
    max_estimators = _max_estimators()
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, max_estimators),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "objective": "reg:squarederror",
        "random_state": 26,
        "n_jobs": _worker_count(),
        "verbosity": 1 if _verbose() else 0,
    }


def catboost_config(trial: optuna.Trial) -> dict:
    max_estimators = _max_estimators()
    return {
        "iterations": trial.suggest_int("iterations", 50, max_estimators),
        "depth": trial.suggest_int("depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "loss_function": "RMSE",
        "random_seed": 26,
        "thread_count": _worker_count(),
        "verbose": _verbose(),
        "allow_writing_files": False,
    }


def init_config(trial: optuna.Trial) -> dict:
    lags_name = trial.suggest_categorical(
        "lags",
        ["1", "1_7", "1_7_20", "1_7_20_60"],
    )
    lags_map = {
        "1": [1],
        "1_7": [1, 7],
        "1_7_20": [1, 7, 20],
        "1_7_20_60": [1, 7, 20, 60],
    }
    return {"lags": lags_map[lags_name]}


def fit_config(trial: optuna.Trial) -> dict:
    return {"static_features": []}


def build_auto_models() -> dict:
    return {
        "Ridge": AutoRidge(config=ridge_config),
        "RandomForest": AutoRandomForest(config=random_forest_config),
        "LightGBM": AutoLightGBM(config=lightgbm_config),
        "XGBoost": AutoXGBoost(config=xgboost_config),
        "CatBoost": AutoCatboost(config=catboost_config),
    }


def build_custom_auto_models() -> dict:
    return {
        "Ridge": AutoModel(model=Ridge(), config=ridge_config),
        "RandomForest": AutoModel(
            model=RandomForestRegressor(),
            config=random_forest_config,
        ),
        "LightGBM": AutoLightGBM(config=lightgbm_config),
        "XGBoost": AutoXGBoost(config=xgboost_config),
        "CatBoost": AutoCatboost(config=catboost_config),
    }
