from __future__ import annotations

from reset_mlflow_stock_close_class import *  # noqa: F403
from reset_mlflow_stock_close_class import MlflowStockCloseResetter

mlflow_stock_close_resetter = MlflowStockCloseResetter()
_should_reset_experiment = mlflow_stock_close_resetter._should_reset_experiment
_should_delete_registered_model = mlflow_stock_close_resetter._should_delete_registered_model
_stock_close_experiments = mlflow_stock_close_resetter._stock_close_experiments
main = mlflow_stock_close_resetter.main
if __name__ == "__main__":
    main()
