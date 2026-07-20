from __future__ import annotations

from .ticker_logs_class import *  # noqa: F403
from .ticker_logs_class import PecnetTickerLogWriter

pecnet_ticker_log_writer = PecnetTickerLogWriter()
log_ticker_selection_metrics = pecnet_ticker_log_writer.log_ticker_selection_metrics
log_ticker_model_metrics = pecnet_ticker_log_writer.log_ticker_model_metrics
log_ticker_model_artifact = pecnet_ticker_log_writer.log_ticker_model_artifact
