"""
Defines the imports for the package

Shows the Public API for running the correlation engine
"""
from .engine import run_correlation_engine
from .preprocessing import enforce_stationary, isStationary
from .analyzer import chunkify, compute_lagged_correlations, aggregate_lags
from .config_generator import generate_json_config

_all_ = [
    run_correlation_engine,
    enforce_stationary,
    isStationary,
    chunkify,
    compute_lagged_correlations,
    aggregate_lags,
    generate_json_config
]