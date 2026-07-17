"""
ai/config/settings.py - Configuration for the AI Trading Engine

RESPONSIBILITY:
Central, typed, dependency-injectable configuration for every AI subsystem.
No hardcoded runtime values elsewhere in the engine.

VERSION: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json
import copy


# ==============================================================================
# SUB-CONFIGS
# ==============================================================================


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    enabled_groups: List[str] = field(
        default_factory=lambda: [
            "price",
            "returns",
            "moving_averages",
            "momentum",
            "volatility",
            "channels",
            "volume",
            "candle_structure",
            "patterns",
            "structure",
            "session",
            "regime",
        ]
    )
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [5, 10, 12, 20, 26, 50, 100])
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    stochastic_k: int = 14
    stochastic_d: int = 3
    williams_period: int = 14
    cci_period: int = 20
    roc_period: int = 12
    momentum_period: int = 10
    donchian_period: int = 20
    keltner_period: int = 20
    keltner_atr_mult: float = 1.5
    supertrend_period: int = 10
    supertrend_mult: float = 3.0
    mfi_period: int = 14
    cmf_period: int = 20
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    fractal_window: int = 2
    swing_lookback: int = 20
    support_resistance_lookback: int = 50
    multi_timeframes: List[str] = field(default_factory=lambda: ["M15", "H1", "H4"])
    correlation_symbols: List[str] = field(default_factory=list)
    correlation_window: int = 50
    dropna: bool = True
    fill_method: str = "ffill"


@dataclass
class LabelConfig:
    """Label generation configuration."""

    methods: List[str] = field(
        default_factory=lambda: [
            "binary_direction",
            "multiclass_direction",
            "future_return",
            "triple_barrier",
        ]
    )
    horizon: int = 5
    horizons: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    binary_threshold: float = 0.0
    multiclass_thresholds: List[float] = field(default_factory=lambda: [-0.002, 0.002])
    take_profit_atr_mult: float = 2.0
    stop_loss_atr_mult: float = 1.0
    volatility_horizon: int = 10
    meta_primary_threshold: float = 0.55
    atr_period: int = 14


@dataclass
class DatasetConfig:
    """Dataset builder configuration."""

    sequence_length: int = 64
    forecast_horizon: int = 5
    stride: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 256
    walk_forward_folds: int = 5
    walk_forward_embargo: int = 5
    shuffle_within_batch: bool = False
    drop_incomplete: bool = True
    max_memory_rows: int = 100_000
    scaling_method: str = "zscore"
    feature_selection_k: Optional[int] = None
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model selection and defaults."""

    model_type: str = "lightgbm"
    task: str = "classification"  # classification | regression
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    lstm_units: int = 64
    lstm_layers: int = 2
    dropout: float = 0.2
    transformer_heads: int = 4
    transformer_dim: int = 64
    ensemble_method: str = "voting"
    ensemble_models: List[str] = field(
        default_factory=lambda: ["random_forest", "lightgbm", "xgboost"]
    )
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    epochs: int = 50
    early_stopping_patience: int = 10
    validation_metric: str = "f1"
    minimize_metric: bool = False
    cross_validation_folds: int = 5
    time_series_splits: int = 5
    use_walk_forward: bool = True
    hyperparameter_search: bool = False
    hyperparameter_trials: int = 30
    checkpoint_every_n: int = 5
    resume_from_checkpoint: bool = True
    mixed_precision: bool = False
    device: str = "auto"  # auto | cpu | cuda
    parallel_jobs: int = 1
    experiment_name: str = "default"
    log_tensorboard: bool = False
    verbose: int = 1


@dataclass
class RiskConfig:
    """Risk management configuration."""

    risk_per_trade: float = 0.01
    max_risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.06
    max_open_trades: int = 5
    max_correlation: float = 0.75
    daily_loss_limit: float = 0.03
    max_drawdown: float = 0.15
    circuit_breaker_loss: float = 0.05  # halt new trades after X% equity loss from peak
    max_positions_per_symbol: int = 1
    max_positions_per_asset_class: int = 3
    position_sizing: str = "fixed_risk"  # fixed_risk | kelly | atr | fixed_lot | volatility
    kelly_fraction: float = 0.25
    atr_stop_mult: float = 1.5
    atr_tp_mult: float = 2.5
    trailing_stop_atr_mult: float = 1.0
    break_even_atr_mult: float = 1.0
    min_confidence: float = 0.55
    min_expected_rr: float = 1.5
    account_currency: str = "USD"
    default_lot_size: float = 0.01
    max_lot_size: float = 10.0
    drawdown_size_scale: bool = True  # shrink size as drawdown approaches max


@dataclass
class ExecutionConfig:
    """Execution and order simulation configuration."""

    default_order_type: str = "market"
    slippage_points: float = 0.5
    commission_per_lot: float = 7.0
    latency_ms: float = 50.0
    partial_fill_enabled: bool = True
    partial_fill_ratio: float = 1.0
    magic_number: int = 260716
    comment_prefix: str = "eTradeAI"
    allow_hedging: bool = False
    close_on_opposite: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and drift detection configuration."""

    enable_latency_tracking: bool = True
    enable_resource_tracking: bool = True
    enable_drift_detection: bool = True
    drift_window: int = 500
    drift_threshold: float = 0.15
    prediction_latency_warn_ms: float = 100.0
    training_memory_warn_mb: float = 4096.0
    alert_on_drift: bool = True


@dataclass
class StorageConfig:
    """Model registry and artifact storage."""

    root_dir: Path = field(default_factory=lambda: Path("ai_artifacts"))
    models_dir: str = "models"
    scalers_dir: str = "scalers"
    experiments_dir: str = "experiments"
    checkpoints_dir: str = "checkpoints"
    predictions_dir: str = "predictions"
    logs_dir: str = "logs"
    compress: bool = True
    keep_last_n_versions: int = 20


@dataclass
class DataDownloadConfig:
    """
    AI-owned market-data acquisition.

    When enabled, the pipeline downloads every configured symbol × timeframe
    itself (MT5 brokers and/or CSV sources) before training or prediction.
    """

    auto_download: bool = True
    min_bars: int = 2000
    years: int = 5
    currency_pairs_only: bool = False
    allow_synthetic_fallback: bool = False  # production path: never invent bars
    require_validated: bool = True  # train only on PASS series by default
    brokers_config: Optional[str] = None  # path to config/brokers.json
    csv_brokers: Dict[str, str] = field(default_factory=dict)  # name -> dir
    include_mt5: bool = True
    include_multi_timeframes: bool = True
    include_correlation_symbols: bool = True
    refresh_interval_seconds: float = 3600.0
    database_path: Optional[str] = None


@dataclass
class ResearchConfig:
    """
    Autonomous quant research engine configuration.

    Continuously expands history, discovers hypotheses, validates strictly,
    paper-trades, and promotes only superior models. Does not invent market data.
    """

    enabled: bool = True
    cycle_interval_seconds: float = 86_400.0
    sleep_seconds: float = 0.0
    max_cycles: Optional[int] = None
    markets: List[str] = field(
        default_factory=lambda: ["FOREX", "METALS", "INDICES", "CRYPTO", "ENERGY"]
    )
    history_start: str = "2010-01-01"
    repair_failed_series: bool = True
    download_ticks: bool = True
    tick_lookback_days: int = 7
    require_validated: bool = True
    skip_collect: bool = False
    allow_synthetic: bool = False
    model_candidates: List[str] = field(
        default_factory=lambda: ["random_forest", "lightgbm", "xgboost"]
    )
    candle_limit: int = 5000
    compare_models: bool = True
    primary_metric: str = "test_f1"
    metric_minimize: bool = False
    min_improvement: float = 0.005
    register_only_improvements: bool = True
    run_feature_discovery: bool = True
    run_strict_validation: bool = True
    run_backtest: bool = True
    run_paper_trade: bool = True
    paper_equity: float = 10_000.0
    detect_drift: bool = True
    generate_hypotheses: bool = True
    run_self_improve: bool = True
    run_production_gate: bool = True
    build_dashboard: bool = True
    # Production readiness thresholds
    min_paper_trades: int = 50
    min_paper_days: float = 14.0
    min_live_sharpe: float = 0.5
    max_live_drawdown: float = 0.20
    min_live_profit_factor: float = 1.2
    # Strict validation thresholds
    min_val_score: float = 0.50
    min_oos_score: float = 0.50
    min_walk_forward_score: float = 0.50
    max_mc_ruin_prob: float = 0.30
    min_backtest_trades: int = 5
    reports_dir: str = "research_cycles"
    state_filename: str = "research_state.json"
    dashboard_dir: str = "dashboards"


# ==============================================================================
# ROOT CONFIG
# ==============================================================================


@dataclass
class AIConfig:
    """
    Root configuration object injected into every AI component.

    All nested configs are mutable dataclasses so callers can override
    specific fields without rebuilding the entire tree.
    """

    project_name: str = "eTradeAI"
    version: str = "1.0.0"
    symbols: List[str] = field(default_factory=lambda: ["EURUSD"])
    timeframes: List[str] = field(default_factory=lambda: ["M15", "H1"])
    primary_timeframe: str = "M15"
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    data: DataDownloadConfig = field(default_factory=DataDownloadConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    random_seed: int = 42
    timezone: str = "UTC"

    def ensure_directories(self) -> None:
        """Create artifact directories if they do not exist."""
        root = Path(self.storage.root_dir)
        for sub in (
            self.storage.models_dir,
            self.storage.scalers_dir,
            self.storage.experiments_dir,
            self.storage.checkpoints_dir,
            self.storage.predictions_dir,
            self.storage.logs_dir,
        ):
            (root / sub).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a plain dictionary."""
        data = asdict(self)
        data["storage"]["root_dir"] = str(self.storage.root_dir)
        return data

    def to_json(self, path: Path | str) -> None:
        """Write configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIConfig":
        """Build configuration from a nested dictionary."""
        payload = copy.deepcopy(data)
        nested_map = {
            "features": FeatureConfig,
            "labels": LabelConfig,
            "datasets": DatasetConfig,
            "model": ModelConfig,
            "training": TrainingConfig,
            "risk": RiskConfig,
            "execution": ExecutionConfig,
            "monitoring": MonitoringConfig,
            "storage": StorageConfig,
            "data": DataDownloadConfig,
            "research": ResearchConfig,
        }
        for key, conf_cls in nested_map.items():
            if key in payload and isinstance(payload[key], dict):
                if key == "storage" and "root_dir" in payload[key]:
                    payload[key]["root_dir"] = Path(payload[key]["root_dir"])
                payload[key] = conf_cls(**payload[key])
        return cls(**payload)

    @classmethod
    def from_json(cls, path: Path | str) -> "AIConfig":
        """Load configuration from a JSON file."""
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def copy(self) -> "AIConfig":
        """Deep copy this configuration."""
        return self.from_dict(self.to_dict())


def create_ai_config(**overrides: Any) -> AIConfig:
    """
    Factory for AIConfig with optional top-level overrides.

    Nested overrides can be passed as dicts, e.g. features={"rsi_period": 21}.
    """
    config = AIConfig()
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise AttributeError(f"Unknown AIConfig field: {key}")
        current = getattr(config, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            for nested_key, nested_value in value.items():
                setattr(current, nested_key, nested_value)
        else:
            setattr(config, key, value)
    return config
