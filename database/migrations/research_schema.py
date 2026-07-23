"""
database/migrations/research_schema.py - Persistent autonomous research archive.

Stores experiments, models, datasets, features, hyperparameters, backtests,
validation reports, deployment history, and paper-trading journals.
Nothing in the research loop should be lost.
"""

from __future__ import annotations

from typing import Any, List


RESEARCH_SCHEMA_SQL: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS research_experiments (
        experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_uuid TEXT UNIQUE NOT NULL,
        cycle_id TEXT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        hypothesis_id TEXT,
        status TEXT NOT NULL DEFAULT 'running',
        objective_metric TEXT,
        objective_value REAL,
        metadata TEXT DEFAULT '{}',
        started_at TEXT NOT NULL,
        finished_at TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_datasets (
        dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_uuid TEXT UNIQUE NOT NULL,
        experiment_id INTEGER,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        n_rows INTEGER,
        n_features INTEGER,
        train_rows INTEGER,
        val_rows INTEGER,
        test_rows INTEGER,
        first_timestamp TEXT,
        last_timestamp TEXT,
        feature_hash TEXT,
        label_method TEXT,
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(experiment_id) REFERENCES research_experiments(experiment_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_features (
        feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature_uuid TEXT UNIQUE NOT NULL,
        experiment_id INTEGER,
        name TEXT NOT NULL,
        group_name TEXT,
        source TEXT,
        importance REAL,
        kept INTEGER DEFAULT 1,
        score REAL,
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(experiment_id) REFERENCES research_experiments(experiment_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_models (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_uuid TEXT UNIQUE NOT NULL,
        experiment_id INTEGER,
        name TEXT NOT NULL,
        version TEXT,
        model_type TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        artifact_path TEXT,
        metrics TEXT DEFAULT '{}',
        hyperparameters TEXT DEFAULT '{}',
        is_champion INTEGER DEFAULT 0,
        is_deployed INTEGER DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'candidate',
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(experiment_id) REFERENCES research_experiments(experiment_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_hyperparameters (
        hyperparam_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER NOT NULL,
        key TEXT NOT NULL,
        value TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(model_id) REFERENCES research_models(model_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_backtests (
        backtest_id INTEGER PRIMARY KEY AUTOINCREMENT,
        backtest_uuid TEXT UNIQUE NOT NULL,
        experiment_id INTEGER,
        model_id INTEGER,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        n_trades INTEGER DEFAULT 0,
        total_return REAL,
        sharpe REAL,
        profit_factor REAL,
        max_drawdown REAL,
        win_rate REAL,
        spread_points REAL,
        commission_per_lot REAL,
        slippage_points REAL,
        metrics TEXT DEFAULT '{}',
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(experiment_id) REFERENCES research_experiments(experiment_id),
        FOREIGN KEY(model_id) REFERENCES research_models(model_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_validations (
        validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        validation_uuid TEXT UNIQUE NOT NULL,
        experiment_id INTEGER,
        model_id INTEGER,
        stage TEXT NOT NULL,
        passed INTEGER NOT NULL,
        metrics TEXT DEFAULT '{}',
        details TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(experiment_id) REFERENCES research_experiments(experiment_id),
        FOREIGN KEY(model_id) REFERENCES research_models(model_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_deployments (
        deployment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        deployment_uuid TEXT UNIQUE NOT NULL,
        model_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        environment TEXT NOT NULL,
        status TEXT NOT NULL,
        previous_model_id INTEGER,
        reason TEXT,
        metrics TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        retired_at TEXT,
        FOREIGN KEY(model_id) REFERENCES research_models(model_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_paper_trades (
        paper_trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_uuid TEXT UNIQUE NOT NULL,
        model_id INTEGER,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        side TEXT NOT NULL,
        quantity REAL,
        entry_time TEXT,
        entry_price REAL,
        exit_time TEXT,
        exit_price REAL,
        pnl REAL,
        return_pct REAL,
        drawdown REAL,
        status TEXT NOT NULL DEFAULT 'open',
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(model_id) REFERENCES research_models(model_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_uuid TEXT UNIQUE NOT NULL,
        model_id INTEGER,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        prediction REAL,
        signal TEXT,
        confidence REAL,
        actual_outcome REAL,
        pnl REAL,
        drawdown REAL,
        explanation TEXT,
        feature_importance TEXT DEFAULT '{}',
        features_snapshot TEXT DEFAULT '{}',
        resolved INTEGER DEFAULT 0,
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        resolved_at TEXT,
        FOREIGN KEY(model_id) REFERENCES research_models(model_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_hypotheses (
        hypothesis_id INTEGER PRIMARY KEY AUTOINCREMENT,
        hypothesis_uuid TEXT UNIQUE NOT NULL,
        symbol TEXT,
        kind TEXT NOT NULL,
        priority REAL,
        rationale TEXT,
        actions TEXT DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'pending',
        experiment_id INTEGER,
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        resolved_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_monitoring_snapshots (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_uuid TEXT UNIQUE NOT NULL,
        symbol TEXT,
        timeframe TEXT,
        model_id INTEGER,
        accuracy REAL,
        sharpe REAL,
        profit_factor REAL,
        max_drawdown REAL,
        drift_score REAL,
        model_age_hours REAL,
        calibration_error REAL,
        feature_importance TEXT DEFAULT '{}',
        metrics TEXT DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_production_gates (
        gate_id INTEGER PRIMARY KEY AUTOINCREMENT,
        gate_uuid TEXT UNIQUE NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        model_id INTEGER,
        paper_trades INTEGER DEFAULT 0,
        paper_days REAL DEFAULT 0,
        sharpe REAL,
        max_drawdown REAL,
        profit_factor REAL,
        passed INTEGER NOT NULL DEFAULT 0,
        live_enabled INTEGER NOT NULL DEFAULT 0,
        thresholds TEXT DEFAULT '{}',
        details TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tick_sync_status (
        market_id INTEGER NOT NULL,
        status TEXT NOT NULL,
        last_synced TEXT,
        last_tick_time TEXT,
        ticks_count INTEGER DEFAULT 0,
        error_message TEXT,
        PRIMARY KEY (market_id)
    )
    """,
]


def apply_research_schema(db: Any) -> None:
    """Create research archive tables."""
    for statement in RESEARCH_SCHEMA_SQL:
        _execute(db, statement)
    _commit(db)
    _create_research_indexes(db)


def _create_research_indexes(db: Any) -> None:
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_research_exp_symbol ON research_experiments(symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_research_exp_cycle ON research_experiments(cycle_id)",
        "CREATE INDEX IF NOT EXISTS idx_research_models_symbol ON research_models(symbol, timeframe, is_champion)",
        "CREATE INDEX IF NOT EXISTS idx_research_models_deployed ON research_models(is_deployed)",
        "CREATE INDEX IF NOT EXISTS idx_research_pred_symbol_ts ON research_predictions(symbol, timeframe, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_research_pred_unresolved ON research_predictions(resolved)",
        "CREATE INDEX IF NOT EXISTS idx_research_paper_symbol ON research_paper_trades(symbol, status)",
        "CREATE INDEX IF NOT EXISTS idx_research_backtests_model ON research_backtests(model_id)",
        "CREATE INDEX IF NOT EXISTS idx_research_validations_model ON research_validations(model_id, stage)",
        "CREATE INDEX IF NOT EXISTS idx_research_hypotheses_status ON research_hypotheses(status, priority)",
        "CREATE INDEX IF NOT EXISTS idx_research_monitor_created ON research_monitoring_snapshots(created_at)",
    ]
    for sql in indexes:
        _execute(db, sql)
    _commit(db)


def _execute(db: Any, sql: str, params: tuple = ()) -> None:
    if hasattr(db, "get_adapter"):
        db.get_adapter().execute(sql, params)
        return
    if hasattr(db, "execute"):
        db.execute(sql, params)
        return
    raise TypeError(f"Unsupported db object: {type(db)!r}")


def _commit(db: Any) -> None:
    if hasattr(db, "commit"):
        db.commit()
    elif hasattr(db, "get_adapter"):
        db.get_adapter().commit()
