"""
database/repositories/research_repository.py - Persistent research archive access.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from database.core.connection import DatabaseManager
from database.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str)


class ResearchRepository(BaseRepository):
    """CRUD helpers for the autonomous research database."""

    TABLE = "research_experiments"
    MODEL = dict

    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager)
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        *,
        symbol: str,
        timeframe: str,
        cycle_id: str | None = None,
        hypothesis_id: str | None = None,
        objective_metric: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        now = _now()
        payload = {
            "experiment_uuid": uuid4().hex,
            "cycle_id": cycle_id,
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "hypothesis_id": hypothesis_id,
            "status": "running",
            "objective_metric": objective_metric,
            "objective_value": None,
            "metadata": _json(metadata),
            "started_at": now,
            "finished_at": None,
            "created_at": now,
        }
        experiment_id = self._insert("research_experiments", payload)
        payload["experiment_id"] = experiment_id
        return payload

    def finish_experiment(
        self,
        experiment_id: int,
        *,
        status: str,
        objective_value: float | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        fields = {
            "status": status,
            "finished_at": _now(),
            "objective_value": objective_value,
        }
        if metadata is not None:
            fields["metadata"] = _json(metadata)
        self._update("research_experiments", fields, "experiment_id = ?", (experiment_id,))

    def record_dataset(
        self,
        experiment_id: int,
        *,
        symbol: str,
        timeframe: str,
        n_rows: int,
        n_features: int,
        train_rows: int,
        val_rows: int,
        test_rows: int,
        first_timestamp: str | None,
        last_timestamp: str | None,
        feature_hash: str | None,
        label_method: str | None,
        metadata: Dict[str, Any] | None = None,
    ) -> int:
        payload = {
            "dataset_uuid": uuid4().hex,
            "experiment_id": experiment_id,
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "n_rows": n_rows,
            "n_features": n_features,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "feature_hash": feature_hash,
            "label_method": label_method,
            "metadata": _json(metadata),
            "created_at": _now(),
        }
        return int(self._insert("research_datasets", payload))

    def record_features(
        self,
        experiment_id: int,
        features: Sequence[Dict[str, Any]],
    ) -> int:
        count = 0
        now = _now()
        for feat in features:
            payload = {
                "feature_uuid": uuid4().hex,
                "experiment_id": experiment_id,
                "name": str(feat.get("name")),
                "group_name": feat.get("group_name"),
                "source": feat.get("source"),
                "importance": feat.get("importance"),
                "kept": 1 if feat.get("kept", True) else 0,
                "score": feat.get("score"),
                "metadata": _json(feat.get("metadata")),
                "created_at": now,
            }
            self._insert("research_features", payload)
            count += 1
        return count

    def record_model(
        self,
        experiment_id: int | None,
        *,
        name: str,
        model_type: str,
        symbol: str,
        timeframe: str,
        version: str | None = None,
        artifact_path: str | None = None,
        metrics: Dict[str, Any] | None = None,
        hyperparameters: Dict[str, Any] | None = None,
        is_champion: bool = False,
        is_deployed: bool = False,
        status: str = "candidate",
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload = {
            "model_uuid": uuid4().hex,
            "experiment_id": experiment_id,
            "name": name,
            "version": version,
            "model_type": model_type,
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "artifact_path": artifact_path,
            "metrics": _json(metrics),
            "hyperparameters": _json(hyperparameters),
            "is_champion": 1 if is_champion else 0,
            "is_deployed": 1 if is_deployed else 0,
            "status": status,
            "metadata": _json(metadata),
            "created_at": _now(),
        }
        model_id = self._insert("research_models", payload)
        payload["model_id"] = model_id
        if hyperparameters:
            for key, value in hyperparameters.items():
                self._insert(
                    "research_hyperparameters",
                    {
                        "model_id": model_id,
                        "key": str(key),
                        "value": json.dumps(value, default=str),
                        "created_at": _now(),
                    },
                )
        return payload

    def set_champion(self, model_id: int, symbol: str, timeframe: str) -> None:
        self._execute(
            """
            UPDATE research_models
            SET is_champion = 0
            WHERE symbol = ? AND timeframe = ? AND is_champion = 1
            """,
            (symbol.upper(), timeframe.upper()),
        )
        self._execute(
            "UPDATE research_models SET is_champion = 1, status = 'champion' WHERE model_id = ?",
            (model_id,),
        )
        self._commit()

    def record_backtest(
        self,
        *,
        experiment_id: int | None,
        model_id: int | None,
        symbol: str,
        timeframe: str,
        n_trades: int,
        metrics: Dict[str, Any],
        spread_points: float,
        commission_per_lot: float,
        slippage_points: float,
        metadata: Dict[str, Any] | None = None,
    ) -> int:
        payload = {
            "backtest_uuid": uuid4().hex,
            "experiment_id": experiment_id,
            "model_id": model_id,
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "n_trades": n_trades,
            "total_return": metrics.get("total_return"),
            "sharpe": metrics.get("sharpe"),
            "profit_factor": metrics.get("profit_factor"),
            "max_drawdown": metrics.get("max_drawdown"),
            "win_rate": metrics.get("win_rate"),
            "spread_points": spread_points,
            "commission_per_lot": commission_per_lot,
            "slippage_points": slippage_points,
            "metrics": _json(metrics),
            "metadata": _json(metadata),
            "created_at": _now(),
        }
        return int(self._insert("research_backtests", payload))

    def record_validation(
        self,
        *,
        experiment_id: int | None,
        model_id: int | None,
        stage: str,
        passed: bool,
        metrics: Dict[str, Any] | None = None,
        details: Dict[str, Any] | None = None,
    ) -> int:
        payload = {
            "validation_uuid": uuid4().hex,
            "experiment_id": experiment_id,
            "model_id": model_id,
            "stage": stage,
            "passed": 1 if passed else 0,
            "metrics": _json(metrics),
            "details": _json(details),
            "created_at": _now(),
        }
        return int(self._insert("research_validations", payload))

    def record_deployment(
        self,
        *,
        model_id: int,
        symbol: str,
        timeframe: str,
        environment: str,
        status: str,
        previous_model_id: int | None = None,
        reason: str | None = None,
        metrics: Dict[str, Any] | None = None,
    ) -> int:
        if environment == "paper" or environment == "live":
            self._execute(
                """
                UPDATE research_deployments
                SET status = 'retired', retired_at = ?
                WHERE symbol = ? AND timeframe = ? AND environment = ? AND status = 'active'
                """,
                (_now(), symbol.upper(), timeframe.upper(), environment),
            )
        if environment in {"paper", "live"} and status == "active":
            self._execute(
                """
                UPDATE research_models
                SET is_deployed = CASE WHEN model_id = ? THEN 1 ELSE 0 END
                WHERE symbol = ? AND timeframe = ?
                """,
                (model_id, symbol.upper(), timeframe.upper()),
            )
        payload = {
            "deployment_uuid": uuid4().hex,
            "model_id": model_id,
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "environment": environment,
            "status": status,
            "previous_model_id": previous_model_id,
            "reason": reason,
            "metrics": _json(metrics),
            "created_at": _now(),
            "retired_at": None,
        }
        dep_id = int(self._insert("research_deployments", payload))
        self._commit()
        return dep_id

    def record_prediction(
        self,
        *,
        model_id: int | None,
        symbol: str,
        timeframe: str,
        timestamp: str,
        prediction: float | None,
        signal: str | None,
        confidence: float | None,
        explanation: str | None = None,
        feature_importance: Dict[str, Any] | None = None,
        features_snapshot: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload = {
            "prediction_uuid": uuid4().hex,
            "model_id": model_id,
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "timestamp": timestamp,
            "prediction": prediction,
            "signal": signal,
            "confidence": confidence,
            "actual_outcome": None,
            "pnl": None,
            "drawdown": None,
            "explanation": explanation,
            "feature_importance": _json(feature_importance),
            "features_snapshot": _json(features_snapshot),
            "resolved": 0,
            "metadata": _json(metadata),
            "created_at": _now(),
            "resolved_at": None,
        }
        prediction_id = self._insert("research_predictions", payload)
        payload["prediction_id"] = prediction_id
        return payload

    def resolve_prediction(
        self,
        prediction_id: int,
        *,
        actual_outcome: float,
        pnl: float,
        drawdown: float | None = None,
    ) -> None:
        self._update(
            "research_predictions",
            {
                "actual_outcome": actual_outcome,
                "pnl": pnl,
                "drawdown": drawdown,
                "resolved": 1,
                "resolved_at": _now(),
            },
            "prediction_id = ?",
            (prediction_id,),
        )

    def list_unresolved_predictions(
        self,
        symbol: str | None = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM research_predictions WHERE resolved = 0"
        params: List[Any] = []
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol.upper())
        sql += " ORDER BY timestamp ASC LIMIT ?"
        params.append(int(limit))
        return self._fetch_all(sql, tuple(params))

    def record_paper_trade(self, **fields: Any) -> int:
        payload = {
            "paper_uuid": uuid4().hex,
            "model_id": fields.get("model_id"),
            "symbol": str(fields["symbol"]).upper(),
            "timeframe": str(fields["timeframe"]).upper(),
            "side": fields["side"],
            "quantity": fields.get("quantity"),
            "entry_time": fields.get("entry_time"),
            "entry_price": fields.get("entry_price"),
            "exit_time": fields.get("exit_time"),
            "exit_price": fields.get("exit_price"),
            "pnl": fields.get("pnl"),
            "return_pct": fields.get("return_pct"),
            "drawdown": fields.get("drawdown"),
            "status": fields.get("status", "open"),
            "metadata": _json(fields.get("metadata")),
            "created_at": _now(),
        }
        return int(self._insert("research_paper_trades", payload))

    def record_hypothesis(self, hyp: Dict[str, Any]) -> int:
        payload = {
            "hypothesis_uuid": hyp.get("id") or uuid4().hex,
            "symbol": (hyp.get("symbol") or "").upper() or None,
            "kind": hyp.get("kind"),
            "priority": hyp.get("priority"),
            "rationale": hyp.get("rationale"),
            "actions": _json(hyp.get("actions") or []),
            "status": hyp.get("status", "pending"),
            "experiment_id": hyp.get("experiment_id"),
            "metadata": _json(hyp.get("metadata")),
            "created_at": _now(),
            "resolved_at": None,
        }
        return int(self._insert("research_hypotheses", payload))

    def record_monitoring_snapshot(self, **fields: Any) -> int:
        payload = {
            "snapshot_uuid": uuid4().hex,
            "symbol": (fields.get("symbol") or "").upper() or None,
            "timeframe": (fields.get("timeframe") or "").upper() or None,
            "model_id": fields.get("model_id"),
            "accuracy": fields.get("accuracy"),
            "sharpe": fields.get("sharpe"),
            "profit_factor": fields.get("profit_factor"),
            "max_drawdown": fields.get("max_drawdown"),
            "drift_score": fields.get("drift_score"),
            "model_age_hours": fields.get("model_age_hours"),
            "calibration_error": fields.get("calibration_error"),
            "feature_importance": _json(fields.get("feature_importance")),
            "metrics": _json(fields.get("metrics")),
            "created_at": _now(),
        }
        return int(self._insert("research_monitoring_snapshots", payload))

    def upsert_production_gate(self, **fields: Any) -> int:
        symbol = str(fields["symbol"]).upper()
        timeframe = str(fields["timeframe"]).upper()
        existing = self._fetch_one(
            "SELECT gate_id FROM research_production_gates WHERE symbol=? AND timeframe=? ORDER BY gate_id DESC LIMIT 1",
            (symbol, timeframe),
        )
        payload = {
            "model_id": fields.get("model_id"),
            "paper_trades": fields.get("paper_trades", 0),
            "paper_days": fields.get("paper_days", 0),
            "sharpe": fields.get("sharpe"),
            "max_drawdown": fields.get("max_drawdown"),
            "profit_factor": fields.get("profit_factor"),
            "passed": 1 if fields.get("passed") else 0,
            "live_enabled": 1 if fields.get("live_enabled") else 0,
            "thresholds": _json(fields.get("thresholds")),
            "details": _json(fields.get("details")),
            "updated_at": _now(),
        }
        if existing:
            gate_id = int(existing["gate_id"] if isinstance(existing, dict) else existing[0])
            self._update("research_production_gates", payload, "gate_id = ?", (gate_id,))
            return gate_id
        payload.update(
            {
                "gate_uuid": uuid4().hex,
                "symbol": symbol,
                "timeframe": timeframe,
                "created_at": _now(),
            }
        )
        return int(self._insert("research_production_gates", payload))

    def get_champion_model(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        return self._fetch_one(
            """
            SELECT * FROM research_models
            WHERE symbol=? AND timeframe=? AND is_champion=1
            ORDER BY model_id DESC LIMIT 1
            """,
            (symbol.upper(), timeframe.upper()),
        )

    def paper_trade_stats(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        row = self._fetch_one(
            """
            SELECT
                COUNT(*) AS n_trades,
                AVG(pnl) AS avg_pnl,
                SUM(pnl) AS total_pnl,
                MIN(pnl) AS worst_pnl,
                AVG(return_pct) AS avg_return
            FROM research_paper_trades
            WHERE symbol=? AND timeframe=? AND status='closed'
            """,
            (symbol.upper(), timeframe.upper()),
        )
        pred = self._fetch_one(
            """
            SELECT
                COUNT(*) AS n_preds,
                AVG(CASE WHEN actual_outcome IS NOT NULL AND (
                    (prediction > 0 AND actual_outcome > 0) OR
                    (prediction < 0 AND actual_outcome < 0)
                ) THEN 1.0 ELSE 0.0 END) AS accuracy,
                SUM(pnl) AS pred_pnl,
                MIN(drawdown) AS min_drawdown
            FROM research_predictions
            WHERE symbol=? AND timeframe=? AND resolved=1
            """,
            (symbol.upper(), timeframe.upper()),
        )
        return {
            "trades": dict(row) if row else {},
            "predictions": dict(pred) if pred else {},
        }

    def latest_monitoring(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._fetch_all(
            "SELECT * FROM research_monitoring_snapshots ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _insert(self, table: str, payload: Dict[str, Any]) -> int:
        keys = list(payload.keys())
        placeholders = ", ".join("?" for _ in keys)
        cols = ", ".join(keys)
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        cur = self._execute(sql, tuple(payload[k] for k in keys))
        self._commit()
        last_id = getattr(cur, "lastrowid", None)
        if last_id is None:
            row = self._fetch_one("SELECT last_insert_rowid() AS id")
            last_id = row["id"] if row else 0
        return int(last_id or 0)

    def _update(self, table: str, fields: Dict[str, Any], where: str, params: tuple) -> None:
        assignments = ", ".join(f"{k}=?" for k in fields)
        sql = f"UPDATE {table} SET {assignments} WHERE {where}"
        self._execute(sql, tuple(fields.values()) + params)
        self._commit()

    def _execute(self, sql: str, params: tuple = ()):
        if hasattr(self, "adapter") and self.adapter is not None:
            return self.adapter.execute(sql, params)
        if hasattr(self.db, "get_adapter"):
            return self.db.get_adapter().execute(sql, params)
        return self.db.execute(sql, params)

    def _commit(self) -> None:
        if hasattr(self, "adapter") and self.adapter is not None:
            self.adapter.commit()
        elif hasattr(self.db, "get_adapter"):
            self.db.get_adapter().commit()
        elif hasattr(self.db, "commit"):
            self.db.commit()

    def _fetch_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        if hasattr(self.db, "fetch_one"):
            row = self.db.fetch_one(sql, params)
        else:
            cur = self._execute(sql, params)
            row = cur.fetchone()
        if row is None:
            return None
        return dict(row) if hasattr(row, "keys") else None

    def _fetch_all(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        if hasattr(self.db, "fetch_all"):
            rows = self.db.fetch_all(sql, params)
        else:
            cur = self._execute(sql, params)
            rows = cur.fetchall()
        return [dict(r) for r in rows if hasattr(r, "keys")]
