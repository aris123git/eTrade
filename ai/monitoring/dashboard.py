"""
ai/monitoring/dashboard.py - Institutional monitoring dashboard generator.

Produces HTML + JSON dashboards from the research database:
prediction accuracy, Sharpe, Profit Factor, drawdown, drift, model age,
confidence calibration, feature importance evolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from database.repositories.research_repository import ResearchRepository


@dataclass
class DashboardArtifacts:
    html_path: Path
    json_path: Path
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "html_path": str(self.html_path),
            "json_path": str(self.json_path),
            "metrics": self.payload.get("summary", {}),
        }


class InstitutionalDashboard:
    """Build monitoring artifacts from persistent research data."""

    def __init__(self, research_repo: ResearchRepository, output_dir: Path | str):
        self.repo = research_repo
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self, *, symbols: Optional[List[str]] = None) -> DashboardArtifacts:
        snapshots = self.repo.latest_monitoring(limit=100)
        champions = self._champions(symbols)
        paper = {
            sym: self.repo.paper_trade_stats(sym, tf)
            for sym, tf in ((c["symbol"], c["timeframe"]) for c in champions)
        } if champions else {}
        importance_evolution = self._importance_evolution()
        calibration = self._calibration()
        summary = self._summary(champions, paper, snapshots, calibration)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "champions": champions,
            "paper": paper,
            "snapshots": snapshots,
            "importance_evolution": importance_evolution,
            "calibration": calibration,
        }
        json_path = self.output_dir / "institutional_dashboard.json"
        html_path = self.output_dir / "institutional_dashboard.html"
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        html_path.write_text(self._render_html(payload), encoding="utf-8")
        return DashboardArtifacts(html_path=html_path, json_path=json_path, payload=payload)

    def _champions(self, symbols: Optional[List[str]]) -> List[Dict[str, Any]]:
        rows = self.repo._fetch_all(
            """
            SELECT * FROM research_models
            WHERE is_champion = 1
            ORDER BY symbol, timeframe, model_id DESC
            """,
        )
        if symbols:
            upper = {s.upper() for s in symbols}
            rows = [r for r in rows if r.get("symbol") in upper]
        out = []
        for row in rows:
            metrics = row.get("metrics")
            if isinstance(metrics, str):
                try:
                    metrics = json.loads(metrics)
                except Exception:
                    metrics = {}
            age_hours = None
            created = row.get("created_at")
            if created:
                try:
                    created_dt = datetime.fromisoformat(str(created).replace("Z", ""))
                    age_hours = (datetime.utcnow() - created_dt).total_seconds() / 3600.0
                except Exception:
                    age_hours = None
            out.append(
                {
                    "model_id": row.get("model_id"),
                    "symbol": row.get("symbol"),
                    "timeframe": row.get("timeframe"),
                    "model_type": row.get("model_type"),
                    "version": row.get("version"),
                    "metrics": metrics or {},
                    "model_age_hours": age_hours,
                }
            )
        return out

    def _importance_evolution(self) -> List[Dict[str, Any]]:
        rows = self.repo._fetch_all(
            """
            SELECT created_at, symbol, timeframe, feature_importance
            FROM research_monitoring_snapshots
            WHERE feature_importance IS NOT NULL AND feature_importance != '{}'
            ORDER BY created_at DESC
            LIMIT 40
            """
        )
        out = []
        for row in rows:
            imp = row.get("feature_importance")
            if isinstance(imp, str):
                try:
                    imp = json.loads(imp)
                except Exception:
                    imp = {}
            out.append(
                {
                    "created_at": row.get("created_at"),
                    "symbol": row.get("symbol"),
                    "timeframe": row.get("timeframe"),
                    "top_features": dict(list((imp or {}).items())[:10]),
                }
            )
        return out

    def _calibration(self) -> Dict[str, Any]:
        rows = self.repo._fetch_all(
            """
            SELECT confidence, actual_outcome, prediction
            FROM research_predictions
            WHERE resolved = 1 AND confidence IS NOT NULL AND actual_outcome IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 2000
            """
        )
        if not rows:
            return {"n": 0, "calibration_error": None, "bins": []}
        bins = {i: {"conf_sum": 0.0, "hit_sum": 0.0, "n": 0} for i in range(10)}
        for row in rows:
            conf = float(row.get("confidence") or 0.0)
            hit = 1.0 if float(row.get("actual_outcome") or 0.0) > 0 else 0.0
            idx = min(9, max(0, int(conf * 10)))
            bins[idx]["conf_sum"] += conf
            bins[idx]["hit_sum"] += hit
            bins[idx]["n"] += 1
        rendered = []
        errors = []
        for idx, bucket in bins.items():
            if bucket["n"] == 0:
                continue
            mean_conf = bucket["conf_sum"] / bucket["n"]
            mean_hit = bucket["hit_sum"] / bucket["n"]
            errors.append(abs(mean_conf - mean_hit))
            rendered.append(
                {
                    "bin": idx,
                    "n": bucket["n"],
                    "mean_confidence": mean_conf,
                    "empirical_hit_rate": mean_hit,
                }
            )
        return {
            "n": len(rows),
            "calibration_error": float(sum(errors) / len(errors)) if errors else None,
            "bins": rendered,
        }

    def _summary(
        self,
        champions: List[Dict[str, Any]],
        paper: Dict[str, Any],
        snapshots: List[Dict[str, Any]],
        calibration: Dict[str, Any],
    ) -> Dict[str, Any]:
        latest = snapshots[0] if snapshots else {}
        return {
            "n_champions": len(champions),
            "latest_accuracy": latest.get("accuracy"),
            "latest_sharpe": latest.get("sharpe"),
            "latest_profit_factor": latest.get("profit_factor"),
            "latest_drawdown": latest.get("max_drawdown"),
            "latest_drift": latest.get("drift_score"),
            "calibration_error": calibration.get("calibration_error"),
            "paper_symbols": list(paper.keys()),
        }

    def _render_html(self, payload: Dict[str, Any]) -> str:
        summary = payload.get("summary") or {}
        rows = []
        for champ in payload.get("champions") or []:
            metrics = champ.get("metrics") or {}
            age = champ.get("model_age_hours")
            age_txt = f"{float(age):.1f}" if isinstance(age, (int, float)) else "-"
            score = metrics.get("test_f1") or metrics.get("f1") or "-"
            rows.append(
                "<tr>"
                f"<td>{champ.get('symbol')}</td>"
                f"<td>{champ.get('timeframe')}</td>"
                f"<td>{champ.get('model_type')}</td>"
                f"<td>{age_txt}</td>"
                f"<td>{score}</td>"
                "</tr>"
            )
        table = "\n".join(rows) or "<tr><td colspan='5'>No champions yet</td></tr>"
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>eTrade Institutional Monitoring</title>
  <style>
    body {{ font-family: "IBM Plex Sans", "Segoe UI", sans-serif; margin: 2rem; background: #0f1419; color: #e7ecf1; }}
    h1 {{ font-weight: 600; letter-spacing: 0.02em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
    .card {{ background: #1a222c; padding: 1rem 1.2rem; border-left: 3px solid #3d8bfd; }}
    .label {{ color: #9aa7b5; font-size: 0.8rem; text-transform: uppercase; }}
    .value {{ font-size: 1.6rem; margin-top: 0.35rem; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1.5rem; }}
    th, td {{ text-align: left; padding: 0.65rem 0.5rem; border-bottom: 1px solid #2a3441; }}
    th {{ color: #9aa7b5; font-size: 0.8rem; text-transform: uppercase; }}
    .meta {{ color: #9aa7b5; margin-top: 2rem; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>eTrade Institutional Monitoring</h1>
  <div class="meta">Generated {payload.get("generated_at")}</div>
  <div class="grid">
    <div class="card"><div class="label">Accuracy</div><div class="value">{summary.get("latest_accuracy") if summary.get("latest_accuracy") is not None else "-"}</div></div>
    <div class="card"><div class="label">Sharpe</div><div class="value">{summary.get("latest_sharpe") if summary.get("latest_sharpe") is not None else "-"}</div></div>
    <div class="card"><div class="label">Profit Factor</div><div class="value">{summary.get("latest_profit_factor") if summary.get("latest_profit_factor") is not None else "-"}</div></div>
    <div class="card"><div class="label">Drawdown</div><div class="value">{summary.get("latest_drawdown") if summary.get("latest_drawdown") is not None else "-"}</div></div>
    <div class="card"><div class="label">Drift</div><div class="value">{summary.get("latest_drift") if summary.get("latest_drift") is not None else "-"}</div></div>
    <div class="card"><div class="label">Calibration Error</div><div class="value">{summary.get("calibration_error") if summary.get("calibration_error") is not None else "-"}</div></div>
  </div>
  <h2>Champion Models</h2>
  <table>
    <thead><tr><th>Symbol</th><th>TF</th><th>Model</th><th>Age (h)</th><th>Score</th></tr></thead>
    <tbody>
      {table}
    </tbody>
  </table>
</body>
</html>
"""
