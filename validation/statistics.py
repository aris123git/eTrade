"""
validation/statistics.py - AI Performance Evaluation Module

RESPONSIBILITY:
Evaluate the performance of AI models, strategies, and predictions.

This module is NOT about market knowledge.
This module IS about evaluating what the AI does.

PURPOSE:
- Prediction statistics (accuracy, precision, recall, F1, confusion matrix)
- Trading performance (profit factor, Sharpe ratio, drawdown, win rate)
- Model evaluation (performance by symbol, timeframe, session)
- Confidence calibration (prediction confidence vs actual accuracy)
- Equity curve analysis (drawdown, recovery, consistency)
- A/B testing and model version comparison

ARCHITECTURAL PRINCIPLES:
1. Repository-based - SQLite for persistence
2. Incremental updates - Add new evaluations without recomputing
3. Extensible - Support new metrics without breaking existing
4. Type-safe - Dataclasses for all data structures
5. Production-ready - Scale to millions of predictions

VERSION: 1.0.0
"""

import json
import logging
import math
import sqlite3
import statistics as stats
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Set, Union, Callable
from collections import defaultdict
from contextlib import contextmanager

from core.config import Config
from core.exceptions import DatabaseError, DataValidationError
from core.utils import to_datetime, format_datetime


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Enums
    'PredictionOutcome',
    'EvaluationPeriod',
    'MetricCategory',
    
    # Data classes - Prediction Metrics
    'PredictionMetrics',
    'ClassificationMetrics',
    'RegressionMetrics',
    'ConfusionMatrix',
    
    # Data classes - Trading Metrics
    'TradingMetrics',
    'RiskMetrics',
    'EquityMetrics',
    'DrawdownMetrics',
    
    # Data classes - Performance Reports
    'PerformanceReport',
    'PeriodPerformance',
    'SymbolPerformance',
    'TimeframePerformance',
    'SessionPerformance',
    'StrategyPerformance',
    'ModelPerformance',
    
    # Main class
    'ValidationStatistics',
    'create_validation_statistics',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class PredictionOutcome(Enum):
    """Outcome of a prediction."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    TRUE_POSITIVE = "true_positive"
    TRUE_NEGATIVE = "true_negative"
    UNKNOWN = "unknown"


class EvaluationPeriod(Enum):
    """Time periods for evaluation."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class MetricCategory(Enum):
    """Categories of metrics."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TRADING = "trading"
    RISK = "risk"
    EQUITY = "equity"
    CONFIDENCE = "confidence"


# ==============================================================================
# DATA CLASSES - PREDICTION METRICS
# ==============================================================================

@dataclass
class ConfusionMatrix:
    """Confusion matrix for classification predictions."""
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total: int = 0
    
    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        if self.total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total
    
    @property
    def precision(self) -> float:
        """Calculate precision."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """Calculate recall."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        """Calculate F1-score."""
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    @property
    def specificity(self) -> float:
        """Calculate specificity."""
        if self.true_negatives + self.false_positives == 0:
            return 0.0
        return self.true_negatives / (self.true_negatives + self.false_positives)
    
    @property
    def false_positive_rate(self) -> float:
        """Calculate false positive rate."""
        if self.true_negatives + self.false_positives == 0:
            return 0.0
        return self.false_positives / (self.true_negatives + self.false_positives)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'total': self.total,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'specificity': self.specificity,
            'false_positive_rate': self.false_positive_rate,
        }


@dataclass
class PredictionMetrics:
    """Comprehensive prediction metrics."""
    symbol: str
    timeframe: str
    model_version: str
    strategy: str
    period: EvaluationPeriod
    
    # Classification metrics
    confusion_matrix: ConfusionMatrix
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    
    # Regression metrics
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float # Root Mean Squared Error
    mape: float # Mean Absolute Percentage Error
    
    # Confidence metrics
    avg_confidence: float
    confidence_calibration: float  # 0-1, how well confidence matches accuracy
    calibration_curve: List[Tuple[float, float]]  # (confidence_bin, actual_accuracy)
    
    # Counts
    total_predictions: int
    correct_predictions: int
    incorrect_predictions: int
    
    # Timestamps
    start_date: datetime
    end_date: datetime
    updated_at: datetime
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'model_version': self.model_version,
            'strategy': self.strategy,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'confusion_matrix': self.confusion_matrix.to_dict(),
            'total_predictions': self.total_predictions,
        }


@dataclass
class ClassificationMetrics:
    """Classification-specific metrics."""
    symbol: str
    timeframe: str
    model_version: str
    confusion_matrix: ConfusionMatrix
    
    # Class-wise metrics
    class_precision: Dict[str, float]
    class_recall: Dict[str, float]
    class_f1: Dict[str, float]
    
    # Macro vs weighted
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionMetrics:
    """Regression-specific metrics."""
    symbol: str
    timeframe: str
    model_version: str
    mae: float
    mse: float
    rmse: float
    mape: float
    r2: float
    adjusted_r2: float
    explained_variance: float
    max_error: float
    mean_absolute_scaled_error: float
    symmetric_mape: float
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# DATA CLASSES - TRADING METRICS
# ==============================================================================

@dataclass
class TradingMetrics:
    """Comprehensive trading performance metrics."""
    symbol: str
    timeframe: str
    model_version: str
    strategy: str
    period: EvaluationPeriod
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Profit statistics
    total_profit: float
    total_loss: float
    net_profit: float
    profit_factor: float
    
    # Trade size statistics
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade: float
    
    # Risk/reward statistics
    avg_rr: float  # Average Risk/Reward ratio
    rrr: float     # Risk/Reward ratio
    
    # Duration statistics
    avg_trade_duration: float  # in seconds
    avg_holding_time: float    # in seconds
    max_holding_time: float
    min_holding_time: float
    
    # Trade counts
    long_trades: int
    short_trades: int
    long_wins: int
    short_wins: int
    
    # Timestamps
    start_date: datetime
    end_date: datetime
    updated_at: datetime
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def expectancy(self) -> float:
        """Expected value per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.net_profit / self.total_trades
    
    @property
    def win_loss_ratio(self) -> float:
        """Ratio of average win to average loss."""
        if self.avg_loss == 0:
            return 0.0
        return self.avg_win / abs(self.avg_loss)
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'model_version': self.model_version,
            'strategy': self.strategy,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'net_profit': self.net_profit,
            'avg_rr': self.avg_rr,
            'expectancy': self.expectancy,
        }


@dataclass
class RiskMetrics:
    """Risk-adjusted performance metrics."""
    symbol: str
    timeframe: str
    model_version: str
    strategy: str
    period: EvaluationPeriod
    
    # Risk ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk statistics
    max_drawdown: float
    max_drawdown_percent: float
    max_drawdown_duration: float  # in seconds
    avg_drawdown: float
    avg_drawdown_percent: float
    
    # Recovery
    avg_recovery_time: float
    max_recovery_time: float
    recovery_factor: float
    
    # Other risk metrics
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    cvar_95: float # 95% Conditional VaR
    ulcer_index: float
    
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown_percent': self.max_drawdown_percent,
            'var_95': self.var_95,
            'var_99': self.var_99,
        }


@dataclass
class EquityMetrics:
    """Equity curve and performance metrics."""
    symbol: str
    timeframe: str
    model_version: str
    strategy: str
    period: EvaluationPeriod
    
    # Equity statistics
    initial_equity: float
    final_equity: float
    peak_equity: float
    trough_equity: float
    
    # Performance
    total_return: float
    total_return_percent: float
    annualized_return: float
    
    # Consistency
    winning_months: int
    losing_months: int
    consecutive_wins: int
    consecutive_losses: int
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Equity curve
    equity_curve: List[Tuple[datetime, float]]
    
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def monthly_win_rate(self) -> float:
        total = self.winning_months + self.losing_months
        if total == 0:
            return 0.0
        return self.winning_months / total


@dataclass
class DrawdownMetrics:
    """Detailed drawdown analysis."""
    symbol: str
    timeframe: str
    model_version: str
    strategy: str
    
    drawdowns: List[Dict[str, Any]]
    max_drawdown: float
    max_drawdown_percent: float
    avg_drawdown: float
    avg_drawdown_percent: float
    drawdown_count: int
    avg_duration: float
    
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# DATA CLASSES - PERFORMANCE REPORTS
# ==============================================================================

@dataclass
class PeriodPerformance:
    """Performance by period (daily, weekly, monthly)."""
    period: str
    period_value: Union[int, str]
    total_trades: int
    win_rate: float
    profit_factor: float
    net_profit: float
    avg_returns: float
    std_returns: float
    max_return: float
    min_return: float


@dataclass
class SymbolPerformance:
    """Performance by symbol."""
    symbol: str
    total_trades: int
    win_rate: float
    profit_factor: float
    net_profit: float
    avg_trade: float
    sharpe_ratio: float
    max_drawdown_percent: float


@dataclass
class TimeframePerformance:
    """Performance by timeframe."""
    timeframe: str
    total_trades: int
    win_rate: float
    profit_factor: float
    net_profit: float
    avg_trade: float


@dataclass
class SessionPerformance:
    """Performance by trading session."""
    session: str
    total_trades: int
    win_rate: float
    profit_factor: float
    net_profit: float


@dataclass
class StrategyPerformance:
    """Performance by strategy."""
    strategy: str
    total_trades: int
    win_rate: float
    profit_factor: float
    net_profit: float
    avg_trade: float
    sharpe_ratio: float
    max_drawdown_percent: float


@dataclass
class ModelPerformance:
    """Performance by model version."""
    model_version: str
    total_trades: int
    win_rate: float
    profit_factor: float
    net_profit: float
    avg_trade: float
    sharpe_ratio: float
    max_drawdown_percent: float
    prediction_accuracy: float


@dataclass
class PerformanceReport:
    """Complete performance report."""
    report_id: str
    report_name: str
    report_type: str
    generated_at: datetime
    period: EvaluationPeriod
    
    # Summary
    summary: Dict[str, Any]
    
    # Detailed breakdowns
    period_performance: List[PeriodPerformance]
    symbol_performance: List[SymbolPerformance]
    timeframe_performance: List[TimeframePerformance]
    session_performance: List[SessionPerformance]
    strategy_performance: List[StrategyPerformance]
    model_performance: List[ModelPerformance]
    
    # Metrics
    trading_metrics: TradingMetrics
    risk_metrics: RiskMetrics
    equity_metrics: EquityMetrics
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'report_id': self.report_id,
            'report_name': self.report_name,
            'report_type': self.report_type,
            'generated_at': self.generated_at.isoformat(),
            'period': self.period.value,
            'summary': self.summary,
            'trading_metrics': {
                'total_trades': self.trading_metrics.total_trades,
                'win_rate': self.trading_metrics.win_rate,
                'profit_factor': self.trading_metrics.profit_factor,
                'net_profit': self.trading_metrics.net_profit,
                'expectancy': self.trading_metrics.expectancy,
            },
            'risk_metrics': {
                'sharpe_ratio': self.risk_metrics.sharpe_ratio,
                'sortino_ratio': self.risk_metrics.sortino_ratio,
                'calmar_ratio': self.risk_metrics.calmar_ratio,
                'max_drawdown_percent': self.risk_metrics.max_drawdown_percent,
            },
            'equity_metrics': {
                'total_return_percent': self.equity_metrics.total_return_percent,
                'annualized_return': self.equity_metrics.annualized_return,
                'consecutive_wins': self.equity_metrics.consecutive_wins,
                'consecutive_losses': self.equity_metrics.consecutive_losses,
            },
        }


# ==============================================================================
# MAIN VALIDATION STATISTICS CLASS
# ==============================================================================

class ValidationStatistics:
    """
    AI Performance Evaluation Engine.
    
    Evaluates models, strategies, and predictions.
    This is about measuring how well the AI performs.
    """
    
    # Database schema version
    SCHEMA_VERSION = 1
    
    def __init__(self, config: Config):
        """
        Initialize the validation statistics engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = getattr(config, 'DB_PATH', 'market_ai.db')
        
        # Caches
        self._prediction_cache: Dict[str, PredictionMetrics] = {}
        self._trading_cache: Dict[str, TradingMetrics] = {}
        self._risk_cache: Dict[str, RiskMetrics] = {}
        self._equity_cache: Dict[str, EquityMetrics] = {}
        self._report_cache: Dict[str, PerformanceReport] = {}
        
        # Initialize database
        self._init_database()
        
        self.logger.info("✅ ValidationStatistics initialized")
    
    # ==========================================================================
    # DATABASE METHODS
    # ==========================================================================
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Prediction metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    period TEXT NOT NULL,
                    total_predictions INTEGER,
                    correct_predictions INTEGER,
                    incorrect_predictions INTEGER,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    specificity REAL,
                    mae REAL,
                    mse REAL,
                    rmse REAL,
                    mape REAL,
                    avg_confidence REAL,
                    confidence_calibration REAL,
                    confusion_matrix TEXT,
                    start_date TIMESTAMP,
                    end_date TIMESTAMP,
                    metadata TEXT,
                    updated_at TIMESTAMP,
                    UNIQUE(symbol, timeframe, model_version, strategy, period)
                )
            """)
            
            # Trading metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_trading (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    period TEXT NOT NULL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    total_profit REAL,
                    total_loss REAL,
                    net_profit REAL,
                    profit_factor REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    largest_win REAL,
                    largest_loss REAL,
                    avg_trade REAL,
                    avg_rr REAL,
                    rrr REAL,
                    avg_trade_duration REAL,
                    avg_holding_time REAL,
                    max_holding_time REAL,
                    min_holding_time REAL,
                    long_trades INTEGER,
                    short_trades INTEGER,
                    long_wins INTEGER,
                    short_wins INTEGER,
                    start_date TIMESTAMP,
                    end_date TIMESTAMP,
                    metadata TEXT,
                    updated_at TIMESTAMP,
                    UNIQUE(symbol, timeframe, model_version, strategy, period)
                )
            """)
            
            # Risk metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_risk (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    period TEXT NOT NULL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    calmar_ratio REAL,
                    max_drawdown REAL,
                    max_drawdown_percent REAL,
                    max_drawdown_duration REAL,
                    avg_drawdown REAL,
                    avg_drawdown_percent REAL,
                    avg_recovery_time REAL,
                    max_recovery_time REAL,
                    recovery_factor REAL,
                    var_95 REAL,
                    var_99 REAL,
                    cvar_95 REAL,
                    ulcer_index REAL,
                    metadata TEXT,
                    updated_at TIMESTAMP,
                    UNIQUE(symbol, timeframe, model_version, strategy, period)
                )
            """)
            
            # Equity metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    period TEXT NOT NULL,
                    initial_equity REAL,
                    final_equity REAL,
                    peak_equity REAL,
                    trough_equity REAL,
                    total_return REAL,
                    total_return_percent REAL,
                    annualized_return REAL,
                    winning_months INTEGER,
                    losing_months INTEGER,
                    consecutive_wins INTEGER,
                    consecutive_losses INTEGER,
                    max_consecutive_wins INTEGER,
                    max_consecutive_losses INTEGER,
                    equity_curve TEXT,
                    metadata TEXT,
                    updated_at TIMESTAMP,
                    UNIQUE(symbol, timeframe, model_version, strategy, period)
                )
            """)
            
            # Performance reports
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT UNIQUE NOT NULL,
                    report_name TEXT NOT NULL,
                    report_type TEXT NOT NULL,
                    period TEXT NOT NULL,
                    generated_at TIMESTAMP,
                    summary TEXT,
                    metadata TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_val_predictions_symbol ON validation_predictions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_val_predictions_model ON validation_predictions(model_version)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_val_trading_symbol ON validation_trading(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_val_trading_strategy ON validation_trading(strategy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_val_risk_symbol ON validation_risk(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_val_equity_symbol ON validation_equity(symbol)")
            
            self.logger.info("✅ Validation database schema initialized")
    
    # ==========================================================================
    # PREDICTION STATISTICS
    # ==========================================================================
    
    def compute_prediction_statistics(
        self,
        symbol: str,
        timeframe: str,
        model_version: str,
        strategy: str,
        predictions: List[Dict[str, Any]],
        period: EvaluationPeriod = EvaluationPeriod.DAILY,
    ) -> PredictionMetrics:
        """
        Compute prediction statistics.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            model_version: Model version
            strategy: Strategy name
            predictions: List of prediction dictionaries with:
                - predicted: predicted value
                - actual: actual value
                - confidence: prediction confidence (0-1)
                - timestamp: prediction timestamp
            period: Evaluation period
            
        Returns:
            PredictionMetrics object
        """
        self.logger.debug(f"Computing prediction statistics for {symbol} {timeframe}")
        
        if not predictions:
            return None
        
        # Extract data
        predicted = [p['predicted'] for p in predictions]
        actual = [p['actual'] for p in predictions]
        confidences = [p.get('confidence', 0.5) for p in predictions]
        timestamps = [p.get('timestamp', datetime.now()) for p in predictions]
        
        # Classification vs regression
        is_classification = all(isinstance(p, (bool, int)) and p in (0, 1) for p in predicted)
        
        if is_classification:
            confusion = self._compute_confusion_matrix(predicted, actual)
            accuracy = confusion.accuracy
            precision = confusion.precision
            recall = confusion.recall
            f1 = confusion.f1_score
            specificity = confusion.specificity
            
            # Regression metrics (not applicable for classification)
            mae = 0.0
            mse = 0.0
            rmse = 0.0
            mape = 0.0
        else:
            confusion = ConfusionMatrix()
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            specificity = 0.0
            
            # Regression metrics
            mae = self._compute_mae(predicted, actual)
            mse = self._compute_mse(predicted, actual)
            rmse = math.sqrt(mse)
            mape = self._compute_mape(predicted, actual)
        
        # Confidence calibration
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        calibration = self._compute_calibration(predicted, actual, confidences)
        
        # Counts
        total = len(predictions)
        correct = sum(1 for p, a in zip(predicted, actual) if p == a)
        incorrect = total - correct
        
        metrics = PredictionMetrics(
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            strategy=strategy,
            period=period,
            confusion_matrix=confusion,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            mae=mae,
            mse=mse,
            rmse=rmse,
            mape=mape,
            avg_confidence=avg_confidence,
            confidence_calibration=calibration,
            calibration_curve=[],  # Would need bins for full curve
            total_predictions=total,
            correct_predictions=correct,
            incorrect_predictions=incorrect,
            start_date=min(timestamps),
            end_date=max(timestamps),
            updated_at=datetime.now(),
        )
        
        # Save to database
        self._save_prediction_metrics(metrics)
        self._prediction_cache[f"{symbol}|{timeframe}|{model_version}|{strategy}|{period.value}"] = metrics
        
        return metrics
    
    def _compute_confusion_matrix(self, predicted: List, actual: List) -> ConfusionMatrix:
        """Compute confusion matrix."""
        tp = tn = fp = fn = 0
        
        for p, a in zip(predicted, actual):
            if p == 1 and a == 1:
                tp += 1
            elif p == 1 and a == 0:
                fp += 1
            elif p == 0 and a == 1:
                fn += 1
            else:
                tn += 1
        
        return ConfusionMatrix(
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            total=len(predicted),
        )
    
    def _compute_mae(self, predicted: List[float], actual: List[float]) -> float:
        """Compute Mean Absolute Error."""
        if not predicted:
            return 0.0
        return sum(abs(p - a) for p, a in zip(predicted, actual)) / len(predicted)
    
    def _compute_mse(self, predicted: List[float], actual: List[float]) -> float:
        """Compute Mean Squared Error."""
        if not predicted:
            return 0.0
        return sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(predicted)
    
    def _compute_mape(self, predicted: List[float], actual: List[float]) -> float:
        """Compute Mean Absolute Percentage Error."""
        if not predicted:
            return 0.0
        total = sum(abs((p - a) / a) for p, a in zip(predicted, actual) if a != 0)
        return total / len(predicted)
    
    def _compute_calibration(self, predicted: List, actual: List, confidences: List[float]) -> float:
        """
        Compute confidence calibration score.
        
        Returns:
            0-1 score where 1 = perfectly calibrated
        """
        if not confidences:
            return 0.0
        
        # Simple calibration: accuracy vs average confidence
        accuracy = sum(1 for p, a in zip(predicted, actual) if p == a) / len(predicted)
        avg_conf = sum(confidences) / len(confidences)
        
        if avg_conf == 0:
            return 0.0
        
        # 1 - |accuracy - avg_confidence|
        return 1.0 - min(1.0, abs(accuracy - avg_conf) / max(avg_conf, 0.01))
    
    # ==========================================================================
    # TRADING STATISTICS
    # ==========================================================================
    
    def compute_trading_statistics(
        self,
        symbol: str,
        timeframe: str,
        model_version: str,
        strategy: str,
        trades: List[Dict[str, Any]],
        period: EvaluationPeriod = EvaluationPeriod.DAILY,
    ) -> TradingMetrics:
        """
        Compute trading statistics.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            model_version: Model version
            strategy: Strategy name
            trades: List of trade dictionaries with:
                - entry_price: entry price
                - exit_price: exit price
                - direction: 'long' or 'short'
                - entry_time: entry timestamp
                - exit_time: exit timestamp
                - profit: profit/loss
                - rr: risk/reward ratio
            period: Evaluation period
            
        Returns:
            TradingMetrics object
        """
        self.logger.debug(f"Computing trading statistics for {symbol} {timeframe}")
        
        if not trades:
            return None
        
        # Basic statistics
        total_trades = len(trades)
        winning = [t for t in trades if t.get('profit', 0) > 0]
        losing = [t for t in trades if t.get('profit', 0) <= 0]
        
        winning_trades = len(winning)
        losing_trades = len(losing)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Profit statistics
        profits = [t['profit'] for t in trades]
        total_profit = sum(p for p in profits if p > 0)
        total_loss = sum(p for p in profits if p < 0)
        net_profit = sum(profits)
        profit_factor = total_profit / abs(total_loss) if total_loss != 0 else 0.0
        
        # Trade size statistics
        wins = [t['profit'] for t in winning]
        losses = [t['profit'] for t in losing]
        
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_trade = net_profit / total_trades if total_trades > 0 else 0.0
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0
        
        # Risk/Reward
        rrs = [t.get('rr', 0.0) for t in trades]
        avg_rr = sum(rrs) / len(rrs) if rrs else 0.0
        
        # Duration
        durations = []
        for t in trades:
            if 'entry_time' in t and 'exit_time' in t:
                duration = (t['exit_time'] - t['entry_time']).total_seconds()
                durations.append(duration)
        
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0
        
        # Long/Short
        longs = [t for t in trades if t.get('direction') == 'long']
        shorts = [t for t in trades if t.get('direction') == 'short']
        
        long_wins = sum(1 for t in longs if t.get('profit', 0) > 0)
        short_wins = sum(1 for t in shorts if t.get('profit', 0) > 0)
        
        metrics = TradingMetrics(
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            strategy=strategy,
            period=period,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            total_loss=total_loss,
            net_profit=net_profit,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade=avg_trade,
            avg_rr=avg_rr,
            rrr=avg_rr,
            avg_trade_duration=avg_duration,
            avg_holding_time=avg_duration,
            max_holding_time=max_duration,
            min_holding_time=min_duration,
            long_trades=len(longs),
            short_trades=len(shorts),
            long_wins=long_wins,
            short_wins=short_wins,
            start_date=min(t.get('entry_time', datetime.now()) for t in trades),
            end_date=max(t.get('exit_time', datetime.now()) for t in trades),
            updated_at=datetime.now(),
        )
        
        # Save
        self._save_trading_metrics(metrics)
        self._trading_cache[f"{symbol}|{timeframe}|{model_version}|{strategy}|{period.value}"] = metrics
        
        return metrics
    
    # ==========================================================================
    # RISK STATISTICS
    # ==========================================================================
    
    def compute_risk_statistics(
        self,
        symbol: str,
        timeframe: str,
        model_version: str,
        strategy: str,
        trades: List[Dict[str, Any]],
        period: EvaluationPeriod = EvaluationPeriod.DAILY,
    ) -> RiskMetrics:
        """
        Compute risk-adjusted statistics.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            model_version: Model version
            strategy: Strategy name
            trades: List of trade dictionaries
            period: Evaluation period
            
        Returns:
            RiskMetrics object
        """
        self.logger.debug(f"Computing risk statistics for {symbol} {timeframe}")
        
        if not trades:
            return None
        
        # Extract returns
        returns = []
        equity = [0.0]
        for t in trades:
            profit = t.get('profit', 0.0)
            returns.append(profit)
            equity.append(equity[-1] + profit)
        
        # Compute drawdown
        peak = max(equity)
        drawdowns = [peak - e for e in equity]
        max_drawdown = max(drawdowns)
        max_drawdown_pct = max_drawdown / peak if peak != 0 else 0.0
        
        # Sharpe ratio
        avg_return = sum(returns) / len(returns) if returns else 0.0
        std_return = stats.stdev(returns) if len(returns) > 1 else 1.0
        sharpe = avg_return / std_return if std_return != 0 else 0.0
        
        # Sortino ratio (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        downside_std = stats.stdev(negative_returns) if len(negative_returns) > 1 else 1.0
        sortino = avg_return / downside_std if downside_std != 0 else 0.0
        
        # Calmar ratio
        calmar = avg_return / max_drawdown_pct if max_drawdown_pct != 0 else 0.0
        
        # VaR
        sorted_returns = sorted(returns)
        var_95 = self._percentile(sorted_returns, 0.05)
        var_99 = self._percentile(sorted_returns, 0.01)
        
        # CVaR
        cvar_95 = sum(r for r in returns if r <= var_95) / len([r for r in returns if r <= var_95]) if returns else 0.0
        
        metrics = RiskMetrics(
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            strategy=strategy,
            period=period,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_pct,
            max_drawdown_duration=0.0,
            avg_drawdown=sum(drawdowns) / len(drawdowns) if drawdowns else 0.0,
            avg_drawdown_percent=0.0,
            avg_recovery_time=0.0,
            max_recovery_time=0.0,
            recovery_factor=0.0,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            ulcer_index=0.0,
            updated_at=datetime.now(),
        )
        
        self._save_risk_metrics(metrics)
        self._risk_cache[f"{symbol}|{timeframe}|{model_version}|{strategy}|{period.value}"] = metrics
        
        return metrics
    
    def _percentile(self, sorted_values: List[float], p: float) -> float:
        """Calculate percentile from sorted values."""
        n = len(sorted_values)
        if n == 0:
            return 0.0
        idx = p * (n - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        if lower == upper:
            return sorted_values[lower]
        weight = idx - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    
    # ==========================================================================
    # EQUITY STATISTICS
    # ==========================================================================
    
    def compute_equity_statistics(
        self,
        symbol: str,
        timeframe: str,
        model_version: str,
        strategy: str,
        trades: List[Dict[str, Any]],
        initial_equity: float = 10000.0,
        period: EvaluationPeriod = EvaluationPeriod.DAILY,
    ) -> EquityMetrics:
        """
        Compute equity curve statistics.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            model_version: Model version
            strategy: Strategy name
            trades: List of trade dictionaries
            initial_equity: Starting equity
            period: Evaluation period
            
        Returns:
            EquityMetrics object
        """
        self.logger.debug(f"Computing equity statistics for {symbol} {timeframe}")
        
        if not trades:
            return None
        
        # Build equity curve
        equity_curve = [(initial_equity, 0)]
        for t in trades:
            profit = t.get('profit', 0.0)
            last = equity_curve[-1][1]
            equity_curve.append((last + profit, last + profit))
        
        # Metrics
        final_equity = equity_curve[-1][1]
        peak_equity = max(e[1] for e in equity_curve)
        trough_equity = min(e[1] for e in equity_curve)
        
        total_return = final_equity - initial_equity
        total_return_pct = total_return / initial_equity if initial_equity != 0 else 0.0
        
        # Annualized return
        start = trades[0].get('entry_time', datetime.now())
        end = trades[-1].get('exit_time', datetime.now())
        days = (end - start).days or 1
        annualized = (1 + total_return_pct) ** (365 / days) - 1
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for t in trades:
            if t.get('profit', 0) > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        metrics = EquityMetrics(
            symbol=symbol,
            timeframe=timeframe,
            model_version=model_version,
            strategy=strategy,
            period=period,
            initial_equity=initial_equity,
            final_equity=final_equity,
            peak_equity=peak_equity,
            trough_equity=trough_equity,
            total_return=total_return,
            total_return_percent=total_return_pct,
            annualized_return=annualized,
            winning_months=0,
            losing_months=0,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            equity_curve=equity_curve,
            updated_at=datetime.now(),
        )
        
        self._save_equity_metrics(metrics)
        self._equity_cache[f"{symbol}|{timeframe}|{model_version}|{strategy}|{period.value}"] = metrics
        
        return metrics
    
    # ==========================================================================
    # REPORT GENERATION
    # ==========================================================================
    
    def generate_performance_report(
        self,
        symbol: str,
        timeframe: str,
        model_version: str,
        strategy: str,
        trades: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        report_name: str,
        period: EvaluationPeriod = EvaluationPeriod.DAILY,
    ) -> PerformanceReport:
        """
        Generate complete performance report.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            model_version: Model version
            strategy: Strategy name
            trades: List of trade dictionaries
            predictions: List of prediction dictionaries
            report_name: Name of the report
            period: Evaluation period
            
        Returns:
            PerformanceReport object
        """
        self.logger.info(f"Generating performance report: {report_name}")
        
        # Compute all metrics
        trading = self.compute_trading_statistics(symbol, timeframe, model_version, strategy, trades, period)
        risk = self.compute_risk_statistics(symbol, timeframe, model_version, strategy, trades, period)
        equity = self.compute_equity_statistics(symbol, timeframe, model_version, strategy, trades, 10000.0, period)
        prediction = self.compute_prediction_statistics(symbol, timeframe, model_version, strategy, predictions, period)
        
        # Build report
        report = PerformanceReport(
            report_id=f"{symbol}_{timeframe}_{int(datetime.now().timestamp())}",
            report_name=report_name,
            report_type='performance',
            generated_at=datetime.now(),
            period=period,
            summary={
                'total_trades': trading.total_trades if trading else 0,
                'win_rate': trading.win_rate if trading else 0.0,
                'profit_factor': trading.profit_factor if trading else 0.0,
                'net_profit': trading.net_profit if trading else 0.0,
                'sharpe_ratio': risk.sharpe_ratio if risk else 0.0,
                'max_drawdown_percent': risk.max_drawdown_percent if risk else 0.0,
                'total_return_percent': equity.total_return_percent if equity else 0.0,
                'prediction_accuracy': prediction.accuracy if prediction else 0.0,
            },
            period_performance=[],
            symbol_performance=[],
            timeframe_performance=[],
            session_performance=[],
            strategy_performance=[],
            model_performance=[],
            trading_metrics=trading,
            risk_metrics=risk,
            equity_metrics=equity,
            metadata={
                'symbol': symbol,
                'timeframe': timeframe,
                'model_version': model_version,
                'strategy': strategy,
                'period': period.value,
            },
        )
        
        # Save report
        self._save_report(report)
        self._report_cache[report.report_id] = report
        
        return report
    
    # ==========================================================================
    # GET METHODS
    # ==========================================================================
    
    def get_prediction_metrics(
        self,
        symbol: str,
        timeframe: str,
        model_version: str,
        strategy: str,
        period: EvaluationPeriod,
    ) -> Optional[PredictionMetrics]:
        """Get prediction metrics from cache or database."""
        cache_key = f"{symbol}|{timeframe}|{model_version}|{strategy}|{period.value}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM validation_predictions
                WHERE symbol = ? AND timeframe = ? AND model_version = ? AND strategy = ? AND period = ?
            """, (symbol, timeframe, model_version, strategy, period.value))
            row = cursor.fetchone()
            
            if row:
                metrics = self._row_to_prediction_metrics(row)
                self._prediction_cache[cache_key] = metrics
                return metrics
        
        return None
    
    def get_trading_metrics(
        self,
        symbol: str,
        timeframe: str,
        model_version: str,
        strategy: str,
        period: EvaluationPeriod,
    ) -> Optional[TradingMetrics]:
        """Get trading metrics from cache or database."""
        cache_key = f"{symbol}|{timeframe}|{model_version}|{strategy}|{period.value}"
        if cache_key in self._trading_cache:
            return self._trading_cache[cache_key]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM validation_trading
                WHERE symbol = ? AND timeframe = ? AND model_version = ? AND strategy = ? AND period = ?
            """, (symbol, timeframe, model_version, strategy, period.value))
            row = cursor.fetchone()
            
            if row:
                metrics = self._row_to_trading_metrics(row)
                self._trading_cache[cache_key] = metrics
                return metrics
        
        return None
    
    def get_risk_metrics(
        self,
        symbol: str,
        timeframe: str,
        model_version: str,
        strategy: str,
        period: EvaluationPeriod,
    ) -> Optional[RiskMetrics]:
        """Get risk metrics from cache or database."""
        cache_key = f"{symbol}|{timeframe}|{model_version}|{strategy}|{period.value}"
        if cache_key in self._risk_cache:
            return self._risk_cache[cache_key]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM validation_risk
                WHERE symbol = ? AND timeframe = ? AND model_version = ? AND strategy = ? AND period = ?
            """, (symbol, timeframe, model_version, strategy, period.value))
            row = cursor.fetchone()
            
            if row:
                metrics = self._row_to_risk_metrics(row)
                self._risk_cache[cache_key] = metrics
                return metrics
        
        return None
    
    def get_equity_metrics(
        self,
        symbol: str,
        timeframe: str,
        model_version: str,
        strategy: str,
        period: EvaluationPeriod,
    ) -> Optional[EquityMetrics]:
        """Get equity metrics from cache or database."""
        cache_key = f"{symbol}|{timeframe}|{model_version}|{strategy}|{period.value}"
        if cache_key in self._equity_cache:
            return self._equity_cache[cache_key]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM validation_equity
                WHERE symbol = ? AND timeframe = ? AND model_version = ? AND strategy = ? AND period = ?
            """, (symbol, timeframe, model_version, strategy, period.value))
            row = cursor.fetchone()
            
            if row:
                metrics = self._row_to_equity_metrics(row)
                self._equity_cache[cache_key] = metrics
                return metrics
        
        return None
    
    def get_report(self, report_id: str) -> Optional[PerformanceReport]:
        """Get performance report by ID."""
        if report_id in self._report_cache:
            return self._report_cache[report_id]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM validation_reports WHERE report_id = ?
            """, (report_id,))
            row = cursor.fetchone()
            
            if row:
                # Load full report from database
                # For simplicity, we return the stored summary
                report = PerformanceReport(
                    report_id=row['report_id'],
                    report_name=row['report_name'],
                    report_type=row['report_type'],
                    generated_at=to_datetime(row['generated_at']),
                    period=EvaluationPeriod(row['period']),
                    summary=json.loads(row['summary']) if row['summary'] else {},
                    period_performance=[],
                    symbol_performance=[],
                    timeframe_performance=[],
                    session_performance=[],
                    strategy_performance=[],
                    model_performance=[],
                    trading_metrics=None,
                    risk_metrics=None,
                    equity_metrics=None,
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                )
                self._report_cache[report_id] = report
                return report
        
        return None
    
    # ==========================================================================
    # SAVE METHODS
    # ==========================================================================
    
    def _save_prediction_metrics(self, metrics: PredictionMetrics):
        """Save prediction metrics to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO validation_predictions
                (symbol, timeframe, model_version, strategy, period,
                 total_predictions, correct_predictions, incorrect_predictions,
                 accuracy, precision, recall, f1_score, specificity,
                 mae, mse, rmse, mape,
                 avg_confidence, confidence_calibration,
                 confusion_matrix, start_date, end_date, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.symbol, metrics.timeframe, metrics.model_version, metrics.strategy,
                metrics.period.value,
                metrics.total_predictions, metrics.correct_predictions, metrics.incorrect_predictions,
                metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score,
                metrics.specificity,
                metrics.mae, metrics.mse, metrics.rmse, metrics.mape,
                metrics.avg_confidence, metrics.confidence_calibration,
                json.dumps(metrics.confusion_matrix.to_dict()),
                metrics.start_date.isoformat(), metrics.end_date.isoformat(),
                json.dumps(metrics.metadata),
                metrics.updated_at.isoformat(),
            ))
    
    def _save_trading_metrics(self, metrics: TradingMetrics):
        """Save trading metrics to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO validation_trading
                (symbol, timeframe, model_version, strategy, period,
                 total_trades, winning_trades, losing_trades, win_rate,
                 total_profit, total_loss, net_profit, profit_factor,
                 avg_win, avg_loss, largest_win, largest_loss, avg_trade,
                 avg_rr, rrr,
                 avg_trade_duration, avg_holding_time, max_holding_time, min_holding_time,
                 long_trades, short_trades, long_wins, short_wins,
                 start_date, end_date, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.symbol, metrics.timeframe, metrics.model_version, metrics.strategy,
                metrics.period.value,
                metrics.total_trades, metrics.winning_trades, metrics.losing_trades,
                metrics.win_rate,
                metrics.total_profit, metrics.total_loss, metrics.net_profit,
                metrics.profit_factor,
                metrics.avg_win, metrics.avg_loss, metrics.largest_win, metrics.largest_loss,
                metrics.avg_trade,
                metrics.avg_rr, metrics.rrr,
                metrics.avg_trade_duration, metrics.avg_holding_time,
                metrics.max_holding_time, metrics.min_holding_time,
                metrics.long_trades, metrics.short_trades,
                metrics.long_wins, metrics.short_wins,
                metrics.start_date.isoformat(), metrics.end_date.isoformat(),
                json.dumps(metrics.metadata),
                metrics.updated_at.isoformat(),
            ))
    
    def _save_risk_metrics(self, metrics: RiskMetrics):
        """Save risk metrics to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO validation_risk
                (symbol, timeframe, model_version, strategy, period,
                 sharpe_ratio, sortino_ratio, calmar_ratio,
                 max_drawdown, max_drawdown_percent, max_drawdown_duration,
                 avg_drawdown, avg_drawdown_percent,
                 avg_recovery_time, max_recovery_time, recovery_factor,
                 var_95, var_99, cvar_95, ulcer_index,
                 metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.symbol, metrics.timeframe, metrics.model_version, metrics.strategy,
                metrics.period.value,
                metrics.sharpe_ratio, metrics.sortino_ratio, metrics.calmar_ratio,
                metrics.max_drawdown, metrics.max_drawdown_percent,
                metrics.max_drawdown_duration,
                metrics.avg_drawdown, metrics.avg_drawdown_percent,
                metrics.avg_recovery_time, metrics.max_recovery_time,
                metrics.recovery_factor,
                metrics.var_95, metrics.var_99, metrics.cvar_95, metrics.ulcer_index,
                json.dumps(metrics.metadata),
                metrics.updated_at.isoformat(),
            ))
    
    def _save_equity_metrics(self, metrics: EquityMetrics):
        """Save equity metrics to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO validation_equity
                (symbol, timeframe, model_version, strategy, period,
                 initial_equity, final_equity, peak_equity, trough_equity,
                 total_return, total_return_percent, annualized_return,
                 winning_months, losing_months,
                 consecutive_wins, consecutive_losses,
                 max_consecutive_wins, max_consecutive_losses,
                 equity_curve, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.symbol, metrics.timeframe, metrics.model_version, metrics.strategy,
                metrics.period.value,
                metrics.initial_equity, metrics.final_equity, metrics.peak_equity,
                metrics.trough_equity,
                metrics.total_return, metrics.total_return_percent,
                metrics.annualized_return,
                metrics.winning_months, metrics.losing_months,
                metrics.consecutive_wins, metrics.consecutive_losses,
                metrics.max_consecutive_wins, metrics.max_consecutive_losses,
                json.dumps(metrics.equity_curve, default=str),
                json.dumps(metrics.metadata),
                metrics.updated_at.isoformat(),
            ))
    
    def _save_report(self, report: PerformanceReport):
        """Save performance report to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO validation_reports
                (report_id, report_name, report_type, period, generated_at, summary, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                report.report_id,
                report.report_name,
                report.report_type,
                report.period.value,
                report.generated_at.isoformat(),
                json.dumps(report.summary),
                json.dumps(report.metadata),
            ))
    
    # ==========================================================================
    # ROW TO OBJECT METHODS
    # ==========================================================================
    
    def _row_to_prediction_metrics(self, row) -> PredictionMetrics:
        """Convert database row to PredictionMetrics."""
        confusion_dict = json.loads(row['confusion_matrix']) if row['confusion_matrix'] else {}
        confusion = ConfusionMatrix(
            true_positives=confusion_dict.get('true_positives', 0),
            true_negatives=confusion_dict.get('true_negatives', 0),
            false_positives=confusion_dict.get('false_positives', 0),
            false_negatives=confusion_dict.get('false_negatives', 0),
            total=confusion_dict.get('total', 0),
        )
        
        return PredictionMetrics(
            symbol=row['symbol'],
            timeframe=row['timeframe'],
            model_version=row['model_version'],
            strategy=row['strategy'],
            period=EvaluationPeriod(row['period']),
            confusion_matrix=confusion,
            accuracy=row['accuracy'],
            precision=row['precision'],
            recall=row['recall'],
            f1_score=row['f1_score'],
            specificity=row['specificity'],
            mae=row['mae'],
            mse=row['mse'],
            rmse=row['rmse'],
            mape=row['mape'],
            avg_confidence=row['avg_confidence'],
            confidence_calibration=row['confidence_calibration'],
            calibration_curve=[],
            total_predictions=row['total_predictions'],
            correct_predictions=row['correct_predictions'],
            incorrect_predictions=row['incorrect_predictions'],
            start_date=to_datetime(row['start_date']),
            end_date=to_datetime(row['end_date']),
            updated_at=to_datetime(row['updated_at']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
        )
    
    def _row_to_trading_metrics(self, row) -> TradingMetrics:
        """Convert database row to TradingMetrics."""
        return TradingMetrics(
            symbol=row['symbol'],
            timeframe=row['timeframe'],
            model_version=row['model_version'],
            strategy=row['strategy'],
            period=EvaluationPeriod(row['period']),
            total_trades=row['total_trades'],
            winning_trades=row['winning_trades'],
            losing_trades=row['losing_trades'],
            win_rate=row['win_rate'],
            total_profit=row['total_profit'],
            total_loss=row['total_loss'],
            net_profit=row['net_profit'],
            profit_factor=row['profit_factor'],
            avg_win=row['avg_win'],
            avg_loss=row['avg_loss'],
            largest_win=row['largest_win'],
            largest_loss=row['largest_loss'],
            avg_trade=row['avg_trade'],
            avg_rr=row['avg_rr'],
            rrr=row['rrr'],
            avg_trade_duration=row['avg_trade_duration'],
            avg_holding_time=row['avg_holding_time'],
            max_holding_time=row['max_holding_time'],
            min_holding_time=row['min_holding_time'],
            long_trades=row['long_trades'],
            short_trades=row['short_trades'],
            long_wins=row['long_wins'],
            short_wins=row['short_wins'],
            start_date=to_datetime(row['start_date']),
            end_date=to_datetime(row['end_date']),
            updated_at=to_datetime(row['updated_at']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
        )
    
    def _row_to_risk_metrics(self, row) -> RiskMetrics:
        """Convert database row to RiskMetrics."""
        return RiskMetrics(
            symbol=row['symbol'],
            timeframe=row['timeframe'],
            model_version=row['model_version'],
            strategy=row['strategy'],
            period=EvaluationPeriod(row['period']),
            sharpe_ratio=row['sharpe_ratio'],
            sortino_ratio=row['sortino_ratio'],
            calmar_ratio=row['calmar_ratio'],
            max_drawdown=row['max_drawdown'],
            max_drawdown_percent=row['max_drawdown_percent'],
            max_drawdown_duration=row['max_drawdown_duration'],
            avg_drawdown=row['avg_drawdown'],
            avg_drawdown_percent=row['avg_drawdown_percent'],
            avg_recovery_time=row['avg_recovery_time'],
            max_recovery_time=row['max_recovery_time'],
            recovery_factor=row['recovery_factor'],
            var_95=row['var_95'],
            var_99=row['var_99'],
            cvar_95=row['cvar_95'],
            ulcer_index=row['ulcer_index'],
            updated_at=to_datetime(row['updated_at']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
        )
    
    def _row_to_equity_metrics(self, row) -> EquityMetrics:
        """Convert database row to EquityMetrics."""
        equity_curve = json.loads(row['equity_curve']) if row['equity_curve'] else []
        equity_curve = [(e[0], e[1]) for e in equity_curve] if equity_curve else []
        
        return EquityMetrics(
            symbol=row['symbol'],
            timeframe=row['timeframe'],
            model_version=row['model_version'],
            strategy=row['strategy'],
            period=EvaluationPeriod(row['period']),
            initial_equity=row['initial_equity'],
            final_equity=row['final_equity'],
            peak_equity=row['peak_equity'],
            trough_equity=row['trough_equity'],
            total_return=row['total_return'],
            total_return_percent=row['total_return_percent'],
            annualized_return=row['annualized_return'],
            winning_months=row['winning_months'],
            losing_months=row['losing_months'],
            consecutive_wins=row['consecutive_wins'],
            consecutive_losses=row['consecutive_losses'],
            max_consecutive_wins=row['max_consecutive_wins'],
            max_consecutive_losses=row['max_consecutive_losses'],
            equity_curve=equity_curve,
            updated_at=to_datetime(row['updated_at']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
        )
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    def clear_cache(self):
        """Clear all caches."""
        self._prediction_cache.clear()
        self._trading_cache.clear()
        self._risk_cache.clear()
        self._equity_cache.clear()
        self._report_cache.clear()
        self.logger.debug("All validation caches cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of stored validation data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM validation_predictions")
            prediction_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM validation_trading")
            trading_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM validation_risk")
            risk_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM validation_equity")
            equity_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM validation_reports")
            report_count = cursor.fetchone()[0]
            
            return {
                'prediction_records': prediction_count,
                'trading_records': trading_count,
                'risk_records': risk_count,
                'equity_records': equity_count,
                'reports': report_count,
                'total_records': prediction_count + trading_count + risk_count + equity_count + report_count,
                'cache_size': {
                    'prediction': len(self._prediction_cache),
                    'trading': len(self._trading_cache),
                    'risk': len(self._risk_cache),
                    'equity': len(self._equity_cache),
                    'report': len(self._report_cache),
                },
            }


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_validation_statistics(config: Config) -> ValidationStatistics:
    """
    Factory function for ValidationStatistics creation.
    
    Args:
        config: Application configuration
        
    Returns:
        ValidationStatistics instance
    """
    return ValidationStatistics(config)