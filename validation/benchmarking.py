"""
validation/benchmarking.py - Performance Benchmarking Module

RESPONSIBILITY:
Benchmark AI models, strategies, and trading systems against historical data and standards.

PURPOSE:
- Evaluate model performance against benchmarks (buy & hold, random, market)
- A/B testing between model versions
- Performance comparison across different strategies
- Competitive analysis against industry standards
- Backtest validation and robustness testing

ARCHITECTURAL PRINCIPLES:
1. Repository-based - SQLite for persistence
2. Statistical rigor - Confidence intervals and significance testing
3. Extensible - Support new benchmark types
4. Type-safe - Dataclasses for all data structures
5. Production-ready - Scale to millions of backtest results

VERSION: 1.0.1
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
    'BenchmarkType',
    'BenchmarkMetric',
    'ComparisonResult',
    
    # Data classes
    'Benchmark',
    'BenchmarkResult',
    'ABTestResult',
    'PerformanceComparison',
    
    # Main class
    'BenchmarkingEngine',
    'create_benchmarking_engine',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class BenchmarkType(Enum):
    """Types of benchmarks."""
    BUY_AND_HOLD = "buy_and_hold"
    RANDOM = "random"
    MARKET = "market"
    SECTOR = "sector"
    STRATEGY = "strategy"
    MODEL_VERSION = "model_version"
    CUSTOM = "custom"


class BenchmarkMetric(Enum):
    """Metrics used for benchmarking."""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    EXPECTANCY = "expectancy"
    CONSISTENCY = "consistency"


class ComparisonResult(Enum):
    """Result of a comparison."""
    BETTER = "better"
    WORSE = "worse"
    EQUIVALENT = "equivalent"
    INCONCLUSIVE = "inconclusive"


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class Benchmark:
    """A benchmark definition."""
    benchmark_id: str
    benchmark_type: BenchmarkType
    name: str
    description: str
    symbol: str
    timeframe: str
    period: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'benchmark_id': self.benchmark_id,
            'benchmark_type': self.benchmark_type.value,
            'name': self.name,
            'description': self.description,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'period': self.period,
            'data': self.data,
            'metadata': self.metadata,
        }


@dataclass
class BenchmarkResult:
    """Result of running a benchmark."""
    benchmark_id: str
    model_version: str
    strategy: str
    symbol: str
    timeframe: str
    period: str
    
    # Model performance
    model_metrics: Dict[str, float]
    
    # Benchmark performance
    benchmark_metrics: Dict[str, float]
    
    # Comparison
    differences: Dict[str, float]
    outperformance: Dict[str, float]
    comparison_results: Dict[str, ComparisonResult]
    
    # Statistical significance
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Overall assessment
    overall_comparison: ComparisonResult
    score: float
    rank: int
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'benchmark_id': self.benchmark_id,
            'model_version': self.model_version,
            'strategy': self.strategy,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'period': self.period,
            'overall_comparison': self.overall_comparison.value,
            'score': self.score,
            'rank': self.rank,
            'timestamp': self.timestamp.isoformat(),
        }
    
    def is_better(self) -> bool:
        return self.overall_comparison == ComparisonResult.BETTER
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        for metric, pvalue in self.p_values.items():
            if pvalue is not None and pvalue > alpha:
                return False
        return True


@dataclass
class ABTestResult:
    """A/B test results between two model versions."""
    test_id: str
    model_a: str
    model_b: str
    symbol: str
    timeframe: str
    period: str
    
    # Performance comparison
    a_metrics: Dict[str, float]
    b_metrics: Dict[str, float]
    differences: Dict[str, float]
    
    # Statistical significance
    p_values: Dict[str, float]
    confidence: float
    
    # Winner
    winner: str
    winning_metrics: Dict[str, str]
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_winner(self) -> str:
        return self.winner
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        for pvalue in self.p_values.values():
            if pvalue is not None and pvalue > alpha:
                return False
        return True


@dataclass
class PerformanceComparison:
    """Comprehensive performance comparison."""
    comparison_id: str
    name: str
    description: str
    items: List[Dict[str, Any]]
    metrics: Dict[str, Dict[str, float]]
    rankings: Dict[str, int]
    overall_scores: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_rank(self, item_id: str) -> int:
        return self.rankings.get(item_id, -1)
    
    def get_score(self, item_id: str) -> float:
        return self.overall_scores.get(item_id, 0.0)
    
    def get_top_items(self, n: int = 5) -> List[Tuple[str, float]]:
        sorted_items = sorted(
            self.overall_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_items[:n]


# ==============================================================================
# BENCHMARKING ENGINE
# ==============================================================================

class BenchmarkingEngine:
    """
    Performance benchmarking engine.
    
    Benchmarks AI models, strategies, and trading systems.
    """
    
    # Database schema version
    SCHEMA_VERSION = 1
    
    # Default benchmarks
    DEFAULT_BENCHMARKS = ['buy_and_hold', 'random', 'market_average']
    
    # Metric weights for overall score
    METRIC_WEIGHTS = {
        'sharpe_ratio': 0.25,
        'total_return': 0.20,
        'win_rate': 0.15,
        'profit_factor': 0.15,
        'max_drawdown': 0.15,
        'consistency': 0.10,
    }
    
    def __init__(self, config: Config):
        """
        Initialize the benchmarking engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = getattr(config, 'DB_PATH', 'market_ai.db')
        
        # Caches
        self._benchmark_cache: Dict[str, Benchmark] = {}
        self._result_cache: Dict[str, BenchmarkResult] = {}
        self._abtest_cache: Dict[str, ABTestResult] = {}
        self._comparison_cache: Dict[str, PerformanceComparison] = {}
        
        # Initialize database
        self._init_database()
        
        self.logger.info("✅ BenchmarkingEngine initialized")
    
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
            
            # Benchmarks
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmarking_benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    benchmark_id TEXT UNIQUE NOT NULL,
                    benchmark_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    period TEXT,
                    data TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            # Benchmark results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmarking_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id TEXT UNIQUE NOT NULL,
                    benchmark_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    period TEXT NOT NULL,
                    model_metrics TEXT,
                    benchmark_metrics TEXT,
                    differences TEXT,
                    outperformance TEXT,
                    comparison_results TEXT,
                    p_values TEXT,
                    confidence_intervals TEXT,
                    overall_comparison TEXT,
                    score REAL,
                    rank INTEGER,
                    metadata TEXT,
                    timestamp TIMESTAMP
                )
            """)
            
            # A/B test results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmarking_abtests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT UNIQUE NOT NULL,
                    model_a TEXT NOT NULL,
                    model_b TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    period TEXT NOT NULL,
                    a_metrics TEXT,
                    b_metrics TEXT,
                    differences TEXT,
                    p_values TEXT,
                    confidence REAL,
                    winner TEXT,
                    winning_metrics TEXT,
                    metadata TEXT,
                    timestamp TIMESTAMP
                )
            """)
            
            # Performance comparisons
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmarking_comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comparison_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    items TEXT,
                    metrics TEXT,
                    rankings TEXT,
                    overall_scores TEXT,
                    metadata TEXT,
                    timestamp TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_results_model ON benchmarking_results(model_version)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_results_symbol ON benchmarking_results(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_abtests_models ON benchmarking_abtests(model_a, model_b)")
            
            self.logger.info("✅ Benchmarking database schema initialized")
    
    # ==========================================================================
    # BENCHMARK MANAGEMENT
    # ==========================================================================
    
    def create_benchmark(
        self,
        benchmark_type: BenchmarkType,
        name: str,
        description: str,
        symbol: str,
        timeframe: str,
        period: str,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Benchmark:
        """
        Create a new benchmark.
        
        Args:
            benchmark_type: Type of benchmark
            name: Benchmark name
            description: Benchmark description
            symbol: Symbol to benchmark against
            timeframe: Timeframe
            period: Period string (e.g., '1Y', '3Y')
            data: Benchmark data
            metadata: Additional metadata
            
        Returns:
            Benchmark object
        """
        benchmark_id = f"{symbol}_{timeframe}_{period}_{int(datetime.now().timestamp())}"
        
        benchmark = Benchmark(
            benchmark_id=benchmark_id,
            benchmark_type=benchmark_type,
            name=name,
            description=description,
            symbol=symbol,
            timeframe=timeframe,
            period=period,
            data=data or {},
            metadata=metadata or {},
        )
        
        self._save_benchmark(benchmark)
        self._benchmark_cache[benchmark_id] = benchmark
        
        self.logger.info(f"✅ Created benchmark: {name} ({benchmark_id})")
        return benchmark
    
    def get_benchmark(self, benchmark_id: str) -> Optional[Benchmark]:
        """Get benchmark by ID."""
        if benchmark_id in self._benchmark_cache:
            return self._benchmark_cache[benchmark_id]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM benchmarking_benchmarks WHERE benchmark_id = ?
            """, (benchmark_id,))
            row = cursor.fetchone()
            
            if row:
                benchmark = Benchmark(
                    benchmark_id=row['benchmark_id'],
                    benchmark_type=BenchmarkType(row['benchmark_type']),
                    name=row['name'],
                    description=row['description'],
                    symbol=row['symbol'],
                    timeframe=row['timeframe'],
                    period=row['period'],
                    data=json.loads(row['data']) if row['data'] else {},
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    created_at=to_datetime(row['created_at']),
                    updated_at=to_datetime(row['updated_at']),
                )
                self._benchmark_cache[benchmark_id] = benchmark
                return benchmark
        
        return None
    
    def get_default_benchmarks(self) -> List[Benchmark]:
        """Get default benchmarks."""
        default_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD']
        
        for name in self.DEFAULT_BENCHMARKS:
            for symbol in default_symbols:
                # Check if benchmark exists
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT benchmark_id FROM benchmarking_benchmarks 
                        WHERE name = ? AND symbol = ?
                    """, (name.replace('_', ' ').title(), symbol))
                    row = cursor.fetchone()
                    if not row:
                        self.create_benchmark(
                            benchmark_type=BenchmarkType[name.upper()],
                            name=f"{name.replace('_', ' ').title()} {symbol}",
                            description=f"Default {name.replace('_', ' ').title()} benchmark for {symbol}",
                            symbol=symbol,
                            timeframe='D1',
                            period='1Y',
                            data={'benchmark': name, 'symbol': symbol},
                        )
        
        # Return all benchmarks
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT benchmark_id FROM benchmarking_benchmarks")
            benchmarks = []
            for row in cursor.fetchall():
                benchmark = self.get_benchmark(row['benchmark_id'])
                if benchmark:
                    benchmarks.append(benchmark)
            return benchmarks
    
    # ==========================================================================
    # BENCHMARK EXECUTION
    # ==========================================================================
    
    def run_benchmark(
        self,
        benchmark_id: str,
        model_version: str,
        strategy: str,
        model_performance: Dict[str, float],
        benchmark_performance: Optional[Dict[str, float]] = None,
    ) -> BenchmarkResult:
        """
        Run a benchmark against a model.
        
        Args:
            benchmark_id: Benchmark ID
            model_version: Model version being tested
            strategy: Strategy name
            model_performance: Model performance metrics
            benchmark_performance: Benchmark performance metrics
            
        Returns:
            BenchmarkResult object
        """
        # Get benchmark
        benchmark = self.get_benchmark(benchmark_id)
        if not benchmark:
            raise DataValidationError(f"Benchmark {benchmark_id} not found")
        
        # Get benchmark performance
        if benchmark_performance is None:
            benchmark_performance = benchmark.data.get('performance', {})
        
        # Validate that we have some data
        if not benchmark_performance:
            self.logger.warning(f"Benchmark {benchmark_id} has no performance data")
            benchmark_performance = {k: 0.0 for k in model_performance.keys()}
        
        # Calculate differences
        differences = {}
        outperformance = {}
        comparison_results = {}
        p_values = {}
        confidence_intervals = {}
        
        all_metrics = set(model_performance.keys()) | set(benchmark_performance.keys())
        
        for metric in all_metrics:
            model_val = model_performance.get(metric, 0.0)
            bench_val = benchmark_performance.get(metric, 0.0)
            
            differences[metric] = model_val - bench_val
            
            if bench_val != 0:
                outperformance[metric] = (model_val - bench_val) / abs(bench_val)
            else:
                outperformance[metric] = 0.0 if model_val == 0 else 1.0
            
            # Determine comparison result
            if model_val > bench_val * 1.05:
                comparison_results[metric] = ComparisonResult.BETTER
            elif model_val < bench_val * 0.95:
                comparison_results[metric] = ComparisonResult.WORSE
            elif abs(model_val - bench_val) <= 0.01:
                comparison_results[metric] = ComparisonResult.EQUIVALENT
            else:
                comparison_results[metric] = ComparisonResult.INCONCLUSIVE
            
            # Approximate p-value
            p_values[metric] = self._calculate_p_value(differences[metric], model_val, bench_val)
            
            # Confidence interval
            confidence_intervals[metric] = (
                model_val - abs(differences[metric]) * 0.5,
                model_val + abs(differences[metric]) * 0.5
            )
        
        # Overall comparison
        better_count = sum(1 for r in comparison_results.values() if r == ComparisonResult.BETTER)
        worse_count = sum(1 for r in comparison_results.values() if r == ComparisonResult.WORSE)
        
        if better_count > worse_count:
            overall_comparison = ComparisonResult.BETTER
        elif worse_count > better_count:
            overall_comparison = ComparisonResult.WORSE
        else:
            overall_comparison = ComparisonResult.EQUIVALENT
        
        # Calculate score
        score = self._calculate_score(model_performance)
        
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            model_version=model_version,
            strategy=strategy,
            symbol=benchmark.symbol,
            timeframe=benchmark.timeframe,
            period=benchmark.period,
            model_metrics=model_performance,
            benchmark_metrics=benchmark_performance,
            differences=differences,
            outperformance=outperformance,
            comparison_results=comparison_results,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            overall_comparison=overall_comparison,
            score=score,
            rank=0,
            metadata={
                'benchmark_name': benchmark.name,
                'benchmark_type': benchmark.benchmark_type.value,
            },
        )
        
        # Save result
        self._save_benchmark_result(result)
        self._result_cache[result.benchmark_id] = result
        
        self.logger.info(
            f"✅ Benchmark complete: {benchmark.name} -> {overall_comparison.value} "
            f"(score: {score:.3f})"
        )
        return result
    
    def compare_models(
        self,
        model_a: str,
        model_b: str,
        symbol: str,
        timeframe: str,
        period: str,
        model_a_metrics: Dict[str, float],
        model_b_metrics: Dict[str, float],
        confidence_level: float = 0.95,
    ) -> ABTestResult:
        """
        Perform A/B test between two models.
        
        Args:
            model_a: Model A version
            model_b: Model B version
            symbol: Symbol
            timeframe: Timeframe
            period: Period string
            model_a_metrics: Model A performance metrics
            model_b_metrics: Model B performance metrics
            confidence_level: Confidence level for significance
            
        Returns:
            ABTestResult object
        """
        test_id = f"abtest_{model_a}_vs_{model_b}_{int(datetime.now().timestamp())}"
        
        # Calculate differences
        differences = {}
        p_values = {}
        winning_metrics = {}
        
        all_metrics = set(model_a_metrics.keys()) | set(model_b_metrics.keys())
        
        for metric in all_metrics:
            a_val = model_a_metrics.get(metric, 0.0)
            b_val = model_b_metrics.get(metric, 0.0)
            differences[metric] = a_val - b_val
            
            # Significance
            p_values[metric] = self._calculate_p_value(differences[metric], a_val, b_val)
            
            # Determine winner for this metric
            if a_val > b_val * 1.05:
                winning_metrics[metric] = 'A'
            elif b_val > a_val * 1.05:
                winning_metrics[metric] = 'B'
            else:
                winning_metrics[metric] = 'TIE'
        
        # Determine overall winner
        a_wins = sum(1 for w in winning_metrics.values() if w == 'A')
        b_wins = sum(1 for w in winning_metrics.values() if w == 'B')
        
        if a_wins > b_wins:
            winner = 'A'
        elif b_wins > a_wins:
            winner = 'B'
        else:
            winner = 'TIE'
        
        # Calculate confidence
        valid_pvalues = [p for p in p_values.values() if p is not None and not math.isnan(p)]
        avg_pvalue = sum(valid_pvalues) / len(valid_pvalues) if valid_pvalues else 0.5
        confidence = 1.0 - avg_pvalue
        
        result = ABTestResult(
            test_id=test_id,
            model_a=model_a,
            model_b=model_b,
            symbol=symbol,
            timeframe=timeframe,
            period=period,
            a_metrics=model_a_metrics,
            b_metrics=model_b_metrics,
            differences=differences,
            p_values=p_values,
            confidence=confidence,
            winner=winner,
            winning_metrics=winning_metrics,
            metadata={
                'confidence_level': confidence_level,
                'a_wins': a_wins,
                'b_wins': b_wins,
            },
        )
        
        self._save_abtest_result(result)
        self._abtest_cache[test_id] = result
        
        self.logger.info(
            f"✅ A/B Test complete: {model_a} vs {model_b} -> {winner} "
            f"(confidence: {confidence:.2%})"
        )
        return result
    
    def compare_multiple(
        self,
        name: str,
        description: str,
        items: List[Dict[str, Any]],
        metrics_data: Dict[str, Dict[str, float]],
    ) -> PerformanceComparison:
        """
        Compare multiple models/strategies.
        
        Args:
            name: Comparison name
            description: Comparison description
            items: List of items {id, name, type}
            metrics_data: Dictionary mapping item_id to metrics
            
        Returns:
            PerformanceComparison object
        """
        comparison_id = f"comp_{int(datetime.now().timestamp())}"
        
        # Calculate scores for each item
        scores = {}
        rankings = {}
        
        for item in items:
            item_id = item['id']
            metrics = metrics_data.get(item_id, {})
            scores[item_id] = self._calculate_score(metrics)
        
        # Rank items
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (item_id, _) in enumerate(sorted_items, 1):
            rankings[item_id] = rank
        
        result = PerformanceComparison(
            comparison_id=comparison_id,
            name=name,
            description=description,
            items=items,
            metrics=metrics_data,
            rankings=rankings,
            overall_scores=scores,
        )
        
        self._save_comparison(result)
        self._comparison_cache[comparison_id] = result
        
        self.logger.info(f"✅ Comparison complete: {name} ({len(items)} items)")
        return result
    
    # ==========================================================================
    # GET METHODS
    # ==========================================================================
    
    def get_benchmark_result(self, benchmark_id: str) -> Optional[BenchmarkResult]:
        """Get benchmark result."""
        if benchmark_id in self._result_cache:
            return self._result_cache[benchmark_id]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM benchmarking_results WHERE benchmark_id = ?
            """, (benchmark_id,))
            row = cursor.fetchone()
            
            if row:
                result = self._row_to_benchmark_result(row)
                self._result_cache[benchmark_id] = result
                return result
        
        return None
    
    def get_abtest_result(self, test_id: str) -> Optional[ABTestResult]:
        """Get A/B test result."""
        if test_id in self._abtest_cache:
            return self._abtest_cache[test_id]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM benchmarking_abtests WHERE test_id = ?
            """, (test_id,))
            row = cursor.fetchone()
            
            if row:
                result = self._row_to_abtest_result(row)
                self._abtest_cache[test_id] = result
                return result
        
        return None
    
    def get_comparison(self, comparison_id: str) -> Optional[PerformanceComparison]:
        """Get performance comparison."""
        if comparison_id in self._comparison_cache:
            return self._comparison_cache[comparison_id]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM benchmarking_comparisons WHERE comparison_id = ?
            """, (comparison_id,))
            row = cursor.fetchone()
            
            if row:
                result = self._row_to_comparison(row)
                self._comparison_cache[comparison_id] = result
                return result
        
        return None
    
    # ==========================================================================
    # SAVE METHODS
    # ==========================================================================
    
    def _save_benchmark(self, benchmark: Benchmark):
        """Save benchmark to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO benchmarking_benchmarks
                (benchmark_id, benchmark_type, name, description, symbol,
                 timeframe, period, data, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                benchmark.benchmark_id,
                benchmark.benchmark_type.value,
                benchmark.name,
                benchmark.description,
                benchmark.symbol,
                benchmark.timeframe,
                benchmark.period,
                json.dumps(benchmark.data),
                json.dumps(benchmark.metadata),
                benchmark.created_at.isoformat(),
                benchmark.updated_at.isoformat(),
            ))
    
    def _save_benchmark_result(self, result: BenchmarkResult):
        """Save benchmark result to database."""
        result_id = f"res_{result.benchmark_id}_{int(datetime.now().timestamp())}"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO benchmarking_results
                (result_id, benchmark_id, model_version, strategy, symbol,
                 timeframe, period, model_metrics, benchmark_metrics,
                 differences, outperformance, comparison_results,
                 p_values, confidence_intervals, overall_comparison,
                 score, rank, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result_id,
                result.benchmark_id,
                result.model_version,
                result.strategy,
                result.symbol,
                result.timeframe,
                result.period,
                json.dumps(result.model_metrics),
                json.dumps(result.benchmark_metrics),
                json.dumps(result.differences),
                json.dumps(result.outperformance),
                json.dumps({k: v.value for k, v in result.comparison_results.items()}),
                json.dumps(result.p_values),
                json.dumps(result.confidence_intervals),
                result.overall_comparison.value,
                result.score,
                result.rank,
                json.dumps(result.metadata),
                result.timestamp.isoformat(),
            ))
    
    def _save_abtest_result(self, result: ABTestResult):
        """Save A/B test result to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO benchmarking_abtests
                (test_id, model_a, model_b, symbol, timeframe, period,
                 a_metrics, b_metrics, differences, p_values,
                 confidence, winner, winning_metrics, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.test_id,
                result.model_a,
                result.model_b,
                result.symbol,
                result.timeframe,
                result.period,
                json.dumps(result.a_metrics),
                json.dumps(result.b_metrics),
                json.dumps(result.differences),
                json.dumps(result.p_values),
                result.confidence,
                result.winner,
                json.dumps(result.winning_metrics),
                json.dumps(result.metadata),
                result.timestamp.isoformat(),
            ))
    
    def _save_comparison(self, comparison: PerformanceComparison):
        """Save performance comparison to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO benchmarking_comparisons
                (comparison_id, name, description, items, metrics,
                 rankings, overall_scores, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                comparison.comparison_id,
                comparison.name,
                comparison.description,
                json.dumps(comparison.items),
                json.dumps(comparison.metrics),
                json.dumps(comparison.rankings),
                json.dumps(comparison.overall_scores),
                json.dumps(comparison.metadata),
                comparison.timestamp.isoformat(),
            ))
    
    # ==========================================================================
    # ROW TO OBJECT METHODS
    # ==========================================================================
    
    def _row_to_benchmark_result(self, row) -> BenchmarkResult:
        """Convert database row to BenchmarkResult."""
        comparison_results = {}
        for k, v in json.loads(row['comparison_results']).items():
            comparison_results[k] = ComparisonResult(v)
        
        return BenchmarkResult(
            benchmark_id=row['benchmark_id'],
            model_version=row['model_version'],
            strategy=row['strategy'],
            symbol=row['symbol'],
            timeframe=row['timeframe'],
            period=row['period'],
            model_metrics=json.loads(row['model_metrics']) if row['model_metrics'] else {},
            benchmark_metrics=json.loads(row['benchmark_metrics']) if row['benchmark_metrics'] else {},
            differences=json.loads(row['differences']) if row['differences'] else {},
            outperformance=json.loads(row['outperformance']) if row['outperformance'] else {},
            comparison_results=comparison_results,
            p_values=json.loads(row['p_values']) if row['p_values'] else {},
            confidence_intervals=json.loads(row['confidence_intervals']) if row['confidence_intervals'] else {},
            overall_comparison=ComparisonResult(row['overall_comparison']),
            score=row['score'],
            rank=row['rank'],
            timestamp=to_datetime(row['timestamp']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
        )
    
    def _row_to_abtest_result(self, row) -> ABTestResult:
        """Convert database row to ABTestResult."""
        return ABTestResult(
            test_id=row['test_id'],
            model_a=row['model_a'],
            model_b=row['model_b'],
            symbol=row['symbol'],
            timeframe=row['timeframe'],
            period=row['period'],
            a_metrics=json.loads(row['a_metrics']) if row['a_metrics'] else {},
            b_metrics=json.loads(row['b_metrics']) if row['b_metrics'] else {},
            differences=json.loads(row['differences']) if row['differences'] else {},
            p_values=json.loads(row['p_values']) if row['p_values'] else {},
            confidence=row['confidence'],
            winner=row['winner'],
            winning_metrics=json.loads(row['winning_metrics']) if row['winning_metrics'] else {},
            timestamp=to_datetime(row['timestamp']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
        )
    
    def _row_to_comparison(self, row) -> PerformanceComparison:
        """Convert database row to PerformanceComparison."""
        return PerformanceComparison(
            comparison_id=row['comparison_id'],
            name=row['name'],
            description=row['description'],
            items=json.loads(row['items']) if row['items'] else [],
            metrics=json.loads(row['metrics']) if row['metrics'] else {},
            rankings=json.loads(row['rankings']) if row['rankings'] else {},
            overall_scores=json.loads(row['overall_scores']) if row['overall_scores'] else {},
            timestamp=to_datetime(row['timestamp']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
        )
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    def _calculate_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall score from metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Score between 0 and 1
        """
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in self.METRIC_WEIGHTS.items():
            if metric in metrics and metrics[metric] is not None:
                value = metrics[metric]
                
                # Validate value
                if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                    continue
                
                # Normalize based on metric type
                if metric == 'max_drawdown':
                    # Lower is better, cap at 50% for normalization
                    normalized = max(0.0, min(1.0, 1.0 - abs(value) / 0.5))
                elif metric in ['sharpe_ratio', 'profit_factor', 'consistency']:
                    # Higher is better, cap at 2.0 for normalization
                    normalized = max(0.0, min(1.0, value / 2.0))
                elif metric in ['total_return', 'win_rate']:
                    # Higher is better
                    normalized = max(0.0, min(1.0, value))
                else:
                    normalized = max(0.0, min(1.0, value))
                
                score += normalized * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return min(1.0, max(0.0, score / total_weight))
    
    def _calculate_p_value(self, difference: float, value_a: float, value_b: float) -> float:
        """
        Approximate p-value for a difference between two values.
        
        Uses a simplified approximation based on the relative difference.
        
        Args:
            difference: Difference between values
            value_a: First value
            value_b: Second value
            
        Returns:
            Approximate p-value between 0 and 1
        """
        # If values are identical, p-value is 1.0
        if value_a == value_b:
            return 1.0
        
        # Calculate relative difference
        avg_val = (abs(value_a) + abs(value_b)) / 2
        if avg_val == 0:
            return 1.0
        
        rel_diff = abs(difference) / avg_val
        
        # Convert relative difference to p-value using exponential decay
        # Small differences = high p-value, large differences = low p-value
        p_value = math.exp(-rel_diff * 2.0)
        
        # Clamp to [0, 1]
        return min(1.0, max(0.0, p_value))
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    def clear_cache(self):
        """Clear all caches."""
        self._benchmark_cache.clear()
        self._result_cache.clear()
        self._abtest_cache.clear()
        self._comparison_cache.clear()
        self.logger.debug("All benchmarking caches cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of stored benchmarking data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM benchmarking_benchmarks")
            benchmark_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM benchmarking_results")
            result_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM benchmarking_abtests")
            abtest_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM benchmarking_comparisons")
            comparison_count = cursor.fetchone()[0]
            
            return {
                'benchmarks': benchmark_count,
                'results': result_count,
                'ab_tests': abtest_count,
                'comparisons': comparison_count,
                'total_records': benchmark_count + result_count + abtest_count + comparison_count,
                'cache_size': {
                    'benchmark': len(self._benchmark_cache),
                    'result': len(self._result_cache),
                    'abtest': len(self._abtest_cache),
                    'comparison': len(self._comparison_cache),
                },
            }


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_benchmarking_engine(config: Config) -> BenchmarkingEngine:
    """
    Factory function for BenchmarkingEngine creation.
    
    Args:
        config: Application configuration
        
    Returns:
        BenchmarkingEngine instance
    """
    return BenchmarkingEngine(config)