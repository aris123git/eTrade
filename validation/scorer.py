"""
validation/scorer.py - Pattern Scoring Module

RESPONSIBILITY:
Score and rank discovered patterns based on multiple criteria.

ARCHITECTURAL PRINCIPLES:
1. Pure pattern scoring - No data storage, no I/O, no business logic
2. Multi-criteria scoring of patterns
3. Type-safe results with validation
4. Configurable scoring weights

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions

VERSION: 1.0.0
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Set, Union, Callable
from enum import Enum
from collections import defaultdict

from core.config import Config
from core.exceptions import ValidationError, DataValidationError


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'ScoreMetric',
    'ScoreResult',
    'Scorer',
    'create_scorer',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class ScoreMetric(Enum):
    """Metrics used for scoring patterns."""
    # Statistical metrics
    CONFIDENCE = "confidence"
    P_VALUE = "p_value"
    EFFECT_SIZE = "effect_size"
    SAMPLE_SIZE = "sample_size"
    
    # Performance metrics
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    
    # Pattern quality metrics
    STRENGTH = "strength"
    RELIABILITY = "reliability"
    CONSISTENCY = "consistency"
    SPECIFICITY = "specificity"
    
    # Risk metrics
    RISK_REWARD = "risk_reward"
    EXPECTED_VALUE = "expected_value"
    CALMAR_RATIO = "calmar_ratio"
    
    # Market metrics
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class ScoreResult:
    """Result of pattern scoring."""
    pattern_id: str
    symbol: str
    timeframe: str
    timestamp: datetime
    overall_score: float
    metrics: Dict[str, float]
    weights: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_metric(self, metric: Union[ScoreMetric, str]) -> Optional[float]:
        """Get a specific metric value."""
        key = metric.value if isinstance(metric, ScoreMetric) else metric
        return self.metrics.get(key)
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if pattern is high quality."""
        return self.overall_score >= threshold
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of scoring."""
        return {
            'pattern_id': self.pattern_id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'overall_score': self.overall_score,
            'top_metrics': sorted(
                self.metrics.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
        }


# ==============================================================================
# SCORER
# ==============================================================================

class Scorer:
    """
    Pattern scoring engine.
    
    Scores and ranks patterns based on multiple criteria.
    """
    
    # Default weights for metrics
    DEFAULT_WEIGHTS = {
        'confidence': 0.20,
        'win_rate': 0.15,
        'sharpe_ratio': 0.15,
        'profit_factor': 0.15,
        'risk_reward': 0.10,
        'consistency': 0.10,
        'sample_size': 0.10,
        'max_drawdown': 0.05,
    }
    
    def __init__(self, config: Config):
        """
        Initialize the scorer.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Weights
        self._weights = self._load_weights(config)
        
        self.logger.info(
            f"✅ Scorer initialized with {len(self._weights)} metrics"
        )
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def score_pattern(
        self,
        pattern_data: Dict[str, Any],
        symbol: str,
        timeframe: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> ScoreResult:
        """
        Score a single pattern.
        
        Args:
            pattern_data: Pattern data dictionary
            symbol: Symbol name
            timeframe: Timeframe
            weights: Custom weights
            
        Returns:
            ScoreResult object
        """
        if not pattern_data:
            raise DataValidationError("No pattern data provided")
        
        # Use custom weights if provided, else defaults
        weights = weights or self._weights
        
        # Extract or compute metrics
        metrics = self._extract_metrics(pattern_data)
        
        # Normalize metrics
        normalized = self._normalize_metrics(metrics)
        
        # Calculate overall score
        overall_score = self._calculate_score(normalized, weights)
        
        result = ScoreResult(
            pattern_id=pattern_data.get('pattern_id', 'unknown'),
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            overall_score=overall_score,
            metrics=metrics,
            weights=weights,
            metadata={
                'pattern_type': pattern_data.get('pattern_type'),
                'candle_count': pattern_data.get('candle_count', 0),
            },
        )
        
        self.logger.debug(
            f"Pattern scored: {result.pattern_id} -> {overall_score:.3f}"
        )
        
        return result
    
    def score_patterns(
        self,
        patterns: List[Dict[str, Any]],
        symbol: str,
        timeframe: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[ScoreResult]:
        """
        Score multiple patterns.
        
        Args:
            patterns: List of pattern data dictionaries
            symbol: Symbol name
            timeframe: Timeframe
            weights: Custom weights
            
        Returns:
            List of ScoreResult objects
        """
        if not patterns:
            return []
        
        results = []
        for pattern in patterns:
            try:
                result = self.score_pattern(
                    pattern, symbol, timeframe, weights
                )
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to score pattern: {e}")
        
        # Sort by overall score
        results.sort(key=lambda r: r.overall_score, reverse=True)
        
        return results
    
    def rank_patterns(
        self,
        results: List[ScoreResult],
    ) -> List[ScoreResult]:
        """
        Rank patterns by score.
        
        Args:
            results: List of ScoreResult objects
            
        Returns:
            Ranked list of ScoreResult objects
        """
        return sorted(results, key=lambda r: r.overall_score, reverse=True)
    
    def get_best_patterns(
        self,
        results: List[ScoreResult],
        limit: int = 10,
        min_score: float = 0.5,
    ) -> List[ScoreResult]:
        """
        Get the best patterns.
        
        Args:
            results: List of ScoreResult objects
            limit: Maximum number to return
            min_score: Minimum score threshold
            
        Returns:
            List of best ScoreResult objects
        """
        filtered = [r for r in results if r.overall_score >= min_score]
        ranked = self.rank_patterns(filtered)
        return ranked[:limit]
    
    def update_weights(self, weights: Dict[str, float]) -> None:
        """
        Update scoring weights.
        
        Args:
            weights: New weights
        """
        # Validate weights
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self._weights = weights
        self.logger.info(f"Weights updated: {weights}")
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self._weights.copy()
    
    def compare_patterns(
        self,
        pattern1: ScoreResult,
        pattern2: ScoreResult,
    ) -> Dict[str, Any]:
        """
        Compare two patterns.
        
        Args:
            pattern1: First ScoreResult
            pattern2: Second ScoreResult
            
        Returns:
            Comparison dictionary
        """
        return {
            'better_pattern': pattern1.pattern_id if pattern1.overall_score > pattern2.overall_score else pattern2.pattern_id,
            'score_diff': abs(pattern1.overall_score - pattern2.overall_score),
            'metric_comparisons': {
                metric: {
                    'pattern1': pattern1.metrics.get(metric, 0),
                    'pattern2': pattern2.metrics.get(metric, 0),
                    'diff': pattern1.metrics.get(metric, 0) - pattern2.metrics.get(metric, 0),
                }
                for metric in set(pattern1.metrics.keys()) | set(pattern2.metrics.keys())
            },
        }
    
    def get_statistics(
        self,
        results: List[ScoreResult],
    ) -> Dict[str, Any]:
        """
        Get statistics from scoring results.
        
        Args:
            results: List of ScoreResult objects
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {
                'total_patterns': 0,
                'avg_score': 0.0,
                'max_score': 0.0,
                'min_score': 0.0,
                'std_score': 0.0,
            }
        
        scores = [r.overall_score for r in results]
        
        return {
            'total_patterns': len(results),
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'std_score': math.sqrt(
                sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)
            ) if len(scores) > 1 else 0.0,
            'high_quality': len([r for r in results if r.is_high_quality()]),
            'thresholds': {
                '0.5': len([r for r in results if r.overall_score >= 0.5]),
                '0.6': len([r for r in results if r.overall_score >= 0.6]),
                '0.7': len([r for r in results if r.overall_score >= 0.7]),
                '0.8': len([r for r in results if r.overall_score >= 0.8]),
                '0.9': len([r for r in results if r.overall_score >= 0.9]),
            },
        }
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _load_weights(self, config: Config) -> Dict[str, float]:
        """Load weights from config."""
        weights = self.DEFAULT_WEIGHTS.copy()
        
        # Override from config
        if hasattr(config, 'SCORER_WEIGHTS'):
            custom_weights = getattr(config, 'SCORER_WEIGHTS')
            if isinstance(custom_weights, dict):
                weights.update(custom_weights)
        
        # Normalize
        total = sum(weights.values())
        if total != 1.0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _extract_metrics(self, pattern_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract metrics from pattern data.
        
        Args:
            pattern_data: Pattern data dictionary
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['confidence'] = pattern_data.get('confidence', 0.5)
        metrics['strength'] = pattern_data.get('strength', 0.5)
        
        # Statistical metrics
        if 'p_value' in pattern_data:
            metrics['p_value'] = pattern_data['p_value']
        
        if 'sample_size' in pattern_data:
            metrics['sample_size'] = min(pattern_data['sample_size'] / 100, 1.0)
        
        # Performance metrics
        if 'win_rate' in pattern_data:
            metrics['win_rate'] = pattern_data['win_rate']
        elif 'outcomes' in pattern_data:
            outcomes = pattern_data['outcomes']
            total = sum(outcomes.values())
            if total > 0:
                win_rate = outcomes.get('win', 0) / total
                metrics['win_rate'] = win_rate
        
        # Sharpe ratio
        if 'returns' in pattern_data:
            returns = pattern_data['returns']
            if len(returns) > 1:
                avg_return = sum(returns) / len(returns)
                std_return = math.sqrt(
                    sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
                ) if len(returns) > 1 else 1.0
                metrics['sharpe_ratio'] = avg_return / std_return if std_return > 0 else 0.0
        
        # Profit factor
        if 'profits' in pattern_data and 'losses' in pattern_data:
            total_profit = pattern_data['profits']
            total_loss = abs(pattern_data['losses'])
            metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else 0.0
        
        # Risk-reward ratio
        if 'avg_win' in pattern_data and 'avg_loss' in pattern_data:
            avg_win = pattern_data['avg_win']
            avg_loss = abs(pattern_data['avg_loss'])
            metrics['risk_reward'] = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # Max drawdown
        if 'max_drawdown' in pattern_data:
            metrics['max_drawdown'] = min(pattern_data['max_drawdown'] / 100, 1.0)
        
        # Consistency
        if 'outcomes' in pattern_data:
            outcomes = pattern_data['outcomes']
            total = sum(outcomes.values())
            if total > 0:
                win_rate = outcomes.get('win', 0) / total
                # Consistency is high when win rate is balanced
                consistency = 1.0 - abs(win_rate - 0.5) * 2
                metrics['consistency'] = max(0, min(1, consistency))
        
        # Effect size
        if 'effect_size' in pattern_data:
            metrics['effect_size'] = min(abs(pattern_data['effect_size']), 1.0)
        
        # Expected value
        if 'avg_win' in pattern_data and 'avg_loss' in pattern_data and 'win_rate' in metrics:
            avg_win = pattern_data['avg_win']
            avg_loss = abs(pattern_data['avg_loss'])
            win_rate = metrics['win_rate']
            expected_value = win_rate * avg_win - (1 - win_rate) * avg_loss
            metrics['expected_value'] = expected_value
        
        # Correlation
        if 'correlation' in pattern_data:
            metrics['correlation'] = abs(pattern_data['correlation'])
        
        return metrics
    
    def _normalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize metrics to [0, 1] range.
        
        Args:
            metrics: Raw metrics
            
        Returns:
            Normalized metrics
        """
        normalized = {}
        
        for key, value in metrics.items():
            if key == 'p_value':
                # p-value: lower is better
                normalized[key] = 1.0 - min(value, 1.0)
            elif key == 'max_drawdown':
                # Max drawdown: lower is better
                normalized[key] = 1.0 - min(value, 1.0)
            elif key == 'sample_size':
                # Sample size: cap at 100
                normalized[key] = min(value / 100, 1.0)
            elif key == 'correlation':
                # Correlation: absolute value
                normalized[key] = min(abs(value), 1.0)
            else:
                # Default: cap at 1.0
                normalized[key] = min(value, 1.0)
        
        return normalized
    
    def _calculate_score(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        """
        Calculate overall score.
        
        Args:
            metrics: Normalized metrics
            weights: Scoring weights
            
        Returns:
            Overall score (0-1)
        """
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                total_score += metrics[metric] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_scorer(config: Config) -> Scorer:
    """
    Factory function for Scorer creation.
    
    Args:
        config: Application configuration
        
    Returns:
        Scorer instance
    """
    return Scorer(config)