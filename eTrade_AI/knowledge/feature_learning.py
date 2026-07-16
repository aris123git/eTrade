"""
discovery/feature_learning.py - Feature Learning Engine

RESPONSIBILITY:
Learn and extract meaningful features from market data using unsupervised and supervised methods.

ARCHITECTURAL PRINCIPLES:
1. Pure feature learning - No data storage, no I/O, no business logic
2. Unsupervised and supervised feature extraction
3. Type-safe results with validation
4. Multiple feature learning methods (PCA, Autoencoder, etc.)

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions

VERSION: 1.0.0
"""

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Set, Union, Callable
from enum import Enum
from collections import defaultdict

from core.config import Config
from core.exceptions import DiscoveryError, DataValidationError


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'LearningMethod',
    'FeatureInfo',
    'FeatureLearningResult',
    'FeatureLearningEngine',
    'create_feature_learning_engine',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class LearningMethod(Enum):
    """Method for feature learning."""
    PCA = "pca"                     # Principal Component Analysis
    ICA = "ica"                     # Independent Component Analysis
    AUTOENCODER = "autoencoder"     # Autoencoder (neural network)
    TSNE = "tsne"                   # t-SNE dimensionality reduction
    UMAP = "umap"                   # UMAP dimensionality reduction
    RANDOM_PROJECTION = "random_projection"
    ISOMAP = "isomap"
    LDA = "lda"                     # Linear Discriminant Analysis


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class FeatureInfo:
    """Information about a learned feature."""
    feature_id: int
    name: str
    variance_explained: float
    importance: float
    coefficients: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureLearningResult:
    """Result of feature learning operation."""
    symbol: str
    timestamp: datetime
    features: List[FeatureInfo]
    transformed_data: List[List[float]]
    original_dim: int
    reduced_dim: int
    method: LearningMethod
    variance_explained: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_feature(self, feature_id: int) -> Optional[FeatureInfo]:
        """Get feature by ID."""
        for feature in self.features:
            if feature.feature_id == feature_id:
                return feature
        return None
    
    def get_top_features(self, n: int = 5) -> List[FeatureInfo]:
        """Get top N features by importance."""
        sorted_features = sorted(self.features, key=lambda f: f.importance, reverse=True)
        return sorted_features[:n]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of feature learning."""
        return {
            'symbol': self.symbol,
            'method': self.method.value,
            'original_dim': self.original_dim,
            'reduced_dim': self.reduced_dim,
            'variance_explained': self.variance_explained,
            'num_features': len(self.features),
            'top_features': [
                {'name': f.name, 'importance': f.importance}
                for f in self.get_top_features(5)
            ],
        }


# ==============================================================================
# FEATURE LEARNING ENGINE
# ==============================================================================

class FeatureLearningEngine:
    """
    Feature learning engine.
    
    Learns and extracts meaningful features from market data.
    """
    
    # Default thresholds
    DEFAULT_PCA_COMPONENTS = 10
    DEFAULT_VARIANCE_THRESHOLD = 0.95
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_MAX_ITER = 1000
    
    def __init__(self, config: Config):
        """
        Initialize the feature learning engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Learning defaults
        self._default_method = LearningMethod.PCA
        self._default_components = getattr(config, 'FEATURE_LEARNING_COMPONENTS', self.DEFAULT_PCA_COMPONENTS)
        self._variance_threshold = getattr(config, 'FEATURE_VARIANCE_THRESHOLD', self.DEFAULT_VARIANCE_THRESHOLD)
        self._random_state = getattr(config, 'RANDOM_STATE', self.DEFAULT_RANDOM_STATE)
        
        self.logger.info(
            f"✅ FeatureLearningEngine initialized: "
            f"method={self._default_method.value}, "
            f"components={self._default_components}"
        )
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def learn_features(
        self,
        data: List[Dict[str, float]],
        symbol: str,
        method: Union[LearningMethod, str] = LearningMethod.PCA,
        n_components: Optional[int] = None,
        variance_threshold: Optional[float] = None,
        **kwargs,
    ) -> FeatureLearningResult:
        """
        Learn features from data.
        
        Args:
            data: List of feature dictionaries
            symbol: Symbol name
            method: Learning method
            n_components: Number of components to learn
            variance_threshold: Variance threshold for PCA
            **kwargs: Additional parameters
            
        Returns:
            FeatureLearningResult object
        """
        if not data:
            raise DataValidationError("No data provided")
        
        method = self._parse_method(method)
        n_components = n_components or self._default_components
        variance_threshold = variance_threshold or self._variance_threshold
        
        self.logger.debug(
            f"Learning features from {len(data)} points using {method.value}"
        )
        
        try:
            # Extract vectors
            vectors, feature_names = self._extract_vectors(data)
            
            if not vectors:
                raise DataValidationError("Empty feature vectors")
            
            original_dim = len(vectors[0])
            
            # Perform feature learning
            if method == LearningMethod.PCA:
                transformed, components, explained_variance = self._pca_learn(
                    vectors, n_components, variance_threshold, **kwargs
                )
            elif method == LearningMethod.RANDOM_PROJECTION:
                transformed, components = self._random_projection_learn(
                    vectors, n_components, **kwargs
                )
                explained_variance = [0.0] * len(components)
            elif method == LearningMethod.ICA:
                transformed, components = self._ica_learn(vectors, n_components, **kwargs)
                explained_variance = [0.0] * len(components)
            else:
                # Default to PCA
                transformed, components, explained_variance = self._pca_learn(
                    vectors, n_components, variance_threshold, **kwargs
                )
            
            # Build feature info
            features = []
            for i, (component, variance) in enumerate(zip(components, explained_variance)):
                importance = abs(variance) if variance > 0 else 0.0
                features.append(FeatureInfo(
                    feature_id=i,
                    name=f"{method.value}_component_{i+1}",
                    variance_explained=variance,
                    importance=importance,
                    coefficients=component,
                ))
            
            # Calculate total variance explained
            total_variance = sum(explained_variance) if explained_variance else 0.0
            
            result = FeatureLearningResult(
                symbol=symbol,
                timestamp=datetime.now(),
                features=features,
                transformed_data=transformed,
                original_dim=original_dim,
                reduced_dim=len(components),
                method=method,
                variance_explained=total_variance,
                metadata={
                    'feature_names': feature_names,
                    'n_components': n_components,
                    'original_data_shape': len(vectors),
                    'method_params': kwargs,
                },
            )
            
            self.logger.debug(
                f"Feature learning complete: {len(components)} components, "
                f"variance explained: {total_variance:.2%}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Feature learning failed: {e}")
            raise DiscoveryError(f"Failed to learn features: {e}")
    
    def reduce_dimensionality(
        self,
        data: List[Dict[str, float]],
        symbol: str,
        method: Union[LearningMethod, str] = LearningMethod.PCA,
        n_components: Optional[int] = None,
    ) -> FeatureLearningResult:
        """
        Reduce dimensionality of data.
        
        Args:
            data: List of feature dictionaries
            symbol: Symbol name
            method: Learning method
            n_components: Number of components
            
        Returns:
            FeatureLearningResult object
        """
        return self.learn_features(data, symbol, method, n_components)
    
    def get_feature_importance(
        self,
        data: List[Dict[str, float]],
        symbol: str,
        method: Union[LearningMethod, str] = LearningMethod.PCA,
    ) -> List[FeatureInfo]:
        """
        Get feature importance scores.
        
        Args:
            data: List of feature dictionaries
            symbol: Symbol name
            method: Learning method
            
        Returns:
            List of FeatureInfo objects sorted by importance
        """
        result = self.learn_features(data, symbol, method)
        return result.get_top_features(len(result.features))
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _parse_method(self, method: Union[LearningMethod, str]) -> LearningMethod:
        """Parse learning method from string or enum."""
        if isinstance(method, LearningMethod):
            return method
        if isinstance(method, str):
            try:
                return LearningMethod(method.lower())
            except ValueError:
                self.logger.warning(f"Unknown method '{method}', using PCA")
                return LearningMethod.PCA
        return self._default_method
    
    def _extract_vectors(
        self,
        data: List[Dict[str, float]]
    ) -> Tuple[List[List[float]], List[str]]:
        """Extract vectors from feature data."""
        if not data:
            return [], []
        
        feature_names = list(data[0].keys())
        
        vectors = []
        for item in data:
            vector = [float(item.get(name, 0.0)) for name in feature_names]
            vectors.append(vector)
        
        # Normalize vectors
        vectors = self._normalize_vectors(vectors)
        
        return vectors, feature_names
    
    def _normalize_vectors(self, vectors: List[List[float]]) -> List[List[float]]:
        """Normalize vectors (z-score)."""
        if not vectors:
            return vectors
        
        n = len(vectors)
        dim = len(vectors[0])
        
        # Calculate mean and std for each dimension
        means = [0.0] * dim
        stds = [0.0] * dim
        
        for d in range(dim):
            values = [v[d] for v in vectors]
            means[d] = sum(values) / n
            variance = sum((v - means[d]) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
            stds[d] = math.sqrt(variance) if variance > 0 else 1.0
        
        # Normalize
        normalized = []
        for vector in vectors:
            norm_vec = [(vector[d] - means[d]) / stds[d] for d in range(dim)]
            normalized.append(norm_vec)
        
        return normalized
    
    def _pca_learn(
        self,
        vectors: List[List[float]],
        n_components: int,
        variance_threshold: float,
        **kwargs,
    ) -> Tuple[List[List[float]], List[List[float]], List[float]]:
        """
        Principal Component Analysis.
        
        Returns:
            Tuple of (transformed_data, components, explained_variance)
        """
        n = len(vectors)
        dim = len(vectors[0]) if vectors else 0
        
        if n < 2 or dim == 0:
            return [], [], []
        
        # Center the data
        means = [sum(v[d] for v in vectors) / n for d in range(dim)]
        centered = [[v[d] - means[d] for d in range(dim)] for v in vectors]
        
        # Compute covariance matrix
        cov = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                cov[i][j] = sum(centered[k][i] * centered[k][j] for k in range(n)) / (n - 1) if n > 1 else 0.0
        
        # Power iteration for eigenvectors
        components = []
        explained_variance = []
        
        # Calculate total variance
        total_variance = sum(cov[i][i] for i in range(dim))
        if total_variance == 0:
            total_variance = 1.0
        
        # Use power iteration to find top eigenvectors
        remaining_cov = [row[:] for row in cov]
        accumulated_variance = 0.0
        
        for _ in range(min(n_components, dim)):
            if accumulated_variance >= variance_threshold:
                break
            
            # Power iteration
            eigenvector = self._power_iteration(remaining_cov, dim)
            
            # Normalize
            norm = math.sqrt(sum(e ** 2 for e in eigenvector))
            if norm > 0:
                eigenvector = [e / norm for e in eigenvector]
            
            # Calculate eigenvalue (variance explained)
            eigenvalue = self._rayleigh_quotient(remaining_cov, eigenvector)
            
            if eigenvalue < 1e-10:
                break
            
            components.append(eigenvector)
            explained_variance.append(eigenvalue / total_variance)
            accumulated_variance += eigenvalue / total_variance
            
            # Deflate covariance matrix
            for i in range(dim):
                for j in range(dim):
                    remaining_cov[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j]
        
        # Project data onto components
        transformed = []
        for vector in centered:
            projected = [sum(vector[d] * components[c][d] for d in range(dim)) for c in range(len(components))]
            transformed.append(projected)
        
        return transformed, components, explained_variance
    
    def _random_projection_learn(
        self,
        vectors: List[List[float]],
        n_components: int,
        **kwargs,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Random projection for dimensionality reduction.
        
        Returns:
            Tuple of (transformed_data, components)
        """
        dim = len(vectors[0]) if vectors else 0
        if dim == 0:
            return [], []
        
        # Generate random projection matrix (Gaussian)
        random.seed(self._random_state)
        components = []
        for _ in range(min(n_components, dim)):
            component = [random.gauss(0, 1) for _ in range(dim)]
            norm = math.sqrt(sum(e ** 2 for e in component))
            if norm > 0:
                component = [e / norm for e in component]
            components.append(component)
        
        # Project data
        transformed = []
        for vector in vectors:
            projected = [sum(vector[d] * components[c][d] for d in range(dim)) for c in range(len(components))]
            transformed.append(projected)
        
        return transformed, components
    
    def _ica_learn(
        self,
        vectors: List[List[float]],
        n_components: int,
        **kwargs,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Independent Component Analysis (simplified).
        
        Returns:
            Tuple of (transformed_data, components)
        """
        # This is a simplified ICA using FastICA-like approach
        # For production, use scikit-learn's FastICA
        
        dim = len(vectors[0]) if vectors else 0
        if dim == 0:
            return [], []
        
        # First, whiten the data
        n = len(vectors)
        means = [sum(v[d] for v in vectors) / n for d in range(dim)]
        centered = [[v[d] - means[d] for d in range(dim)] for v in vectors]
        
        # Compute covariance
        cov = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                cov[i][j] = sum(centered[k][i] * centered[k][j] for k in range(n)) / n
        
        # SVD for whitening
        u, s, v = self._svd(cov)
        
        # Whitening matrix
        s_inv_sqrt = [1.0 / math.sqrt(max(ss, 1e-10)) for ss in s]
        whitening = [[u[i][j] * s_inv_sqrt[j] for j in range(dim)] for i in range(dim)]
        
        # Whiten data
        whitened = []
        for vector in centered:
            w_vec = [sum(vector[d] * whitening[d][i] for d in range(dim)) for i in range(dim)]
            whitened.append(w_vec)
        
        # Approximate ICA using fixed-point iteration
        n_components = min(n_components, dim)
        components = []
        
        for _ in range(n_components):
            # Initialize random vector
            w = [random.gauss(0, 1) for _ in range(dim)]
            
            # Fixed-point iteration
            for _ in range(100):
                # Compute G(w^T x) and G'(w^T x)
                g = []
                g_prime = []
                for x in whitened:
                    wx = sum(w[i] * x[i] for i in range(dim))
                    g.append(math.tanh(wx))
                    g_prime.append(1 - math.tanh(wx) ** 2)
                
                # Update w
                new_w = [0.0] * dim
                for i in range(dim):
                    new_w[i] = sum(g[j] * whitened[j][i] for j in range(len(whitened))) / len(whitened)
                    new_w[i] -= sum(g_prime) / len(whitened) * w[i]
                
                # Normalize
                norm = math.sqrt(sum(e ** 2 for e in new_w))
                if norm > 0:
                    new_w = [e / norm for e in new_w]
                
                # Check convergence
                diff = math.sqrt(sum((new_w[i] - w[i]) ** 2 for i in range(dim)))
                w = new_w
                if diff < 1e-6:
                    break
            
            # Orthogonalize against previous components
            for comp in components:
                dot = sum(w[i] * comp[i] for i in range(dim))
                w = [w[i] - dot * comp[i] for i in range(dim)]
                norm = math.sqrt(sum(e ** 2 for e in w))
                if norm > 0:
                    w = [e / norm for e in w]
            
            components.append(w)
        
        # Project data
        transformed = []
        for vector in whitened:
            projected = [sum(vector[d] * components[c][d] for d in range(dim)) for c in range(len(components))]
            transformed.append(projected)
        
        return transformed, components
    
    def _power_iteration(self, matrix: List[List[float]], dim: int, max_iter: int = 1000) -> List[float]:
        """Power iteration for largest eigenvector."""
        # Initialize random vector
        vector = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(e ** 2 for e in vector))
        if norm > 0:
            vector = [e / norm for e in vector]
        
        for _ in range(max_iter):
            # Multiply matrix by vector
            new_vector = [0.0] * dim
            for i in range(dim):
                new_vector[i] = sum(matrix[i][j] * vector[j] for j in range(dim))
            
            # Normalize
            norm = math.sqrt(sum(e ** 2 for e in new_vector))
            if norm < 1e-10:
                break
            new_vector = [e / norm for e in new_vector]
            
            # Check convergence
            diff = math.sqrt(sum((new_vector[i] - vector[i]) ** 2 for i in range(dim)))
            vector = new_vector
            if diff < 1e-8:
                break
        
        return vector
    
    def _rayleigh_quotient(self, matrix: List[List[float]], vector: List[float]) -> float:
        """Compute Rayleigh quotient for matrix and vector."""
        dim = len(matrix)
        numerator = sum(matrix[i][j] * vector[i] * vector[j] for i in range(dim) for j in range(dim))
        denominator = sum(v ** 2 for v in vector)
        return numerator / denominator if denominator > 0 else 0.0
    
    def _svd(self, matrix: List[List[float]]) -> Tuple[List[List[float]], List[float], List[List[float]]]:
        """Simplified SVD using power iteration."""
        m = len(matrix)
        n = len(matrix[0]) if matrix else 0
        
        if m == 0 or n == 0:
            return [], [], []
        
        # Compute A^T A
        ata = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                ata[i][j] = sum(matrix[k][i] * matrix[k][j] for k in range(m))
        
        # Compute A A^T
        aat = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                aat[i][j] = sum(matrix[i][k] * matrix[j][k] for k in range(n))
        
        # Find eigenvalues of A^T A (singular values squared)
        singular_values = []
        eigenvectors = []
        
        # Use power iteration for largest eigenvalues
        remaining = [row[:] for row in ata]
        for _ in range(min(m, n)):
            eigenvector = self._power_iteration(remaining, n)
            norm = math.sqrt(sum(e ** 2 for e in eigenvector))
            if norm > 0:
                eigenvector = [e / norm for e in eigenvector]
            
            eigenvalue = self._rayleigh_quotient(remaining, eigenvector)
            if eigenvalue < 1e-10:
                break
            
            singular_values.append(math.sqrt(eigenvalue))
            eigenvectors.append(eigenvector)
            
            # Deflate
            for i in range(n):
                for j in range(n):
                    remaining[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j]
        
        # Compute U from V and singular values
        u = []
        for i in range(m):
            row = []
            for j in range(len(singular_values)):
                val = 0.0
                for k in range(n):
                    val += matrix[i][k] * eigenvectors[j][k]
                if singular_values[j] > 0:
                    val /= singular_values[j]
                row.append(val)
            u.append(row)
        
        return u, singular_values, eigenvectors


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_feature_learning_engine(config: Config) -> FeatureLearningEngine:
    """
    Factory function for FeatureLearningEngine creation.
    
    Args:
        config: Application configuration
        
    Returns:
        FeatureLearningEngine instance
    """
    return FeatureLearningEngine(config)