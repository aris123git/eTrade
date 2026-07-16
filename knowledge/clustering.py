"""
discovery/clustering.py - Market Clustering Engine

RESPONSIBILITY:
Discover and analyze clusters in market data.

ARCHITECTURAL PRINCIPLES:
1. Pure clustering - No data storage, no I/O, no business logic
2. Statistical clustering of market patterns and behaviors
3. Type-safe results with validation
4. Multiple clustering algorithms (K-means, DBSCAN, hierarchical)

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
import random

from core.config import Config
from core.exceptions import DiscoveryError, DataValidationError


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'ClusterMethod',
    'DistanceMetric',
    'Cluster',
    'ClusterResult',
    'ClusteringEngine',
    'create_clustering_engine',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class ClusterMethod(Enum):
    """Method for clustering."""
    KMEANS = "kmeans"           # K-means clustering
    DBSCAN = "dbscan"           # Density-based clustering
    HIERARCHICAL = "hierarchical"  # Hierarchical clustering
    AGGREGATIVE = "aggregative"    # Agglomerative clustering
    SPECTRAL = "spectral"       # Spectral clustering


class DistanceMetric(Enum):
    """Metric for distance calculation."""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    CORRELATION = "correlation"
    CHEBYSHEV = "chebyshev"


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class Cluster:
    """A discovered cluster."""
    cluster_id: int
    centroid: List[float]
    points: List[Dict[str, Any]]
    size: int
    diameter: float
    intra_cluster_distance: float
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cluster_id': self.cluster_id,
            'centroid': self.centroid,
            'size': self.size,
            'diameter': self.diameter,
            'intra_cluster_distance': self.intra_cluster_distance,
            'labels': self.labels,
            'metadata': self.metadata,
        }


@dataclass
class ClusterResult:
    """Result of clustering operation."""
    symbol: str
    timestamp: datetime
    clusters: List[Cluster]
    num_clusters: int
    method: ClusterMethod
    distance_metric: DistanceMetric
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_cluster(self, cluster_id: int) -> Optional[Cluster]:
        """Get cluster by ID."""
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None
    
    def get_largest_cluster(self) -> Optional[Cluster]:
        """Get the largest cluster."""
        if not self.clusters:
            return None
        return max(self.clusters, key=lambda c: c.size)
    
    def get_smallest_cluster(self) -> Optional[Cluster]:
        """Get the smallest cluster."""
        if not self.clusters:
            return None
        return min(self.clusters, key=lambda c: c.size)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of clustering."""
        return {
            'symbol': self.symbol,
            'num_clusters': self.num_clusters,
            'total_points': sum(c.size for c in self.clusters),
            'method': self.method.value,
            'distance_metric': self.distance_metric.value,
            'avg_cluster_size': sum(c.size for c in self.clusters) / len(self.clusters) if self.clusters else 0,
            'cluster_sizes': [c.size for c in self.clusters],
            'cluster_diameters': [c.diameter for c in self.clusters],
        }


# ==============================================================================
# CLUSTERING ENGINE
# ==============================================================================

class ClusteringEngine:
    """
    Market clustering engine.
    
    Discovers and analyzes clusters in market data.
    """
    
    # Default thresholds
    DEFAULT_KMEANS_MAX_ITER = 100
    DEFAULT_KMEANS_INIT = 'kmeans++'
    DEFAULT_DBSCAN_EPS = 0.5
    DEFAULT_DBSCAN_MIN_SAMPLES = 5
    DEFAULT_HIERARCHICAL_METHOD = 'ward'
    DEFAULT_MAX_CLUSTERS = 10
    DEFAULT_MIN_CLUSTERS = 2
    
    def __init__(self, config: Config):
        """
        Initialize the clustering engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Clustering defaults
        self._default_method = ClusterMethod.KMEANS
        self._default_metric = DistanceMetric.EUCLIDEAN
        self._max_clusters = getattr(config, 'CLUSTER_MAX_CLUSTERS', self.DEFAULT_MAX_CLUSTERS)
        self._min_clusters = getattr(config, 'CLUSTER_MIN_CLUSTERS', self.DEFAULT_MIN_CLUSTERS)
        
        self.logger.info(
            f"✅ ClusteringEngine initialized: "
            f"max_clusters={self._max_clusters}, "
            f"min_clusters={self._min_clusters}"
        )
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def cluster_markets(
        self,
        feature_data: List[Dict[str, float]],
        symbol: str,
        method: Union[ClusterMethod, str] = ClusterMethod.KMEANS,
        metric: Union[DistanceMetric, str] = DistanceMetric.EUCLIDEAN,
        num_clusters: Optional[int] = None,
        **kwargs,
    ) -> ClusterResult:
        """
        Cluster markets based on feature data.
        
        Args:
            feature_data: List of feature dictionaries
            symbol: Symbol name
            method: Clustering method
            metric: Distance metric
            num_clusters: Number of clusters (for K-means)
            **kwargs: Additional parameters for specific methods
            
        Returns:
            ClusterResult object
        """
        if not feature_data:
            raise DataValidationError("No feature data provided")
        
        method = self._parse_method(method)
        metric = self._parse_metric(metric)
        
        self.logger.debug(
            f"Clustering {len(feature_data)} points using {method.value}"
        )
        
        try:
            # Convert features to vector format
            vectors, feature_names = self._extract_vectors(feature_data)
            
            if len(vectors) < self._min_clusters:
                raise DataValidationError(
                    f"Insufficient points for clustering: {len(vectors)} < {self._min_clusters}"
                )
            
            # Determine number of clusters
            if num_clusters is None:
                if method in (ClusterMethod.KMEANS, ClusterMethod.AGGREGATIVE):
                    num_clusters = self._determine_optimal_clusters(vectors, method, metric, **kwargs)
                else:
                    num_clusters = self._min_clusters
            
            # Perform clustering
            clusters = self._cluster(
                vectors, feature_data, method, metric, num_clusters, **kwargs
            )
            
            result = ClusterResult(
                symbol=symbol,
                timestamp=datetime.now(),
                clusters=clusters,
                num_clusters=len(clusters),
                method=method,
                distance_metric=metric,
                metadata={
                    'feature_names': feature_names,
                    'num_points': len(vectors),
                    'method_params': kwargs,
                },
            )
            
            self.logger.debug(
                f"Clustering complete: {len(clusters)} clusters found"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            raise DiscoveryError(f"Failed to cluster markets: {e}")
    
    def cluster_patterns(
        self,
        patterns: List[Dict[str, Any]],
        symbol: str,
        method: Union[ClusterMethod, str] = ClusterMethod.KMEANS,
        metric: Union[DistanceMetric, str] = DistanceMetric.EUCLIDEAN,
        num_clusters: Optional[int] = None,
    ) -> ClusterResult:
        """
        Cluster patterns based on their features.
        
        Args:
            patterns: List of pattern dictionaries
            symbol: Symbol name
            method: Clustering method
            metric: Distance metric
            num_clusters: Number of clusters
            
        Returns:
            ClusterResult object
        """
        # Extract pattern features
        feature_data = []
        for pattern in patterns:
            features = {
                'strength': pattern.get('strength', 0.5),
                'confidence': pattern.get('confidence', 0.5),
                'duration': pattern.get('candle_count', 1),
                'range': pattern.get('price_range', (0, 1))[1] - pattern.get('price_range', (0, 1))[0],
            }
            feature_data.append(features)
        
        return self.cluster_markets(feature_data, symbol, method, metric, num_clusters)
    
    def find_optimal_clusters(
        self,
        feature_data: List[Dict[str, float]],
        method: Union[ClusterMethod, str] = ClusterMethod.KMEANS,
        metric: Union[DistanceMetric, str] = DistanceMetric.EUCLIDEAN,
        max_clusters: Optional[int] = None,
    ) -> int:
        """
        Find optimal number of clusters using elbow method.
        
        Args:
            feature_data: List of feature dictionaries
            method: Clustering method
            metric: Distance metric
            max_clusters: Maximum number of clusters
            
        Returns:
            Optimal number of clusters
        """
        method = self._parse_method(method)
        metric = self._parse_metric(metric)
        max_clusters = max_clusters or self._max_clusters
        
        vectors, _ = self._extract_vectors(feature_data)
        
        if len(vectors) < max_clusters:
            max_clusters = len(vectors) - 1
        
        if max_clusters < self._min_clusters:
            return self._min_clusters
        
        return self._determine_optimal_clusters(vectors, method, metric, max_clusters=max_clusters)
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _parse_method(self, method: Union[ClusterMethod, str]) -> ClusterMethod:
        """Parse clustering method from string or enum."""
        if isinstance(method, ClusterMethod):
            return method
        if isinstance(method, str):
            try:
                return ClusterMethod(method.lower())
            except ValueError:
                self.logger.warning(f"Unknown method '{method}', using KMEANS")
                return ClusterMethod.KMEANS
        return self._default_method
    
    def _parse_metric(self, metric: Union[DistanceMetric, str]) -> DistanceMetric:
        """Parse distance metric from string or enum."""
        if isinstance(metric, DistanceMetric):
            return metric
        if isinstance(metric, str):
            try:
                return DistanceMetric(metric.lower())
            except ValueError:
                self.logger.warning(f"Unknown metric '{metric}', using EUCLIDEAN")
                return DistanceMetric.EUCLIDEAN
        return self._default_metric
    
    def _extract_vectors(
        self,
        feature_data: List[Dict[str, float]]
    ) -> Tuple[List[List[float]], List[str]]:
        """Extract vectors from feature data."""
        if not feature_data:
            return [], []
        
        # Get feature names from first item
        feature_names = list(feature_data[0].keys())
        
        vectors = []
        for item in feature_data:
            vector = [float(item.get(name, 0.0)) for name in feature_names]
            vectors.append(vector)
        
        return vectors, feature_names
    
    def _cluster(
        self,
        vectors: List[List[float]],
        feature_data: List[Dict[str, float]],
        method: ClusterMethod,
        metric: DistanceMetric,
        num_clusters: int,
        **kwargs,
    ) -> List[Cluster]:
        """Perform clustering using specified method."""
        if method == ClusterMethod.KMEANS:
            return self._kmeans_clustering(vectors, feature_data, num_clusters, metric, **kwargs)
        elif method == ClusterMethod.DBSCAN:
            return self._dbscan_clustering(vectors, feature_data, metric, **kwargs)
        elif method == ClusterMethod.HIERARCHICAL:
            return self._hierarchical_clustering(vectors, feature_data, metric, num_clusters, **kwargs)
        elif method == ClusterMethod.AGGREGATIVE:
            return self._aggregative_clustering(vectors, feature_data, metric, num_clusters, **kwargs)
        else:
            return self._kmeans_clustering(vectors, feature_data, num_clusters, metric, **kwargs)
    
    def _kmeans_clustering(
        self,
        vectors: List[List[float]],
        feature_data: List[Dict[str, float]],
        num_clusters: int,
        metric: DistanceMetric,
        **kwargs,
    ) -> List[Cluster]:
        """K-means clustering."""
        max_iter = kwargs.get('max_iter', self.DEFAULT_KMEANS_MAX_ITER)
        init_method = kwargs.get('init', self.DEFAULT_KMEANS_INIT)
        
        # Initialize centroids using k-means++
        centroids = self._kmeans_plus_plus_init(vectors, num_clusters, metric)
        
        # Iterate
        for iteration in range(max_iter):
            # Assign points to nearest centroid
            clusters = [[] for _ in range(num_clusters)]
            for i, vector in enumerate(vectors):
                distances = [self._distance(vector, c, metric) for c in centroids]
                nearest = min(range(len(distances)), key=lambda j: distances[j])
                clusters[nearest].append(i)
            
            # Update centroids
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    cluster_vectors = [vectors[i] for i in cluster]
                    centroid = self._compute_centroid(cluster_vectors)
                    new_centroids.append(centroid)
                else:
                    new_centroids.append(centroids[len(new_centroids)])
            
            # Check convergence
            if self._centroids_converged(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        # Build cluster objects
        return self._build_clusters(vectors, feature_data, centroids, clusters, metric)
    
    def _dbscan_clustering(
        self,
        vectors: List[List[float]],
        feature_data: List[Dict[str, float]],
        metric: DistanceMetric,
        **kwargs,
    ) -> List[Cluster]:
        """DBSCAN clustering."""
        eps = kwargs.get('eps', self.DEFAULT_DBSCAN_EPS)
        min_samples = kwargs.get('min_samples', self.DEFAULT_DBSCAN_MIN_SAMPLES)
        
        n = len(vectors)
        visited = [False] * n
        cluster_labels = [-1] * n  # -1 = noise
        cluster_id = 0
        
        for i in range(n):
            if visited[i]:
                continue
            
            visited[i] = True
            
            # Find neighbors
            neighbors = []
            for j in range(n):
                if i != j and self._distance(vectors[i], vectors[j], metric) <= eps:
                    neighbors.append(j)
            
            if len(neighbors) < min_samples:
                cluster_labels[i] = -1  # Noise
            else:
                # Expand cluster
                cluster_labels[i] = cluster_id
                for j in neighbors:
                    if not visited[j]:
                        visited[j] = True
                        # Find neighbors of neighbor
                        sub_neighbors = []
                        for k in range(n):
                            if j != k and self._distance(vectors[j], vectors[k], metric) <= eps:
                                sub_neighbors.append(k)
                        
                        if len(sub_neighbors) >= min_samples:
                            neighbors.extend(sub_neighbors)
                    
                    if cluster_labels[j] == -1:
                        cluster_labels[j] = cluster_id
                
                cluster_id += 1
        
        # Build clusters from labels
        clusters_dict = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label >= 0:
                clusters_dict[label].append(i)
        
        # Build cluster objects
        clusters = []
        for cid, indices in clusters_dict.items():
            cluster_vectors = [vectors[i] for i in indices]
            centroid = self._compute_centroid(cluster_vectors)
            
            # Calculate intra-cluster distance
            intra_dist = sum(
                self._distance(centroid, vectors[i], metric) for i in indices
            ) / len(indices) if indices else 0.0
            
            # Calculate diameter
            diameter = 0.0
            for i in indices:
                for j in indices:
                    if i != j:
                        dist = self._distance(vectors[i], vectors[j], metric)
                        if dist > diameter:
                            diameter = dist
            
            clusters.append(Cluster(
                cluster_id=cid,
                centroid=centroid,
                points=[feature_data[i] for i in indices],
                size=len(indices),
                diameter=diameter,
                intra_cluster_distance=intra_dist,
                labels=[f"cluster_{cid}" for _ in indices],
            ))
        
        return clusters
    
    def _hierarchical_clustering(
        self,
        vectors: List[List[float]],
        feature_data: List[Dict[str, float]],
        metric: DistanceMetric,
        num_clusters: int,
        **kwargs,
    ) -> List[Cluster]:
        """Hierarchical clustering (agglomerative)."""
        method = kwargs.get('method', self.DEFAULT_HIERARCHICAL_METHOD)
        
        n = len(vectors)
        if n <= num_clusters:
            return self._build_clusters_from_indices(
                vectors, feature_data, [[i] for i in range(n)], metric
            )
        
        # Initialize clusters (each point is its own cluster)
        clusters_indices = [[i] for i in range(n)]
        cluster_distances = self._compute_distance_matrix(vectors, clusters_indices, metric)
        
        while len(clusters_indices) > num_clusters:
            # Find closest clusters
            min_dist = float('inf')
            min_i, min_j = 0, 1
            
            for i in range(len(clusters_indices)):
                for j in range(i + 1, len(clusters_indices)):
                    dist = cluster_distances.get((i, j), float('inf'))
                    if dist < min_dist:
                        min_dist = dist
                        min_i, min_j = i, j
            
            # Merge clusters
            merged = clusters_indices[min_i] + clusters_indices[min_j]
            clusters_indices[min_i] = merged
            clusters_indices.pop(min_j)
            
            # Update distance matrix
            cluster_distances = self._compute_distance_matrix(
                vectors, clusters_indices, metric
            )
        
        # Build clusters
        return self._build_clusters_from_indices(
            vectors, feature_data, clusters_indices, metric
        )
    
    def _aggregative_clustering(
        self,
        vectors: List[List[float]],
        feature_data: List[Dict[str, float]],
        metric: DistanceMetric,
        num_clusters: int,
        **kwargs,
    ) -> List[Cluster]:
        """Agglomerative clustering."""
        return self._hierarchical_clustering(vectors, feature_data, metric, num_clusters, **kwargs)
    
    def _kmeans_plus_plus_init(
        self,
        vectors: List[List[float]],
        num_clusters: int,
        metric: DistanceMetric,
    ) -> List[List[float]]:
        """K-means++ initialization."""
        n = len(vectors)
        if n == 0:
            return []
        
        # Choose first centroid randomly
        centroids = [vectors[random.randint(0, n - 1)].copy()]
        
        for _ in range(1, num_clusters):
            # Compute distances to nearest centroid
            distances = []
            for vector in vectors:
                min_dist = min(self._distance(vector, c, metric) for c in centroids)
                distances.append(min_dist ** 2)
            
            total = sum(distances)
            if total == 0:
                break
            
            # Choose next centroid with probability proportional to distance
            r = random.random() * total
            cumulative = 0.0
            for i, dist in enumerate(distances):
                cumulative += dist
                if cumulative >= r:
                    centroids.append(vectors[i].copy())
                    break
        
        return centroids
    
    def _compute_centroid(self, vectors: List[List[float]]) -> List[float]:
        """Compute centroid of vectors."""
        if not vectors:
            return []
        
        n = len(vectors)
        dim = len(vectors[0])
        centroid = [0.0] * dim
        
        for vector in vectors:
            for i in range(dim):
                centroid[i] += vector[i]
        
        for i in range(dim):
            centroid[i] /= n
        
        return centroid
    
    def _distance(self, v1: List[float], v2: List[float], metric: DistanceMetric) -> float:
        """Calculate distance between two vectors."""
        if len(v1) != len(v2):
            return float('inf')
        
        if metric == DistanceMetric.EUCLIDEAN:
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
        elif metric == DistanceMetric.MANHATTAN:
            return sum(abs(a - b) for a, b in zip(v1, v2))
        elif metric == DistanceMetric.COSINE:
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            if norm1 == 0 or norm2 == 0:
                return 1.0
            return 1.0 - dot / (norm1 * norm2)
        elif metric == DistanceMetric.CORRELATION:
            return 1.0 - self._pearson_correlation(v1, v2)
        elif metric == DistanceMetric.CHEBYSHEV:
            return max(abs(a - b) for a, b in zip(v1, v2))
        else:
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
    
    def _pearson_correlation(self, v1: List[float], v2: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(v1)
        if n < 2:
            return 0.0
        
        mean1 = sum(v1) / n
        mean2 = sum(v2) / n
        
        cov = 0.0
        var1 = 0.0
        var2 = 0.0
        
        for i in range(n):
            d1 = v1[i] - mean1
            d2 = v2[i] - mean2
            cov += d1 * d2
            var1 += d1 * d1
            var2 += d2 * d2
        
        if var1 == 0 or var2 == 0:
            return 0.0
        
        return cov / (math.sqrt(var1) * math.sqrt(var2))
    
    def _compute_distance_matrix(
        self,
        vectors: List[List[float]],
        clusters: List[List[int]],
        metric: DistanceMetric,
    ) -> Dict[Tuple[int, int], float]:
        """Compute distance matrix between clusters."""
        distances = {}
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Use complete linkage (max distance)
                max_dist = 0.0
                for idx_i in clusters[i]:
                    for idx_j in clusters[j]:
                        dist = self._distance(vectors[idx_i], vectors[idx_j], metric)
                        if dist > max_dist:
                            max_dist = dist
                distances[(i, j)] = max_dist
        return distances
    
    def _centroids_converged(
        self,
        old_centroids: List[List[float]],
        new_centroids: List[List[float]],
        eps: float = 1e-6,
    ) -> bool:
        """Check if centroids have converged."""
        if len(old_centroids) != len(new_centroids):
            return False
        
        for i in range(len(old_centroids)):
            dist = self._distance(old_centroids[i], new_centroids[i], DistanceMetric.EUCLIDEAN)
            if dist > eps:
                return False
        
        return True
    
    def _build_clusters(
        self,
        vectors: List[List[float]],
        feature_data: List[Dict[str, float]],
        centroids: List[List[float]],
        clusters: List[List[int]],
        metric: DistanceMetric,
    ) -> List[Cluster]:
        """Build cluster objects from clustering results."""
        cluster_objects = []
        
        for cid, indices in enumerate(clusters):
            if not indices:
                continue
            
            cluster_vectors = [vectors[i] for i in indices]
            
            # Calculate intra-cluster distance
            centroid = centroids[cid]
            intra_dist = sum(
                self._distance(centroid, vectors[i], metric) for i in indices
            ) / len(indices) if indices else 0.0
            
            # Calculate diameter
            diameter = 0.0
            for i in indices:
                for j in indices:
                    if i != j:
                        dist = self._distance(vectors[i], vectors[j], metric)
                        if dist > diameter:
                            diameter = dist
            
            cluster_objects.append(Cluster(
                cluster_id=cid,
                centroid=centroid,
                points=[feature_data[i] for i in indices],
                size=len(indices),
                diameter=diameter,
                intra_cluster_distance=intra_dist,
                labels=[f"cluster_{cid}" for _ in indices],
            ))
        
        return cluster_objects
    
    def _build_clusters_from_indices(
        self,
        vectors: List[List[float]],
        feature_data: List[Dict[str, float]],
        cluster_indices: List[List[int]],
        metric: DistanceMetric,
    ) -> List[Cluster]:
        """Build clusters from indices."""
        clusters = []
        
        for cid, indices in enumerate(cluster_indices):
            cluster_vectors = [vectors[i] for i in indices]
            centroid = self._compute_centroid(cluster_vectors)
            
            # Calculate intra-cluster distance
            intra_dist = sum(
                self._distance(centroid, vectors[i], metric) for i in indices
            ) / len(indices) if indices else 0.0
            
            # Calculate diameter
            diameter = 0.0
            for i in indices:
                for j in indices:
                    if i != j:
                        dist = self._distance(vectors[i], vectors[j], metric)
                        if dist > diameter:
                            diameter = dist
            
            clusters.append(Cluster(
                cluster_id=cid,
                centroid=centroid,
                points=[feature_data[i] for i in indices],
                size=len(indices),
                diameter=diameter,
                intra_cluster_distance=intra_dist,
                labels=[f"cluster_{cid}" for _ in indices],
            ))
        
        return clusters
    
    def _determine_optimal_clusters(
        self,
        vectors: List[List[float]],
        method: ClusterMethod,
        metric: DistanceMetric,
        max_clusters: Optional[int] = None,
        **kwargs,
    ) -> int:
        """Determine optimal number of clusters using elbow method."""
        max_clusters = max_clusters or self._max_clusters
        n = len(vectors)
        
        if n <= self._min_clusters:
            return self._min_clusters
        
        max_clusters = min(max_clusters, n - 1)
        
        # Calculate inertia for different cluster counts
        inertias = []
        for k in range(self._min_clusters, min(max_clusters, n) + 1):
            clusters = self._kmeans_clustering(
                vectors, [{} for _ in range(n)], k, metric, max_iter=10, **kwargs
            )
            inertia = sum(
                self._distance(vectors[i], cluster.centroid, metric) ** 2
                for cluster in clusters
                for i in range(len(cluster.points))
            )
            inertias.append(inertia)
        
        if not inertias:
            return self._min_clusters
        
        # Find elbow using rate of change
        if len(inertias) < 3:
            return self._min_clusters
        
        # Calculate rate of change
        changes = []
        for i in range(1, len(inertias)):
            if inertias[i-1] > 0:
                change = (inertias[i-1] - inertias[i]) / inertias[i-1]
                changes.append(change)
            else:
                changes.append(0.0)
        
        # Find point where rate of change drops significantly
        if changes:
            avg_change = sum(changes) / len(changes)
            threshold = avg_change * 0.3
            
            for i, change in enumerate(changes):
                if change < threshold and i > 0:
                    return self._min_clusters + i
        
        return self._min_clusters


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_clustering_engine(config: Config) -> ClusteringEngine:
    """
    Factory function for ClusteringEngine creation.
    
    Args:
        config: Application configuration
        
    Returns:
        ClusteringEngine instance
    """
    return ClusteringEngine(config)