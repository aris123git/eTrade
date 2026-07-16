"""
knowledge/graph.py - Knowledge Graph Module

RESPONSIBILITY:
Build and manage a knowledge graph of market relationships and patterns.

ARCHITECTURAL PRINCIPLES:
1. Pure knowledge representation - No data storage, no I/O, no business logic
2. Graph-based relationships between market entities
3. Type-safe results with validation
4. Multiple relationship types (correlation, causation, pattern, etc.)

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions
- ❌ Analyze patterns (only represents relationships)

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
from core.exceptions import DiscoveryError, DataValidationError


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'NodeType',
    'EdgeType',
    'RelationshipStrength',
    'GraphNode',
    'GraphEdge',
    'KnowledgeGraph',
    'create_knowledge_graph',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    MARKET = "market"
    PATTERN = "pattern"
    INDICATOR = "indicator"
    EVENT = "event"
    SESSION = "session"
    CORRELATION = "correlation"
    HYPOTHESIS = "hypothesis"
    STRATEGY = "strategy"
    METRIC = "metric"
    UNKNOWN = "unknown"


class EdgeType(Enum):
    """Types of edges in the knowledge graph."""
    CORRELATES_WITH = "correlates_with"
    CAUSES = "causes"
    LEADS_TO = "leads_to"
    FOLLOWS = "follows"
    CONTAINS = "contains"
    PART_OF = "part_of"
    PRECEDES = "precedes"
    SUCCEEDS = "succeeds"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    DERIVED_FROM = "derived_from"
    INFLUENCES = "influences"
    CONFLICTS_WITH = "conflicts_with"
    UNKNOWN = "unknown"


class RelationshipStrength(Enum):
    """Strength of a relationship in the knowledge graph."""
    VERY_STRONG = 1.0
    STRONG = 0.8
    MODERATE = 0.6
    WEAK = 0.4
    VERY_WEAK = 0.2
    NONE = 0.0


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    id: str
    node_type: NodeType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            'id': self.id,
            'type': self.node_type.value,
            'label': self.label,
            'properties': self.properties,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    strength: RelationshipStrength
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary."""
        return {
            'source': self.source_id,
            'target': self.target_id,
            'type': self.edge_type.value,
            'strength': self.strength.value,
            'weight': self.weight,
            'properties': self.properties,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }


# ==============================================================================
# KNOWLEDGE GRAPH
# ==============================================================================

class KnowledgeGraph:
    """
    Knowledge graph for market relationships and patterns.
    
    Builds and manages a graph of market entities and their relationships.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the knowledge graph.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Graph storage
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)
        
        # Indexes
        self._nodes_by_type: Dict[NodeType, Set[str]] = defaultdict(set)
        self._nodes_by_label: Dict[str, Set[str]] = defaultdict(set)
        self._edges_by_type: Dict[EdgeType, Set[str]] = defaultdict(set)
        
        self._node_counter = 0
        self._edge_counter = 0
        
        self.logger.info("✅ KnowledgeGraph initialized")
    
    # ==========================================================================
    # NODE OPERATIONS
    # ==========================================================================
    
    def add_node(
        self,
        node_type: NodeType,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphNode:
        """
        Add a node to the graph.
        
        Args:
            node_type: Type of node
            label: Display label
            properties: Node properties
            node_id: Optional node ID (auto-generated if not provided)
            metadata: Additional metadata
            
        Returns:
            Created GraphNode
        """
        if node_id is None:
            node_id = self._generate_node_id()
        
        if node_id in self._nodes:
            self.logger.warning(f"Node {node_id} already exists, updating")
            node = self._nodes[node_id]
            node.node_type = node_type
            node.label = label
            node.properties = properties or {}
            node.metadata = metadata or {}
            node.updated_at = datetime.now()
            return node
        
        node = GraphNode(
            id=node_id,
            node_type=node_type,
            label=label,
            properties=properties or {},
            metadata=metadata or {},
        )
        
        self._nodes[node_id] = node
        self._nodes_by_type[node_type].add(node_id)
        self._nodes_by_label[label].add(node_id)
        
        self._node_counter += 1
        
        self.logger.debug(f"Added node: {node_id} ({node_type.value}: {label})")
        return node
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        return [self._nodes[nid] for nid in self._nodes_by_type.get(node_type, set()) if nid in self._nodes]
    
    def get_nodes_by_label(self, label: str) -> List[GraphNode]:
        """Get all nodes with a specific label."""
        return [self._nodes[nid] for nid in self._nodes_by_label.get(label, set()) if nid in self._nodes]
    
    def update_node(
        self,
        node_id: str,
        properties: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[GraphNode]:
        """Update a node's properties."""
        node = self._nodes.get(node_id)
        if not node:
            return None
        
        if label:
            # Update label index
            old_label = node.label
            if old_label in self._nodes_by_label:
                self._nodes_by_label[old_label].discard(node_id)
            node.label = label
            self._nodes_by_label[label].add(node_id)
        
        if properties:
            node.properties.update(properties)
        
        if metadata:
            node.metadata.update(metadata)
        
        node.updated_at = datetime.now()
        return node
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its edges."""
        if node_id not in self._nodes:
            return False
        
        # Remove all edges involving this node
        for edge_id in list(self._adjacency.get(node_id, set())):
            self.delete_edge(edge_id)
        for edge_id in list(self._reverse_adjacency.get(node_id, set())):
            self.delete_edge(edge_id)
        
        # Remove from indexes
        node = self._nodes[node_id]
        self._nodes_by_type[node.node_type].discard(node_id)
        self._nodes_by_label[node.label].discard(node_id)
        
        # Remove from storage
        del self._nodes[node_id]
        del self._adjacency[node_id]
        del self._reverse_adjacency[node_id]
        
        self._node_counter -= 1
        
        self.logger.debug(f"Deleted node: {node_id}")
        return True
    
    # ==========================================================================
    # EDGE OPERATIONS
    # ==========================================================================
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        strength: RelationshipStrength = RelationshipStrength.MODERATE,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        edge_id: Optional[str] = None,
    ) -> Optional[GraphEdge]:
        """
        Add an edge between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            strength: Strength of relationship
            weight: Edge weight
            properties: Edge properties
            metadata: Additional metadata
            edge_id: Optional edge ID (auto-generated if not provided)
            
        Returns:
            Created GraphEdge or None if nodes don't exist
        """
        if source_id not in self._nodes:
            self.logger.warning(f"Source node {source_id} not found")
            return None
        
        if target_id not in self._nodes:
            self.logger.warning(f"Target node {target_id} not found")
            return None
        
        if edge_id is None:
            edge_id = self._generate_edge_id()
        
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            strength=strength,
            weight=weight,
            properties=properties or {},
            metadata=metadata or {},
        )
        
        self._edges[edge_id] = edge
        self._adjacency[source_id].add(edge_id)
        self._reverse_adjacency[target_id].add(edge_id)
        self._edges_by_type[edge_type].add(edge_id)
        
        self._edge_counter += 1
        
        self.logger.debug(
            f"Added edge: {source_id} -> {target_id} ({edge_type.value}, {strength.value})"
        )
        return edge
    
    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        """Get an edge by ID."""
        return self._edges.get(edge_id)
    
    def get_edges(self, source_id: str) -> List[GraphEdge]:
        """Get all edges from a source node."""
        return [self._edges[eid] for eid in self._adjacency.get(source_id, set()) if eid in self._edges]
    
    def get_reverse_edges(self, target_id: str) -> List[GraphEdge]:
        """Get all edges to a target node."""
        return [self._edges[eid] for eid in self._reverse_adjacency.get(target_id, set()) if eid in self._edges]
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[GraphEdge]:
        """Get all edges of a specific type."""
        return [self._edges[eid] for eid in self._edges_by_type.get(edge_type, set()) if eid in self._edges]
    
    def update_edge(
        self,
        edge_id: str,
        strength: Optional[RelationshipStrength] = None,
        weight: Optional[float] = None,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[GraphEdge]:
        """Update an edge's properties."""
        edge = self._edges.get(edge_id)
        if not edge:
            return None
        
        if strength:
            edge.strength = strength
        if weight is not None:
            edge.weight = weight
        if properties:
            edge.properties.update(properties)
        if metadata:
            edge.metadata.update(metadata)
        
        edge.updated_at = datetime.now()
        return edge
    
    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge."""
        if edge_id not in self._edges:
            return False
        
        edge = self._edges[edge_id]
        self._adjacency[edge.source_id].discard(edge_id)
        self._reverse_adjacency[edge.target_id].discard(edge_id)
        self._edges_by_type[edge.edge_type].discard(edge_id)
        
        del self._edges[edge_id]
        
        self._edge_counter -= 1
        
        self.logger.debug(f"Deleted edge: {edge_id}")
        return True
    
    # ==========================================================================
    # QUERY OPERATIONS
    # ==========================================================================
    
    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = 'both',
    ) -> List[GraphNode]:
        """
        Get neighbors of a node.
        
        Args:
            node_id: Node ID
            edge_type: Filter by edge type
            direction: 'out', 'in', or 'both'
            
        Returns:
            List of neighbor nodes
        """
        neighbors = set()
        
        if direction in ('out', 'both'):
            for edge_id in self._adjacency.get(node_id, set()):
                edge = self._edges.get(edge_id)
                if edge and (edge_type is None or edge.edge_type == edge_type):
                    neighbors.add(edge.target_id)
        
        if direction in ('in', 'both'):
            for edge_id in self._reverse_adjacency.get(node_id, set()):
                edge = self._edges.get(edge_id)
                if edge and (edge_type is None or edge.edge_type == edge_type):
                    neighbors.add(edge.source_id)
        
        return [self._nodes[nid] for nid in neighbors if nid in self._nodes]
    
    def get_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> List[List[GraphNode]]:
        """
        Find paths between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_depth: Maximum path depth
            
        Returns:
            List of paths (each path is a list of nodes)
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return []
        
        if source_id == target_id:
            return [[self._nodes[source_id]]]
        
        # BFS to find paths
        visited = set()
        queue = [(source_id, [source_id])]
        paths = []
        
        while queue and max_depth > 0:
            node_id, path = queue.pop(0)
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            
            for neighbor in self.get_neighbors(node_id, direction='out'):
                if neighbor.id in visited:
                    continue
                
                new_path = path + [neighbor.id]
                
                if neighbor.id == target_id:
                    paths.append([self._nodes[nid] for nid in new_path])
                    continue
                
                if len(new_path) < max_depth:
                    queue.append((neighbor.id, new_path))
        
        return paths
    
    def get_subgraph(
        self,
        node_ids: Set[str],
        edge_types: Optional[Set[EdgeType]] = None,
    ) -> 'KnowledgeGraph':
        """
        Extract a subgraph containing specific nodes.
        
        Args:
            node_ids: Set of node IDs to include
            edge_types: Filter by edge types
            
        Returns:
            New KnowledgeGraph instance
        """
        subgraph = KnowledgeGraph(self.config)
        
        # Add nodes
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node:
                subgraph.add_node(
                    node_type=node.node_type,
                    label=node.label,
                    properties=node.properties,
                    node_id=node.id,
                    metadata=node.metadata,
                )
        
        # Add edges
        for edge_id, edge in self._edges.items():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                if edge_types is None or edge.edge_type in edge_types:
                    subgraph.add_edge(
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                        edge_type=edge.edge_type,
                        strength=edge.strength,
                        weight=edge.weight,
                        properties=edge.properties,
                        metadata=edge.metadata,
                        edge_id=edge_id,
                    )
        
        return subgraph
    
    # ==========================================================================
    # PATTERN RELATIONSHIPS
    # ==========================================================================
    
    def create_pattern_relationships(
        self,
        pattern_nodes: List[GraphNode],
        market_nodes: List[GraphNode],
        correlation_matrix: Dict[Tuple[str, str], float],
    ) -> int:
        """
        Create relationships between patterns and markets based on correlations.
        
        Args:
            pattern_nodes: List of pattern nodes
            market_nodes: List of market nodes
            correlation_matrix: Dict mapping (pattern_id, market_id) to correlation value
            
        Returns:
            Number of edges created
        """
        created = 0
        
        for pattern in pattern_nodes:
            for market in market_nodes:
                key = (pattern.id, market.id)
                corr = correlation_matrix.get(key, 0.0)
                
                if abs(corr) > 0.3:
                    if corr > 0:
                        edge_type = EdgeType.CORRELATES_WITH
                    else:
                        edge_type = EdgeType.OPPOSITE_OF
                    
                    if abs(corr) > 0.8:
                        strength = RelationshipStrength.STRONG
                    elif abs(corr) > 0.6:
                        strength = RelationshipStrength.MODERATE
                    else:
                        strength = RelationshipStrength.WEAK
                    
                    edge = self.add_edge(
                        source_id=pattern.id,
                        target_id=market.id,
                        edge_type=edge_type,
                        strength=strength,
                        weight=abs(corr),
                        properties={'correlation': corr},
                    )
                    if edge:
                        created += 1
        
        self.logger.info(f"Created {created} pattern-market relationships")
        return created
    
    # ==========================================================================
    # UTILITY OPERATIONS
    # ==========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'total_nodes': len(self._nodes),
            'total_edges': len(self._edges),
            'nodes_by_type': {
                k.value: len(v) for k, v in self._nodes_by_type.items()
            },
            'edges_by_type': {
                k.value: len(v) for k, v in self._edges_by_type.items()
            },
            'density': self._calculate_density(),
            'avg_degree': self._calculate_avg_degree(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary."""
        return {
            'nodes': [n.to_dict() for n in self._nodes.values()],
            'edges': [e.to_dict() for e in self._edges.values()],
            'statistics': self.get_statistics(),
        }
    
    def to_networkx(self) -> Any:
        """
        Convert to NetworkX graph.
        
        Returns:
            NetworkX graph if available, None otherwise
        """
        try:
            import networkx as nx
            G = nx.DiGraph()
            
            for node in self._nodes.values():
                G.add_node(
                    node.id,
                    type=node.node_type.value,
                    label=node.label,
                    **node.properties
                )
            
            for edge in self._edges.values():
                G.add_edge(
                    edge.source_id,
                    edge.target_id,
                    type=edge.edge_type.value,
                    strength=edge.strength.value,
                    weight=edge.weight,
                    **edge.properties
                )
            
            return G
            
        except ImportError:
            self.logger.warning("NetworkX not available")
            return None
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _generate_node_id(self) -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"node_{self._node_counter}_{datetime.now().timestamp()}"
    
    def _generate_edge_id(self) -> str:
        """Generate a unique edge ID."""
        self._edge_counter += 1
        return f"edge_{self._edge_counter}_{datetime.now().timestamp()}"
    
    def _calculate_density(self) -> float:
        """Calculate graph density."""
        n = len(self._nodes)
        e = len(self._edges)
        if n < 2:
            return 0.0
        max_edges = n * (n - 1)
        return e / max_edges if max_edges > 0 else 0.0
    
    def _calculate_avg_degree(self) -> float:
        """Calculate average degree."""
        if not self._nodes:
            return 0.0
        total_degree = sum(
            len(self._adjacency.get(nid, set())) + len(self._reverse_adjacency.get(nid, set()))
            for nid in self._nodes
        )
        return total_degree / len(self._nodes)


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_knowledge_graph(config: Config) -> KnowledgeGraph:
    """
    Factory function for KnowledgeGraph creation.
    
    Args:
        config: Application configuration
        
    Returns:
        KnowledgeGraph instance
    """
    return KnowledgeGraph(config)