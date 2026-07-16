"""
knowledge/ontology.py - Market Ontology Module

RESPONSIBILITY:
Define and manage the ontology of market concepts and relationships.

ARCHITECTURAL PRINCIPLES:
1. Pure knowledge representation - No data storage, no I/O, no business logic
2. Hierarchical classification of market concepts
3. Type-safe results with validation
4. Extensible ontology with inheritance

WHAT IT NEVER DOES:
- ❌ Connect to MT5
- ❌ Store data
- ❌ Download data
- ❌ Make trading decisions

VERSION: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Set, Union, Callable
from enum import Enum
from collections import defaultdict

from core.config import Config
from core.exceptions import DiscoveryError, DataValidationError


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    'MarketDomain',
    'ConceptType',
    'RelationType',
    'OntologyConcept',
    'OntologyRelation',
    'MarketOntology',
    'create_market_ontology',
]


# ==============================================================================
# ENUMS
# ==============================================================================

class MarketDomain(Enum):
    """Market domains in the ontology."""
    FOREX = "forex"
    CRYPTO = "crypto"
    INDICES = "indices"
    COMMODITIES = "commodities"
    STOCKS = "stocks"
    BONDS = "bonds"
    FUTURES = "futures"
    OPTIONS = "options"
    DERIVATIVES = "derivatives"
    ETF = "etf"
    UNKNOWN = "unknown"


class ConceptType(Enum):
    """Types of concepts in the ontology."""
    # Asset classes
    ASSET = "asset"
    CURRENCY = "currency"
    COMMODITY = "commodity"
    EQUITY = "equity"
    INDEX = "index"
    
    # Market concepts
    MARKET = "market"
    EXCHANGE = "exchange"
    SESSION = "session"
    TIMEFRAME = "timeframe"
    
    # Trading concepts
    STRATEGY = "strategy"
    INDICATOR = "indicator"
    PATTERN = "pattern"
    SIGNAL = "signal"
    
    # Risk concepts
    RISK = "risk"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    
    # Fundamental concepts
    ECONOMIC = "economic"
    EVENT = "event"
    SENTIMENT = "sentiment"
    
    # Data concepts
    CANDLE = "candle"
    ORDER = "order"
    POSITION = "position"
    ACCOUNT = "account"


class RelationType(Enum):
    """Types of relations in the ontology."""
    IS_A = "is_a"                   # Inheritance
    HAS_A = "has_a"                 # Composition
    PART_OF = "part_of"             # Aggregation
    RELATED_TO = "related_to"       # General relation
    TRADES_ON = "trades_on"         # Market exchange
    INFLUENCES = "influences"       # Causal relation
    CORRELATES = "correlates"       # Correlation
    PRECEDES = "precedes"           # Temporal
    SUCCEEDS = "succeeds"           # Temporal
    SIMILAR_TO = "similar_to"       # Similarity
    OPPOSITE_OF = "opposite_of"     # Opposition
    DERIVED_FROM = "derived_from"   # Derivation
    APPLIES_TO = "applies_to"       # Application
    USES = "uses"                   # Usage
    REQUIRES = "requires"           # Dependency


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class OntologyConcept:
    """A concept in the ontology."""
    id: str
    name: str
    concept_type: ConceptType
    domain: MarketDomain
    description: str
    parent_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'concept_type': self.concept_type.value,
            'domain': self.domain.value,
            'description': self.description,
            'parent_id': self.parent_id,
            'properties': self.properties,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }
    
    def is_a(self, concept_type: ConceptType) -> bool:
        """Check if concept is of a given type."""
        return self.concept_type == concept_type


@dataclass
class OntologyRelation:
    """A relation in the ontology."""
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'source': self.source_id,
            'target': self.target_id,
            'relation_type': self.relation_type.value,
            'properties': self.properties,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
        }


# ==============================================================================
# MARKET ONTOLOGY
# ==============================================================================

class MarketOntology:
    """
    Market ontology engine.
    
    Defines and manages the ontology of market concepts and relationships.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the market ontology.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Ontology storage
        self._concepts: Dict[str, OntologyConcept] = {}
        self._relations: List[OntologyRelation] = []
        self._concepts_by_type: Dict[ConceptType, Set[str]] = defaultdict(set)
        self._concepts_by_domain: Dict[MarketDomain, Set[str]] = defaultdict(set)
        self._children: Dict[str, Set[str]] = defaultdict(set)
        self._parents: Dict[str, Set[str]] = defaultdict(set)
        
        # Build ontology
        self._build_ontology()
        
        self.logger.info(
            f"✅ MarketOntology initialized: "
            f"{len(self._concepts)} concepts, {len(self._relations)} relations"
        )
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def get_concept(self, concept_id: str) -> Optional[OntologyConcept]:
        """Get a concept by ID."""
        return self._concepts.get(concept_id)
    
    def get_concept_by_name(self, name: str) -> Optional[OntologyConcept]:
        """Get a concept by name."""
        for concept in self._concepts.values():
            if concept.name == name:
                return concept
        return None
    
    def get_concepts_by_type(self, concept_type: ConceptType) -> List[OntologyConcept]:
        """Get all concepts of a given type."""
        return [
            self._concepts[concept_id]
            for concept_id in self._concepts_by_type.get(concept_type, set())
            if concept_id in self._concepts
        ]
    
    def get_concepts_by_domain(self, domain: MarketDomain) -> List[OntologyConcept]:
        """Get all concepts in a given domain."""
        return [
            self._concepts[concept_id]
            for concept_id in self._concepts_by_domain.get(domain, set())
            if concept_id in self._concepts
        ]
    
    def get_children(self, concept_id: str) -> List[OntologyConcept]:
        """Get all children of a concept."""
        return [
            self._concepts[child_id]
            for child_id in self._children.get(concept_id, set())
            if child_id in self._concepts
        ]
    
    def get_parents(self, concept_id: str) -> List[OntologyConcept]:
        """Get all parents of a concept."""
        return [
            self._concepts[parent_id]
            for parent_id in self._parents.get(concept_id, set())
            if parent_id in self._concepts
        ]
    
    def get_ancestors(self, concept_id: str) -> List[OntologyConcept]:
        """Get all ancestors of a concept."""
        ancestors = []
        visited = set()
        queue = list(self._parents.get(concept_id, set()))
        
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in self._concepts:
                ancestors.append(self._concepts[current])
                queue.extend(self._parents.get(current, set()))
        
        return ancestors
    
    def get_descendants(self, concept_id: str) -> List[OntologyConcept]:
        """Get all descendants of a concept."""
        descendants = []
        visited = set()
        queue = list(self._children.get(concept_id, set()))
        
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in self._concepts:
                descendants.append(self._concepts[current])
                queue.extend(self._children.get(current, set()))
        
        return descendants
    
    def get_relations(
        self,
        concept_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = 'both',
    ) -> List[OntologyRelation]:
        """
        Get relations for a concept.
        
        Args:
            concept_id: Concept ID
            relation_type: Filter by relation type
            direction: 'out', 'in', or 'both'
            
        Returns:
            List of OntologyRelation objects
        """
        relations = []
        
        for relation in self._relations:
            if direction in ('out', 'both') and relation.source_id == concept_id:
                if relation_type is None or relation.relation_type == relation_type:
                    relations.append(relation)
            if direction in ('in', 'both') and relation.target_id == concept_id:
                if relation_type is None or relation.relation_type == relation_type:
                    relations.append(relation)
        
        return relations
    
    def add_concept(
        self,
        name: str,
        concept_type: ConceptType,
        domain: MarketDomain,
        description: str,
        parent_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        concept_id: Optional[str] = None,
    ) -> OntologyConcept:
        """
        Add a concept to the ontology.
        
        Args:
            name: Concept name
            concept_type: Concept type
            domain: Market domain
            description: Description
            parent_id: Parent concept ID
            properties: Additional properties
            concept_id: Optional concept ID
            
        Returns:
            Created OntologyConcept
        """
        if concept_id is None:
            concept_id = self._generate_concept_id(name)
        
        if concept_id in self._concepts:
            self.logger.warning(f"Concept {concept_id} already exists")
            return self._concepts[concept_id]
        
        concept = OntologyConcept(
            id=concept_id,
            name=name,
            concept_type=concept_type,
            domain=domain,
            description=description,
            parent_id=parent_id,
            properties=properties or {},
        )
        
        self._concepts[concept_id] = concept
        self._concepts_by_type[concept_type].add(concept_id)
        self._concepts_by_domain[domain].add(concept_id)
        
        if parent_id:
            self._children[parent_id].add(concept_id)
            self._parents[concept_id].add(parent_id)
        
        self.logger.debug(f"Added concept: {name} ({concept_type.value})")
        return concept
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: Optional[Dict[str, Any]] = None,
    ) -> OntologyRelation:
        """
        Add a relation to the ontology.
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            relation_type: Relation type
            properties: Additional properties
            
        Returns:
            Created OntologyRelation
        """
        if source_id not in self._concepts:
            raise DataValidationError(f"Source concept {source_id} not found")
        
        if target_id not in self._concepts:
            raise DataValidationError(f"Target concept {target_id} not found")
        
        relation = OntologyRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
        )
        
        self._relations.append(relation)
        
        self.logger.debug(
            f"Added relation: {source_id} -> {target_id} ({relation_type.value})"
        )
        return relation
    
    def get_ancestor_types(self, concept_id: str) -> Set[ConceptType]:
        """Get all ancestor concept types."""
        types = set()
        for ancestor in self.get_ancestors(concept_id):
            types.add(ancestor.concept_type)
        return types
    
    def is_subtype_of(self, concept_id: str, concept_type: ConceptType) -> bool:
        """Check if a concept is a subtype of a given type."""
        if concept_id not in self._concepts:
            return False
        
        if self._concepts[concept_id].concept_type == concept_type:
            return True
        
        ancestor_types = self.get_ancestor_types(concept_id)
        return concept_type in ancestor_types
    
    def get_domain_concepts(self, domain: MarketDomain) -> List[OntologyConcept]:
        """Get all concepts in a domain."""
        return self.get_concepts_by_domain(domain)
    
    def get_related_concepts(
        self,
        concept_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = 'both',
    ) -> List[OntologyConcept]:
        """
        Get all concepts related to a concept.
        
        Args:
            concept_id: Concept ID
            relation_type: Filter by relation type
            direction: 'out', 'in', or 'both'
            
        Returns:
            List of OntologyConcept objects
        """
        relations = self.get_relations(concept_id, relation_type, direction)
        related_ids = set()
        
        for relation in relations:
            if direction in ('out', 'both'):
                related_ids.add(relation.target_id)
            if direction in ('in', 'both'):
                related_ids.add(relation.source_id)
        
        return [
            self._concepts[rel_id]
            for rel_id in related_ids
            if rel_id in self._concepts
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ontology statistics."""
        return {
            'total_concepts': len(self._concepts),
            'total_relations': len(self._relations),
            'concepts_by_type': {
                k.value: len(v) for k, v in self._concepts_by_type.items()
            },
            'concepts_by_domain': {
                k.value: len(v) for k, v in self._concepts_by_domain.items()
            },
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ontology to dictionary."""
        return {
            'concepts': [c.to_dict() for c in self._concepts.values()],
            'relations': [r.to_dict() for r in self._relations],
            'statistics': self.get_statistics(),
        }
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _generate_concept_id(self, name: str) -> str:
        """Generate a concept ID."""
        import re
        from datetime import datetime
        base = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        return f"{base}_{int(datetime.now().timestamp())}"
    
    def _build_ontology(self):
        """
        Build the market ontology with all concepts and relations.
        """
        # ======================================================================
        # ASSET CLASSES
        # ======================================================================
        
        self.add_concept(
            name="Asset",
            concept_type=ConceptType.ASSET,
            domain=MarketDomain.UNKNOWN,
            description="Base asset class",
            concept_id="asset",
        )
        
        self.add_concept(
            name="Currency",
            concept_type=ConceptType.CURRENCY,
            domain=MarketDomain.FOREX,
            description="Currency asset",
            parent_id="asset",
            concept_id="currency",
        )
        
        self.add_concept(
            name="Commodity",
            concept_type=ConceptType.COMMODITY,
            domain=MarketDomain.COMMODITIES,
            description="Commodity asset",
            parent_id="asset",
            concept_id="commodity",
        )
        
        self.add_concept(
            name="Equity",
            concept_type=ConceptType.EQUITY,
            domain=MarketDomain.STOCKS,
            description="Equity asset",
            parent_id="asset",
            concept_id="equity",
        )
        
        self.add_concept(
            name="Index",
            concept_type=ConceptType.INDEX,
            domain=MarketDomain.INDICES,
            description="Index asset",
            parent_id="asset",
            concept_id="index",
        )
        
        # ======================================================================
        # FOREX CONCEPTS
        # ======================================================================
        
        self.add_concept(
            name="ForexMarket",
            concept_type=ConceptType.MARKET,
            domain=MarketDomain.FOREX,
            description="Forex market",
            parent_id="asset",
            concept_id="forex_market",
        )
        
        self.add_concept(
            name="MajorPair",
            concept_type=ConceptType.MARKET,
            domain=MarketDomain.FOREX,
            description="Major currency pair",
            parent_id="forex_market",
            concept_id="major_pair",
        )
        
        self.add_concept(
            name="MinorPair",
            concept_type=ConceptType.MARKET,
            domain=MarketDomain.FOREX,
            description="Minor currency pair",
            parent_id="forex_market",
            concept_id="minor_pair",
        )
        
        self.add_concept(
            name="ExoticPair",
            concept_type=ConceptType.MARKET,
            domain=MarketDomain.FOREX,
            description="Exotic currency pair",
            parent_id="forex_market",
            concept_id="exotic_pair",
        )
        
        # ======================================================================
        # CRYPTO CONCEPTS
        # ======================================================================
        
        self.add_concept(
            name="CryptoMarket",
            concept_type=ConceptType.MARKET,
            domain=MarketDomain.CRYPTO,
            description="Cryptocurrency market",
            parent_id="asset",
            concept_id="crypto_market",
        )
        
        self.add_concept(
            name="Bitcoin",
            concept_type=ConceptType.CURRENCY,
            domain=MarketDomain.CRYPTO,
            description="Bitcoin cryptocurrency",
            parent_id="crypto_market",
            concept_id="bitcoin",
        )
        
        self.add_concept(
            name="Ethereum",
            concept_type=ConceptType.CURRENCY,
            domain=MarketDomain.CRYPTO,
            description="Ethereum cryptocurrency",
            parent_id="crypto_market",
            concept_id="ethereum",
        )
        
        # ======================================================================
        # INDICES CONCEPTS
        # ======================================================================
        
        self.add_concept(
            name="IndexMarket",
            concept_type=ConceptType.MARKET,
            domain=MarketDomain.INDICES,
            description="Index market",
            parent_id="asset",
            concept_id="index_market",
        )
        
        self.add_concept(
            name="S&P500",
            concept_type=ConceptType.INDEX,
            domain=MarketDomain.INDICES,
            description="S&P 500 index",
            parent_id="index_market",
            concept_id="sp500",
        )
        
        self.add_concept(
            name="NASDAQ",
            concept_type=ConceptType.INDEX,
            domain=MarketDomain.INDICES,
            description="NASDAQ index",
            parent_id="index_market",
            concept_id="nasdaq",
        )
        
        self.add_concept(
            name="DowJones",
            concept_type=ConceptType.INDEX,
            domain=MarketDomain.INDICES,
            description="Dow Jones Industrial Average",
            parent_id="index_market",
            concept_id="dow_jones",
        )
        
        # ======================================================================
        # COMMODITY CONCEPTS
        # ======================================================================
        
        self.add_concept(
            name="CommodityMarket",
            concept_type=ConceptType.MARKET,
            domain=MarketDomain.COMMODITIES,
            description="Commodity market",
            parent_id="asset",
            concept_id="commodity_market",
        )
        
        self.add_concept(
            name="Gold",
            concept_type=ConceptType.COMMODITY,
            domain=MarketDomain.COMMODITIES,
            description="Gold commodity",
            parent_id="commodity_market",
            concept_id="gold",
        )
        
        self.add_concept(
            name="Silver",
            concept_type=ConceptType.COMMODITY,
            domain=MarketDomain.COMMODITIES,
            description="Silver commodity",
            parent_id="commodity_market",
            concept_id="silver",
        )
        
        self.add_concept(
            name="Oil",
            concept_type=ConceptType.COMMODITY,
            domain=MarketDomain.COMMODITIES,
            description="Oil commodity",
            parent_id="commodity_market",
            concept_id="oil",
        )
        
        # ======================================================================
        # TRADING CONCEPTS
        # ======================================================================
        
        self.add_concept(
            name="Strategy",
            concept_type=ConceptType.STRATEGY,
            domain=MarketDomain.UNKNOWN,
            description="Trading strategy",
            concept_id="strategy",
        )
        
        self.add_concept(
            name="TrendFollowing",
            concept_type=ConceptType.STRATEGY,
            domain=MarketDomain.UNKNOWN,
            description="Trend following strategy",
            parent_id="strategy",
            concept_id="trend_following",
        )
        
        self.add_concept(
            name="MeanReversion",
            concept_type=ConceptType.STRATEGY,
            domain=MarketDomain.UNKNOWN,
            description="Mean reversion strategy",
            parent_id="strategy",
            concept_id="mean_reversion",
        )
        
        self.add_concept(
            name="Breakout",
            concept_type=ConceptType.STRATEGY,
            domain=MarketDomain.UNKNOWN,
            description="Breakout strategy",
            parent_id="strategy",
            concept_id="breakout",
        )
        
        # ======================================================================
        # INDICATOR CONCEPTS
        # ======================================================================
        
        self.add_concept(
            name="Indicator",
            concept_type=ConceptType.INDICATOR,
            domain=MarketDomain.UNKNOWN,
            description="Technical indicator",
            concept_id="indicator",
        )
        
        self.add_concept(
            name="RSI",
            concept_type=ConceptType.INDICATOR,
            domain=MarketDomain.UNKNOWN,
            description="Relative Strength Index",
            parent_id="indicator",
            concept_id="rsi",
        )
        
        self.add_concept(
            name="MACD",
            concept_type=ConceptType.INDICATOR,
            domain=MarketDomain.UNKNOWN,
            description="Moving Average Convergence Divergence",
            parent_id="indicator",
            concept_id="macd",
        )
        
        self.add_concept(
            name="BollingerBands",
            concept_type=ConceptType.INDICATOR,
            domain=MarketDomain.UNKNOWN,
            description="Bollinger Bands",
            parent_id="indicator",
            concept_id="bollinger_bands",
        )
        
        self.add_concept(
            name="ATR",
            concept_type=ConceptType.INDICATOR,
            domain=MarketDomain.UNKNOWN,
            description="Average True Range",
            parent_id="indicator",
            concept_id="atr",
        )
        
        # ======================================================================
        # PATTERN CONCEPTS
        # ======================================================================
        
        self.add_concept(
            name="Pattern",
            concept_type=ConceptType.PATTERN,
            domain=MarketDomain.UNKNOWN,
            description="Market pattern",
            concept_id="pattern",
        )
        
        self.add_concept(
            name="Doji",
            concept_type=ConceptType.PATTERN,
            domain=MarketDomain.UNKNOWN,
            description="Doji candle pattern",
            parent_id="pattern",
            concept_id="doji",
        )
        
        self.add_concept(
            name="Hammer",
            concept_type=ConceptType.PATTERN,
            domain=MarketDomain.UNKNOWN,
            description="Hammer candle pattern",
            parent_id="pattern",
            concept_id="hammer",
        )
        
        self.add_concept(
            name="ShootingStar",
            concept_type=ConceptType.PATTERN,
            domain=MarketDomain.UNKNOWN,
            description="Shooting star candle pattern",
            parent_id="pattern",
            concept_id="shooting_star",
        )
        
        self.add_concept(
            name="Engulfing",
            concept_type=ConceptType.PATTERN,
            domain=MarketDomain.UNKNOWN,
            description="Engulfing pattern",
            parent_id="pattern",
            concept_id="engulfing",
        )
        
        self.add_concept(
            name="HeadAndShoulders",
            concept_type=ConceptType.PATTERN,
            domain=MarketDomain.UNKNOWN,
            description="Head and shoulders pattern",
            parent_id="pattern",
            concept_id="head_shoulders",
        )
        
        # ======================================================================
        # RISK CONCEPTS
        # ======================================================================
        
        self.add_concept(
            name="Risk",
            concept_type=ConceptType.RISK,
            domain=MarketDomain.UNKNOWN,
            description="Risk concept",
            concept_id="risk",
        )
        
        self.add_concept(
            name="Volatility",
            concept_type=ConceptType.VOLATILITY,
            domain=MarketDomain.UNKNOWN,
            description="Volatility concept",
            parent_id="risk",
            concept_id="volatility",
        )
        
        self.add_concept(
            name="Liquidity",
            concept_type=ConceptType.LIQUIDITY,
            domain=MarketDomain.UNKNOWN,
            description="Liquidity concept",
            parent_id="risk",
            concept_id="liquidity",
        )
        
        # ======================================================================
        # DATA CONCEPTS
        # ======================================================================
        
        self.add_concept(
            name="Candle",
            concept_type=ConceptType.CANDLE,
            domain=MarketDomain.UNKNOWN,
            description="Candle data",
            concept_id="candle",
        )
        
        self.add_concept(
            name="Order",
            concept_type=ConceptType.ORDER,
            domain=MarketDomain.UNKNOWN,
            description="Order data",
            concept_id="order",
        )
        
        self.add_concept(
            name="Position",
            concept_type=ConceptType.POSITION,
            domain=MarketDomain.UNKNOWN,
            description="Position data",
            concept_id="position",
        )
        
        self.add_concept(
            name="Account",
            concept_type=ConceptType.ACCOUNT,
            domain=MarketDomain.UNKNOWN,
            description="Account data",
            concept_id="account",
        )
        
        # ======================================================================
        # RELATIONS
        # ======================================================================
        
        # Asset relations
        self.add_relation("currency", "forex_market", RelationType.PART_OF)
        self.add_relation("commodity", "commodity_market", RelationType.PART_OF)
        self.add_relation("equity", "stock_market", RelationType.PART_OF)
        self.add_relation("index", "index_market", RelationType.PART_OF)
        
        # Forex relations
        self.add_relation("major_pair", "forex_market", RelationType.PART_OF)
        self.add_relation("minor_pair", "forex_market", RelationType.PART_OF)
        self.add_relation("exotic_pair", "forex_market", RelationType.PART_OF)
        
        # Indicator relations
        self.add_relation("rsi", "indicator", RelationType.IS_A)
        self.add_relation("macd", "indicator", RelationType.IS_A)
        self.add_relation("bollinger_bands", "indicator", RelationType.IS_A)
        self.add_relation("atr", "indicator", RelationType.IS_A)
        
        # Pattern relations
        self.add_relation("doji", "pattern", RelationType.IS_A)
        self.add_relation("hammer", "pattern", RelationType.IS_A)
        self.add_relation("shooting_star", "pattern", RelationType.IS_A)
        self.add_relation("engulfing", "pattern", RelationType.IS_A)
        self.add_relation("head_shoulders", "pattern", RelationType.IS_A)
        
        # Strategy relations
        self.add_relation("trend_following", "strategy", RelationType.IS_A)
        self.add_relation("mean_reversion", "strategy", RelationType.IS_A)
        self.add_relation("breakout", "strategy", RelationType.IS_A)
        
        # Risk relations
        self.add_relation("volatility", "risk", RelationType.IS_A)
        self.add_relation("liquidity", "risk", RelationType.IS_A)
        
        # Data relations
        self.add_relation("candle", "data", RelationType.IS_A)
        self.add_relation("order", "data", RelationType.IS_A)
        self.add_relation("position", "data", RelationType.IS_A)
        self.add_relation("account", "data", RelationType.IS_A)
        
        self.logger.info(f"Built ontology with {len(self._concepts)} concepts and {len(self._relations)} relations")


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_market_ontology(config: Config) -> MarketOntology:
    """
    Factory function for MarketOntology creation.
    
    Args:
        config: Application configuration
        
    Returns:
        MarketOntology instance
    """
    return MarketOntology(config)