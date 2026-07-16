"""
database/repositories/currency_repository.py - Currency Repository

RESPONSIBILITY:
Manage currency data in the database.

ARCHITECTURAL PRINCIPLES:
1. Single Responsibility - Only handles currency data
2. Repository Pattern - Mediates between domain and data mapping
3. Type Safety - Uses Currency model with validation
4. Business Logic - Currency-specific queries and operations

SCALABILITY VISION:
This repository will handle currency definitions, exchange rates,
and currency relationships across the entire platform.

VERSION: 1.0.0
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Set, Tuple
from uuid import uuid4

from database.core.connection import DatabaseManager
from database.repositories.base_repository import BaseRepository
from database.models.currency import Currency, CurrencyType, CurrencyStatus


logger = logging.getLogger(__name__)


class CurrencyRepository(BaseRepository[Currency]):
    """
    Repository for currency data.
    
    Provides CRUD operations and currency-specific queries.
    
    USAGE:
        repo = CurrencyRepository(db_manager)
        
        # Create a currency
        usd = repo.create(
            code="USD",
            name="US Dollar",
            currency_type=CurrencyType.FIAT,
            symbol="$",
        )
        
        # Find major currencies
        major = repo.find_major()
        
        # Find by code
        eur = repo.find_by_code("EUR")
    """
    
    TABLE = "currencies"
    MODEL = Currency
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the currency repository."""
        super().__init__(db_manager)
        self.logger = logging.getLogger(__name__)
    
    # ==========================================================================
    # CREATE OPERATIONS
    # ==========================================================================
    
    def create(
        self,
        code: str,
        name: str,
        currency_type: CurrencyType,
        symbol: Optional[str] = None,
        iso_number: Optional[int] = None,
        decimals: int = 2,
        description: Optional[str] = None,
        status: CurrencyStatus = CurrencyStatus.ACTIVE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Currency:
        """
        Create a new currency.
        
        Args:
            code: Currency code (e.g., "USD")
            name: Currency name (e.g., "US Dollar")
            currency_type: Type of currency
            symbol: Currency symbol (e.g., "$")
            iso_number: ISO numeric code (e.g., 840)
            decimals: Number of decimal places
            description: Currency description
            status: Currency status
            metadata: Additional metadata
            
        Returns:
            Created Currency object
        """
        currency = Currency(
            currency_id=None,
            currency_uuid=str(uuid4()),
            code=code.upper(),
            name=name,
            currency_type=currency_type,
            symbol=symbol,
            iso_number=iso_number,
            decimals=decimals,
            description=description,
            status=status,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        # Insert into database
        data = self._entity_to_dict(currency)
        currency_id = self.insert_dict(data)
        currency.currency_id = currency_id
        
        self.logger.info(f"✅ Currency created: {code} (ID: {currency_id})")
        return currency
    
    def create_or_update(
        self,
        code: str,
        data: Dict[str, Any],
    ) -> Currency:
        """
        Create or update a currency by code.
        
        Args:
            code: Currency code
            data: Currency data
            
        Returns:
            Currency object
        """
        existing = self.find_by_code(code)
        
        if existing:
            # Update existing
            self.update(existing.currency_id, data)
            return self.find_by_code(code)
        else:
            # Create new
            return self.create(code=code, **data)
    
    def create_batch(
        self,
        currencies: List[Dict[str, Any]],
    ) -> List[Currency]:
        """
        Create multiple currencies in batch.
        
        Args:
            currencies: List of currency data dictionaries
            
        Returns:
            List of created Currency objects
        """
        results = []
        for data in currencies:
            try:
                currency = self.create(**data)
                results.append(currency)
            except Exception as e:
                self.logger.warning(f"Failed to create currency {data.get('code')}: {e}")
        
        self.logger.info(f"✅ Batch created {len(results)} currencies")
        return results
    
    def create_default_currencies(self) -> List[Currency]:
        """
        Create default currency set (major fiat and crypto).
        
        Returns:
            List of created Currency objects
        """
        default_currencies = [
            # Major fiat currencies
            {'code': 'USD', 'name': 'US Dollar', 'currency_type': CurrencyType.FIAT, 'symbol': '$', 'iso_number': 840},
            {'code': 'EUR', 'name': 'Euro', 'currency_type': CurrencyType.FIAT, 'symbol': '€', 'iso_number': 978},
            {'code': 'GBP', 'name': 'British Pound', 'currency_type': CurrencyType.FIAT, 'symbol': '£', 'iso_number': 826},
            {'code': 'JPY', 'name': 'Japanese Yen', 'currency_type': CurrencyType.FIAT, 'symbol': '¥', 'iso_number': 392},
            {'code': 'CHF', 'name': 'Swiss Franc', 'currency_type': CurrencyType.FIAT, 'symbol': 'Fr', 'iso_number': 756},
            {'code': 'AUD', 'name': 'Australian Dollar', 'currency_type': CurrencyType.FIAT, 'symbol': 'A$', 'iso_number': 36},
            {'code': 'CAD', 'name': 'Canadian Dollar', 'currency_type': CurrencyType.FIAT, 'symbol': 'C$', 'iso_number': 124},
            {'code': 'NZD', 'name': 'New Zealand Dollar', 'currency_type': CurrencyType.FIAT, 'symbol': 'NZ$', 'iso_number': 554},
            
            # Major crypto currencies
            {'code': 'BTC', 'name': 'Bitcoin', 'currency_type': CurrencyType.CRYPTO, 'symbol': '₿', 'decimals': 8},
            {'code': 'ETH', 'name': 'Ethereum', 'currency_type': CurrencyType.CRYPTO, 'symbol': 'Ξ', 'decimals': 18},
            {'code': 'SOL', 'name': 'Solana', 'currency_type': CurrencyType.CRYPTO, 'symbol': '◎', 'decimals': 9},
            {'code': 'XRP', 'name': 'Ripple', 'currency_type': CurrencyType.CRYPTO, 'symbol': 'XRP', 'decimals': 6},
            {'code': 'ADA', 'name': 'Cardano', 'currency_type': CurrencyType.CRYPTO, 'symbol': '₳', 'decimals': 6},
            {'code': 'DOGE', 'name': 'Dogecoin', 'currency_type': CurrencyType.CRYPTO, 'symbol': 'Ð', 'decimals': 8},
            
            # Commodity currencies
            {'code': 'XAU', 'name': 'Gold', 'currency_type': CurrencyType.COMMODITY, 'symbol': 'Au', 'decimals': 2},
            {'code': 'XAG', 'name': 'Silver', 'currency_type': CurrencyType.COMMODITY, 'symbol': 'Ag', 'decimals': 2},
            {'code': 'XPT', 'name': 'Platinum', 'currency_type': CurrencyType.COMMODITY, 'symbol': 'Pt', 'decimals': 2},
            {'code': 'XPD', 'name': 'Palladium', 'currency_type': CurrencyType.COMMODITY, 'symbol': 'Pd', 'decimals': 2},
        ]
        
        return self.create_batch(default_currencies)
    
    # ==========================================================================
    # READ OPERATIONS
    # ==========================================================================
    
    def find_by_code(self, code: str) -> Optional[Currency]:
        """
        Find a currency by code.
        
        Args:
            code: Currency code
            
        Returns:
            Currency object or None
        """
        row = self.adapter.fetch_one(
            f"SELECT * FROM {self.TABLE} WHERE code = ?",
            (code.upper(),)
        )
        return self._row_to_entity(row) if row else None
    
    def find_by_uuid(self, currency_uuid: str) -> Optional[Currency]:
        """
        Find a currency by UUID.
        
        Args:
            currency_uuid: Currency UUID
            
        Returns:
            Currency object or None
        """
        row = self.adapter.fetch_one(
            f"SELECT * FROM {self.TABLE} WHERE currency_uuid = ?",
            (currency_uuid,)
        )
        return self._row_to_entity(row) if row else None
    
    def find_by_type(self, currency_type: CurrencyType) -> List[Currency]:
        """
        Find all currencies of a specific type.
        
        Args:
            currency_type: Currency type
            
        Returns:
            List of Currency objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE currency_type = ?",
            (currency_type.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_active(self) -> List[Currency]:
        """
        Find all active currencies.
        
        Returns:
            List of active Currency objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE status = ?",
            (CurrencyStatus.ACTIVE.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_by_status(self, status: CurrencyStatus) -> List[Currency]:
        """
        Find all currencies with a specific status.
        
        Args:
            status: Currency status
            
        Returns:
            List of Currency objects
        """
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE status = ?",
            (status.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_fiat(self) -> List[Currency]:
        """Find all fiat currencies."""
        return self.find_by_type(CurrencyType.FIAT)
    
    def find_crypto(self) -> List[Currency]:
        """Find all crypto currencies."""
        return self.find_by_type(CurrencyType.CRYPTO)
    
    def find_commodities(self) -> List[Currency]:
        """Find all commodity currencies."""
        return self.find_by_type(CurrencyType.COMMODITY)
    
    def find_major(self) -> List[Currency]:
        """
        Find major currencies (USD, EUR, GBP, JPY, CHF, AUD, CAD, NZD).
        
        Returns:
            List of major Currency objects
        """
        major_codes = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
        placeholders = ','.join('?' for _ in major_codes)
        
        rows = self.adapter.fetch_all(
            f"SELECT * FROM {self.TABLE} WHERE code IN ({placeholders}) AND status = ?",
            tuple(major_codes) + (CurrencyStatus.ACTIVE.value,)
        )
        return [self._row_to_entity(row) for row in rows]
    
    def find_minor(self) -> List[Currency]:
        """
        Find minor currencies (non-major fiat).
        
        Returns:
            List of minor Currency objects
        """
        major_codes = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
        placeholders = ','.join('?' for _ in major_codes)
        
        rows = self.adapter.fetch_all(f"""
            SELECT * FROM {self.TABLE} 
            WHERE currency_type = ? 
            AND code NOT IN ({placeholders}) 
            AND status = ?
        """, (CurrencyType.FIAT.value,) + tuple(major_codes) + (CurrencyStatus.ACTIVE.value,))
        return [self._row_to_entity(row) for row in rows]
    
    def find_all_codes(self) -> List[str]:
        """
        Get all currency codes.
        
        Returns:
            List of currency codes
        """
        rows = self.adapter.fetch_all(
            f"SELECT code FROM {self.TABLE} WHERE status = ?",
            (CurrencyStatus.ACTIVE.value,)
        )
        return [row['code'] for row in rows]
    
    def find_by_iso_number(self, iso_number: int) -> Optional[Currency]:
        """
        Find a currency by ISO number.
        
        Args:
            iso_number: ISO numeric code
            
        Returns:
            Currency object or None
        """
        row = self.adapter.fetch_one(
            f"SELECT * FROM {self.TABLE} WHERE iso_number = ?",
            (iso_number,)
        )
        return self._row_to_entity(row) if row else None
    
    def find_by_symbol(self, symbol: str) -> Optional[Currency]:
        """
        Find a currency by symbol.
        
        Args:
            symbol: Currency symbol
            
        Returns:
            Currency object or None
        """
        row = self.adapter.fetch_one(
            f"SELECT * FROM {self.TABLE} WHERE symbol = ?",
            (symbol,)
        )
        return self._row_to_entity(row) if row else None
    
    def get_type_counts(self) -> Dict[str, int]:
        """
        Get count of currencies by type.
        
        Returns:
            Dictionary mapping type to count
        """
        rows = self.adapter.fetch_all(f"""
            SELECT currency_type, COUNT(*) as count 
            FROM {self.TABLE} 
            WHERE status = ?
            GROUP BY currency_type
        """, (CurrencyStatus.ACTIVE.value,))
        
        return {row['currency_type']: row['count'] for row in rows}
    
    def get_currency_pairs(self) -> List[Tuple[str, str]]:
        """
        Get all currency pairs (base, quote) for trading.
        
        Returns:
            List of (base_code, quote_code) tuples
        """
        # Get all active currencies
        currencies = self.find_active()
        pairs = []
        
        # Generate pairs (major combinations)
        major = self.find_major()
        for base in major:
            for quote in major:
                if base.code != quote.code:
                    pairs.append((base.code, quote.code))
        
        # Add crypto pairs
        crypto = self.find_crypto()
        fiat = self.find_fiat()
        for crypto_curr in crypto:
            for fiat_curr in fiat:
                pairs.append((crypto_curr.code, fiat_curr.code))
        
        return pairs
    
    # ==========================================================================
    # UPDATE OPERATIONS
    # ==========================================================================
    
    def update_status(self, currency_id: int, status: CurrencyStatus) -> bool:
        """
        Update currency status.
        
        Args:
            currency_id: Currency ID
            status: New status
            
        Returns:
            True if updated, False otherwise
        """
        return self.update(currency_id, {'status': status.value})
    
    def activate(self, currency_id: int) -> bool:
        """Activate a currency."""
        return self.update_status(currency_id, CurrencyStatus.ACTIVE)
    
    def deactivate(self, currency_id: int) -> bool:
        """Deactivate a currency."""
        return self.update_status(currency_id, CurrencyStatus.INACTIVE)
    
    def update_metadata(self, currency_id: int, metadata: Dict[str, Any]) -> bool:
        """
        Update currency metadata.
        
        Args:
            currency_id: Currency ID
            metadata: New metadata
            
        Returns:
            True if updated, False otherwise
        """
        existing = self.get_by_id(currency_id)
        if not existing:
            return False
        
        new_metadata = {**existing.metadata, **metadata}
        return self.update(currency_id, {'metadata': new_metadata})
    
    # ==========================================================================
    # DELETE OPERATIONS
    # ==========================================================================
    
    def delete_by_code(self, code: str) -> bool:
        """
        Delete a currency by code.
        
        Args:
            code: Currency code
            
        Returns:
            True if deleted, False otherwise
        """
        currency = self.find_by_code(code)
        if not currency:
            return False
        return self.delete(currency.currency_id)
    
    def archive(self, currency_id: int) -> bool:
        """
        Archive a currency (soft delete).
        
        Args:
            currency_id: Currency ID
            
        Returns:
            True if archived, False otherwise
        """
        return self.update_status(currency_id, CurrencyStatus.ARCHIVED)
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    def currency_exists(self, code: str) -> bool:
        """
        Check if a currency exists.
        
        Args:
            code: Currency code
            
        Returns:
            True if exists, False otherwise
        """
        return self.count(f"code = ?", (code.upper(),)) > 0
    
    def get_or_create(self, code: str, name: str = None) -> Currency:
        """
        Get a currency or create it if it doesn't exist.
        
        Args:
            code: Currency code
            name: Currency name (optional)
            
        Returns:
            Currency object
        """
        existing = self.find_by_code(code)
        if existing:
            return existing
        
        if name is None:
            name = code
        
        return self.create(
            code=code,
            name=name,
            currency_type=CurrencyType.UNKNOWN,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics.
        
        Returns:
            Dictionary with statistics
        """
        base_stats = super().get_statistics()
        
        active_count = self.count("status = ?", (CurrencyStatus.ACTIVE.value,))
        inactive_count = self.count("status = ?", (CurrencyStatus.INACTIVE.value,))
        archived_count = self.count("status = ?", (CurrencyStatus.ARCHIVED.value,))
        type_counts = self.get_type_counts()
        
        return {
            **base_stats,
            'active_currencies': active_count,
            'inactive_currencies': inactive_count,
            'archived_currencies': archived_count,
            'currency_types': type_counts,
            'fiat_count': len(self.find_fiat()),
            'crypto_count': len(self.find_crypto()),
            'commodity_count': len(self.find_commodities()),
        }
    
    def get_currency_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all currencies.
        
        Returns:
            Dictionary with currency summary
        """
        active = self.find_active()
        
        return {
            'total_active': len(active),
            'by_type': self.get_type_counts(),
            'major_currencies': [c.code for c in self.find_major()],
            'crypto_currencies': [c.code for c in self.find_crypto()],
            'commodity_currencies': [c.code for c in self.find_commodities()],
        }
    
    def get_currency_by_country(self, country: str) -> Optional[Currency]:
        """
        Find currency by country (if stored in metadata).
        
        Args:
            country: Country name
            
        Returns:
            Currency object or None
        """
        rows = self.adapter.fetch_all(f"""
            SELECT * FROM {self.TABLE} 
            WHERE metadata LIKE ? 
            AND status = ?
            LIMIT 1
        """, (f'%"{country}"%', CurrencyStatus.ACTIVE.value))
        
        if rows:
            return self._row_to_entity(rows[0])
        return None
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _entity_to_dict(self, currency: Currency) -> Dict[str, Any]:
        """Convert Currency entity to dictionary."""
        return {
            'currency_id': currency.currency_id,
            'currency_uuid': currency.currency_uuid,
            'code': currency.code,
            'name': currency.name,
            'currency_type': currency.currency_type.value if currency.currency_type else None,
            'symbol': currency.symbol,
            'iso_number': currency.iso_number,
            'decimals': currency.decimals,
            'description': currency.description,
            'status': currency.status.value if currency.status else None,
            'metadata': json.dumps(currency.metadata) if currency.metadata else '{}',
            'created_at': currency.created_at.isoformat() if currency.created_at else None,
            'updated_at': currency.updated_at.isoformat() if currency.updated_at else None,
        }
    
    def _row_to_entity(self, row: Dict[str, Any]) -> Currency:
        """Convert database row to Currency entity."""
        return Currency(
            currency_id=row['currency_id'],
            currency_uuid=row['currency_uuid'],
            code=row['code'],
            name=row['name'],
            currency_type=CurrencyType(row['currency_type']) if row['currency_type'] else None,
            symbol=row['symbol'],
            iso_number=row['iso_number'],
            decimals=row['decimals'],
            description=row['description'],
            status=CurrencyStatus(row['status']) if row['status'] else None,
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
        )
    
    def _get_id(self, currency: Currency) -> int:
        """Get ID from Currency entity."""
        return currency.currency_id


# ==============================================================================
# REGISTER IN REPOSITORY MANAGER
# ==============================================================================

# To be added to REPOSITORIES list in repository_manager.py:
# ('currencies', CurrencyRepository, None)