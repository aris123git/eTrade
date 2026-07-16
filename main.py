import MetaTrader5 as mt5

from database.database import Database
from database.schema import create_schema
from database.indexes import create_indexes
from database.seed import seed

from collector.symbol_manager import SymbolManager
from collector.downloader import Downloader

print("=" * 60)
print("eTrade Collector")
print("=" * 60)

if not mt5.initialize():

    raise Exception("MT5 not initialized")

db = Database()

create_schema(db)

create_indexes(db)

seed(db)

print("Database ready.")

manager = SymbolManager(db)

manager.discover()

print("Markets discovered.")

collector = Downloader(db)

collector.download_all()

db.close()

mt5.shutdown()

print("Finished.")