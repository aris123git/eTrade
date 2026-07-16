"""
=========================================================
eTrade Database Engine
=========================================================
Author : Aristide & ChatGPT
=========================================================
"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager

from config import DATABASE_PATH


class Database:

    def __init__(self):

        self.connection = sqlite3.connect(
            DATABASE_PATH,
            check_same_thread=False
        )

        self.connection.row_factory = sqlite3.Row

        self.cursor = self.connection.cursor()

        self.enable_performance()

    # --------------------------------------------------

    def enable_performance(self):

        self.cursor.execute("PRAGMA journal_mode=WAL;")

        self.cursor.execute("PRAGMA synchronous=NORMAL;")

        self.cursor.execute("PRAGMA temp_store=MEMORY;")

        self.cursor.execute("PRAGMA foreign_keys=ON;")

        self.connection.commit()

    # --------------------------------------------------

    @contextmanager
    def transaction(self):

        try:

            yield

            self.connection.commit()

        except Exception:

            self.connection.rollback()

            raise

    # --------------------------------------------------

    def execute(self, query, values=()):

        self.cursor.execute(query, values)

        return self.cursor

    # --------------------------------------------------

    def executemany(self, query, values):

        self.cursor.executemany(query, values)

    # --------------------------------------------------

    def fetchone(self):

        return self.cursor.fetchone()

    # --------------------------------------------------

    def fetchall(self):

        return self.cursor.fetchall()

    # --------------------------------------------------

    def close(self):

        self.connection.close()