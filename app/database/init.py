"""Idempotently create configured persistence tables without dropping data."""

import logging

from app.config.settings import DatabaseSettings
from app.database.core import Database

logger = logging.getLogger(__name__)


def main() -> int:
    database = None
    try:
        database = Database(DatabaseSettings.from_env())
        if not database.configured:
            print("MySQL is not configured. Set MYSQL_USER and MYSQL_DATABASE first.")
            return 1
        database.check_connection()
        database.initialize()
        print("DataPilot database tables are ready.")
        return 0
    except Exception as exc:
        logger.error("Database initialization failed (%s)", type(exc).__name__)
        print("Database initialization failed. Check MySQL availability and environment settings.")
        return 1
    finally:
        if database is not None:
            database.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
