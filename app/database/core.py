from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import URL, create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config.settings import DatabaseSettings
from app.database.models import Base


class DatabaseUnavailableError(RuntimeError):
    """Database persistence is not configured or cannot be used."""


class Database:
    """SQLAlchemy engine/session boundary for MySQL and offline SQLite tests."""

    def __init__(self, settings: DatabaseSettings | None = None, *, url: str | URL | None = None) -> None:
        settings = settings or DatabaseSettings.from_env()
        self.engine: Engine | None = None
        if url is not None:
            engine_url = url
        elif settings.database_configured:
            engine_url = URL.create(
                "mysql+pymysql",
                username=settings.username,
                password=settings.password.get_secret_value() if settings.password else "",
                host=settings.host,
                port=settings.port,
                database=settings.database,
                query={"charset": "utf8mb4"},
            )
        else:
            engine_url = None

        if engine_url is not None:
            url_text = str(engine_url)
            is_memory_sqlite = url_text in {"sqlite://", "sqlite:///:memory:", "sqlite+pysqlite:///:memory:"}
            options = {"pool_pre_ping": True}
            if is_memory_sqlite:
                options.update(poolclass=StaticPool, connect_args={"check_same_thread": False})
            elif str(engine_url).startswith("mysql"):
                options["connect_args"] = {"connect_timeout": 5}
            self.engine = create_engine(engine_url, **options)
            if self.engine.dialect.name == "sqlite":
                event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)
            self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        else:
            self._session_factory = None

    @property
    def configured(self) -> bool:
        return self.engine is not None

    def check_connection(self) -> None:
        if self.engine is None:
            raise DatabaseUnavailableError("MySQL persistence is not configured.")
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def initialize(self) -> None:
        """Create missing tables only; this is idempotent and never drops data."""
        if self.engine is None:
            raise DatabaseUnavailableError("MySQL persistence is not configured.")
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        if self._session_factory is None:
            raise DatabaseUnavailableError("MySQL persistence is not configured.")
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        if self.engine is not None:
            self.engine.dispose()

    @staticmethod
    def _enable_sqlite_foreign_keys(connection, _record) -> None:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
