"""SQLite engine + session factory."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine


@dataclass
class StorageEngine:
    engine: Engine

    def session(self) -> Session:
        return Session(self.engine)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        sess = self.session()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()


@lru_cache(maxsize=4)
def get_engine(sqlite_path: str) -> StorageEngine:
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(
        f"sqlite:///{path.resolve()}",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    return StorageEngine(engine=eng)


def init_db(sqlite_path: str) -> StorageEngine:
    """Create all tables. Idempotent."""
    # Ensure table classes are imported before metadata is materialized.
    from deepresearch.storage import tables  # noqa: F401

    se = get_engine(sqlite_path)
    SQLModel.metadata.create_all(se.engine)
    return se
