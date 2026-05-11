"""FastAPI app factory.

Usage:
  uvicorn deepresearch.api.app:create_app --factory --port 8765
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from deepresearch.api.deps import build_dependencies
from deepresearch.api.routes import health, memory, runs
from deepresearch.config import get_config
from deepresearch.observability.logging import configure_logging


@asynccontextmanager
async def _lifespan(app: FastAPI):
    config = get_config()
    configure_logging(level=config.app.log_level, json=config.app.log_json)
    deps = await build_dependencies(config)
    app.state.deps = deps
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="deepresearch-playground", version="0.1.0", lifespan=_lifespan)
    app.include_router(health.router)
    app.include_router(runs.router)
    app.include_router(memory.router)
    return app
