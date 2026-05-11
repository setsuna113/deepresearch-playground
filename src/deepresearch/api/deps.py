"""Dependency assembly — used by both the FastAPI app and the Typer CLI.

`build_dependencies(config)` constructs the whole graph (storage engine,
repos, recorder, model client, router, memory service, tools) and returns
a `RunDependencies` ready to feed to the orchestrator.
"""

from __future__ import annotations

from deepresearch.agents.context import RunDependencies
from deepresearch.config.schema import AppConfig
from deepresearch.memory.service import MemoryService
from deepresearch.models.client import ModelClient
from deepresearch.models.endpoints import EndpointSet
from deepresearch.models.router import Router
from deepresearch.observability.trace import TraceRecorder
from deepresearch.storage.db import init_db
from deepresearch.storage.repository import Repositories
from deepresearch.tools.fetch.fetcher import PageFetcher
from deepresearch.tools.registry import ToolRegistry
from deepresearch.tools.search.serpapi import SerpAPISearch
from deepresearch.tools.search.tavily import TavilySearch


async def build_dependencies(config: AppConfig) -> RunDependencies:
    engine = init_db(config.app.sqlite_path)
    repos = Repositories.from_engine(engine)
    recorder = TraceRecorder(repos=repos)

    endpoints = EndpointSet.from_config(config.models)
    model_client = ModelClient(endpoints=endpoints, recorder=recorder)
    router = Router(endpoints=endpoints)

    memory = await MemoryService.create(config.memory, recorder=recorder)

    search_providers = {}
    if "tavily" in config.search.providers:
        search_providers["tavily"] = TavilySearch(
            api_key=config.search.providers["tavily"].api_key,
            timeout_s=config.search.fetch.timeout_s,
        )
    if "serpapi" in config.search.providers:
        search_providers["serpapi"] = SerpAPISearch(
            api_key=config.search.providers["serpapi"].api_key,
            timeout_s=config.search.fetch.timeout_s,
        )
    fetcher = PageFetcher(
        user_agent=config.search.fetch.user_agent,
        timeout_s=config.search.fetch.timeout_s,
        max_bytes=config.search.fetch.max_bytes,
    )
    tools = ToolRegistry(search=search_providers, fetcher=fetcher)

    return RunDependencies(
        config=config,
        repos=repos,
        recorder=recorder,
        model_client=model_client,
        router=router,
        memory=memory,
        tools=tools,
    )
