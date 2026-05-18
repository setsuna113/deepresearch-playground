"""AppConfig — typed mirror of config.yaml."""

from __future__ import annotations

from pydantic import BaseModel, Field

from deepresearch.schemas.privacy import PrivacyEnvelope


class AppSection(BaseModel):
    data_dir: str = "./data"
    sqlite_path: str = "./data/deepresearch.sqlite"
    log_level: str = "INFO"
    log_json: bool = True


class EndpointConfig(BaseModel):
    base_url: str
    api_key: str = "EMPTY"
    model_id: str
    role_hint: str | None = None


class ModelProfileConfig(BaseModel):
    """Maps role name -> endpoint name for one model profile.

    Phase 1.5: the new LangGraph role set lives here. Legacy STORM
    fields stay optional for transition compatibility — they're never
    routed to but their presence in YAML doesn't break loading.
    """

    # LangGraph roles (used by RouterChatModel post-1.5).
    supervisor: str = "local"
    researcher: str = "local"
    compressor: str = "local"
    final_report: str = "local"
    reflector: str = "local"

    # Legacy STORM roles — accepted but unused; remove once all YAML
    # configs are updated.
    planner: str | None = None
    searcher: str | None = None
    reader: str | None = None
    synthesizer: str | None = None

    model_config = {"extra": "ignore"}


class ModelsSection(BaseModel):
    endpoints: dict[str, EndpointConfig]
    profiles: dict[str, ModelProfileConfig]


class ReMeVectorStoreConfig(BaseModel):
    backend: str = "qdrant"
    url: str = "http://localhost:6333"
    collection_prefix: str = "dr_reme_"


class ReMeLLMConfig(BaseModel):
    endpoint: str = "local"


class ReMeEmbeddingConfig(BaseModel):
    provider: str = "local"
    model_id: str = "bge-m3"


class ReMeSection(BaseModel):
    enabled: bool = True
    working_dir: str = "./data/reme"
    vector_store: ReMeVectorStoreConfig = Field(default_factory=ReMeVectorStoreConfig)
    llm: ReMeLLMConfig = Field(default_factory=ReMeLLMConfig)
    embedding: ReMeEmbeddingConfig = Field(default_factory=ReMeEmbeddingConfig)


class WorkingMemoryConfig(BaseModel):
    # If `local_path` is set, qdrant-client runs in embedded mode against that
    # on-disk directory and `qdrant_url` is ignored. This lets Phase 1 run
    # without docker. Set `local_path: null` to force server mode.
    local_path: str | None = "./data/qdrant_working"
    qdrant_url: str = "http://localhost:6333"
    collection_template: str = "dr_working_{user}_{project}"
    # `hash-fallback` for semantic-blind tests; any sentence-transformers
    # model_id for real embeddings (default: BAAI/bge-small-en-v1.5).
    embedding_model: str = "BAAI/bge-small-en-v1.5"


class MemoryProfileConfig(BaseModel):
    personal_top_k: int = 5
    task_top_k: int = 5
    tool_top_k: int = 3
    working_top_k: int = 0
    score_floor: float = 0.55


class MemorySection(BaseModel):
    reme: ReMeSection = Field(default_factory=ReMeSection)
    working: WorkingMemoryConfig = Field(default_factory=WorkingMemoryConfig)
    profiles: dict[str, MemoryProfileConfig] = Field(default_factory=dict)


class SearchProviderConfig(BaseModel):
    api_key: str = ""
    max_results: int = 8


class FetchConfig(BaseModel):
    user_agent: str = "deepresearch-playground/0.1 (research)"
    timeout_s: int = 20
    max_bytes: int = 2_500_000
    respect_robots: bool = True


class SearchSection(BaseModel):
    default_provider: str = "tavily"
    providers: dict[str, SearchProviderConfig] = Field(default_factory=dict)
    fetch: FetchConfig = Field(default_factory=FetchConfig)


class ApiSection(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8765


class PrivacySection(BaseModel):
    default_envelope: PrivacyEnvelope = Field(default_factory=PrivacyEnvelope.default_public)


class EvalSection(BaseModel):
    golden_suite_path: str = "./scripts/golden_queries.yaml"
    judge_endpoint: str = "judge"


class AppConfig(BaseModel):
    app: AppSection = Field(default_factory=AppSection)
    models: ModelsSection
    memory: MemorySection = Field(default_factory=MemorySection)
    search: SearchSection = Field(default_factory=SearchSection)
    api: ApiSection = Field(default_factory=ApiSection)
    privacy: PrivacySection = Field(default_factory=PrivacySection)
    eval: EvalSection = Field(default_factory=EvalSection)
