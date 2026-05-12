"""Repositories — thin pydantic <-> SQLModel adapters.

Each repo owns one table and exposes a small typed surface; the agents and
API never touch SQLAlchemy directly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlmodel import select

from deepresearch.schemas.agents import AgentRole, AgentStep, StepStatus
from deepresearch.schemas.memory import MemoryEvent, MemoryEventKind, MemoryType
from deepresearch.schemas.models import ModelCallRecord
from deepresearch.schemas.runs import Depth, ResearchRun, RunRequest, RunStatus
from deepresearch.schemas.search import Citation, SearchDocument
from deepresearch.storage.db import StorageEngine
from deepresearch.storage.tables import (
    AgentStepTable,
    MemoryEventTable,
    ModelCallRecordTable,
    ResearchRunTable,
    SearchDocumentTable,
)


def _json_dumps(obj: object) -> str:
    return json.dumps(obj, default=str)


@dataclass
class RunRepo:
    engine: StorageEngine

    def create(self, run: ResearchRun) -> ResearchRun:
        req = run.request
        row = ResearchRunTable(
            id=run.id,
            user_id=req.user_id,
            project_id=req.project_id,
            query=req.query,
            depth=req.depth.value,
            max_searches=req.max_searches,
            model_profile=req.model_profile,
            memory_profile=req.memory_profile,
            status=run.status.value,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
            report_md=run.report_md,
            citations_json=_json_dumps([c.model_dump() for c in run.citations]),
            error=run.error,
            privacy_envelope_json=req.privacy_envelope.model_dump_json(),
            metrics_json=run.metrics.model_dump_json(),
        )
        with self.engine.session_scope() as s:
            s.add(row)
        return run

    def mark_running(self, run_id: UUID) -> None:
        with self.engine.session_scope() as s:
            row = s.get(ResearchRunTable, run_id)
            if row is None:
                raise KeyError(run_id)
            row.status = RunStatus.running.value
            row.started_at = datetime.utcnow()
            s.add(row)

    def mark_done(self, run: ResearchRun) -> None:
        with self.engine.session_scope() as s:
            row = s.get(ResearchRunTable, run.id)
            if row is None:
                raise KeyError(run.id)
            row.status = run.status.value
            row.finished_at = run.finished_at or datetime.utcnow()
            row.report_md = run.report_md
            row.citations_json = _json_dumps([c.model_dump() for c in run.citations])
            row.error = run.error
            row.metrics_json = run.metrics.model_dump_json()
            s.add(row)

    def get(self, run_id: UUID) -> ResearchRun | None:
        with self.engine.session_scope() as s:
            row = s.get(ResearchRunTable, run_id)
            if row is None:
                return None
            return _row_to_run(row)


def _row_to_run(row: ResearchRunTable) -> ResearchRun:
    from deepresearch.schemas.privacy import PrivacyEnvelope
    from deepresearch.schemas.runs import RunMetrics

    envelope = PrivacyEnvelope.model_validate_json(row.privacy_envelope_json)
    citations = [Citation.model_validate(c) for c in json.loads(row.citations_json or "[]")]
    metrics = (
        RunMetrics.model_validate_json(row.metrics_json) if row.metrics_json else RunMetrics()
    )
    req = RunRequest(
        query=row.query,
        user_id=row.user_id,
        project_id=row.project_id,
        depth=Depth(row.depth),
        max_searches=row.max_searches,
        model_profile=row.model_profile,
        memory_profile=row.memory_profile,
        privacy_envelope=envelope,
    )
    return ResearchRun(
        id=row.id,
        request=req,
        status=RunStatus(row.status),
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        report_md=row.report_md,
        citations=citations,
        error=row.error,
        metrics=metrics,
    )


@dataclass
class StepRepo:
    engine: StorageEngine

    def append(self, step: AgentStep) -> AgentStep:
        row = AgentStepTable(
            id=step.id,
            run_id=step.run_id,
            seq=step.seq,
            role=step.role.value,
            status=step.status.value,
            input_json=_json_dumps(step.input),
            output_json=_json_dumps(step.output),
            started_at=step.started_at,
            finished_at=step.finished_at,
            latency_ms=step.latency_ms,
            model_call_ids_json=_json_dumps([str(i) for i in step.model_call_ids]),
            error=step.error,
        )
        with self.engine.session_scope() as s:
            s.add(row)
        return step

    def list_for_run(self, run_id: UUID) -> list[AgentStep]:
        with self.engine.session_scope() as s:
            stmt = (
                select(AgentStepTable)
                .where(AgentStepTable.run_id == run_id)
                .order_by(AgentStepTable.seq)  # type: ignore[arg-type]
            )
            rows: Sequence[AgentStepTable] = s.exec(stmt).all()
            return [
                AgentStep(
                    id=r.id,
                    run_id=r.run_id,
                    seq=r.seq,
                    role=AgentRole(r.role),
                    status=StepStatus(r.status),
                    input=json.loads(r.input_json),
                    output=json.loads(r.output_json),
                    started_at=r.started_at,
                    finished_at=r.finished_at,
                    latency_ms=r.latency_ms,
                    model_call_ids=[UUID(i) for i in json.loads(r.model_call_ids_json or "[]")],
                    error=r.error,
                )
                for r in rows
            ]


@dataclass
class SearchDocRepo:
    engine: StorageEngine

    def append(self, run_id: UUID, doc: SearchDocument) -> SearchDocument:
        row = SearchDocumentTable(
            id=doc.id,
            run_id=run_id,
            subquestion_id=doc.subquestion_id,
            url=doc.url,
            title=doc.title,
            snippet=doc.snippet,
            content_md=doc.content_md,
            fetched_at=doc.fetched_at,
            source_provider=doc.source_provider,
            score=doc.score,
        )
        with self.engine.session_scope() as s:
            s.add(row)
        return doc


@dataclass
class MemoryEventRepo:
    engine: StorageEngine

    def append(self, event: MemoryEvent) -> MemoryEvent:
        row = MemoryEventTable(
            id=event.id,
            run_id=event.run_id,
            user_id=event.user_id,
            project_id=event.project_id,
            kind=event.kind.value,
            memory_type=event.memory_type.value,
            query=event.query,
            payload_json=_json_dumps([r.model_dump() for r in event.records]),
            reme_ids_json=_json_dumps([r.id for r in event.records]),
            backend=event.backend,
            at=event.at,
        )
        with self.engine.session_scope() as s:
            s.add(row)
        return event

    def list_for_run(self, run_id: UUID) -> list[MemoryEvent]:
        with self.engine.session_scope() as s:
            stmt = select(MemoryEventTable).where(MemoryEventTable.run_id == run_id)
            rows: Sequence[MemoryEventTable] = s.exec(stmt).all()
            out: list[MemoryEvent] = []
            for r in rows:
                from deepresearch.schemas.memory import MemoryRecord

                records = [MemoryRecord.model_validate(p) for p in json.loads(r.payload_json)]
                out.append(
                    MemoryEvent(
                        id=r.id,
                        run_id=r.run_id,
                        user_id=r.user_id,
                        project_id=r.project_id,
                        kind=MemoryEventKind(r.kind),
                        memory_type=MemoryType(r.memory_type),
                        query=r.query,
                        records=records,
                        backend=r.backend,
                        at=r.at,
                    )
                )
            return out


@dataclass
class ModelCallRepo:
    engine: StorageEngine

    def append(self, rec: ModelCallRecord) -> ModelCallRecord:
        row = ModelCallRecordTable(
            id=rec.id,
            run_id=rec.run_id,
            step_id=rec.step_id,
            endpoint_name=rec.endpoint_name,
            model_id=rec.model_id,
            role=rec.role,
            prompt_tokens=rec.prompt_tokens,
            completion_tokens=rec.completion_tokens,
            latency_ms=rec.latency_ms,
            started_at=rec.started_at,
            privacy_envelope_json=rec.envelope.model_dump_json(),
            request_hash=rec.request_hash,
            error=rec.error,
        )
        with self.engine.session_scope() as s:
            s.add(row)
        return rec

    def list_for_run(self, run_id: UUID) -> list[ModelCallRecord]:
        with self.engine.session_scope() as s:
            stmt = select(ModelCallRecordTable).where(ModelCallRecordTable.run_id == run_id)
            rows: Sequence[ModelCallRecordTable] = s.exec(stmt).all()
            from deepresearch.schemas.privacy import PrivacyEnvelope

            return [
                ModelCallRecord(
                    id=r.id,
                    run_id=r.run_id,
                    step_id=r.step_id,
                    endpoint_name=r.endpoint_name,
                    model_id=r.model_id,
                    role=r.role,
                    prompt_tokens=r.prompt_tokens,
                    completion_tokens=r.completion_tokens,
                    latency_ms=r.latency_ms,
                    started_at=r.started_at,
                    envelope=PrivacyEnvelope.model_validate_json(r.privacy_envelope_json),
                    request_hash=r.request_hash,
                    error=r.error,
                )
                for r in rows
            ]


@dataclass
class Repositories:
    runs: RunRepo
    steps: StepRepo
    docs: SearchDocRepo
    memory_events: MemoryEventRepo
    model_calls: ModelCallRepo

    @classmethod
    def from_engine(cls, engine: StorageEngine) -> Repositories:
        return cls(
            runs=RunRepo(engine),
            steps=StepRepo(engine),
            docs=SearchDocRepo(engine),
            memory_events=MemoryEventRepo(engine),
            model_calls=ModelCallRepo(engine),
        )
