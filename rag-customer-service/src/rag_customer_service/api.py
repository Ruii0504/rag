import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from rag_customer_service.bootstrap import AppContainer
from rag_customer_service.models import (
    Document,
    Evidence,
    EvidenceDecision,
    KnowledgeBase,
    RetrievalResult,
    TraceEvent,
)


logger = logging.getLogger(__name__)


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("知识库名称不能为空")
        return name


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        content = value.strip()
        if not content:
            raise ValueError("历史消息不能为空")
        return content


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=100)
    kb_ids: list[str] = Field(min_length=1, max_length=50)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("问题不能为空")
        return question

    @field_validator("kb_ids")
    @classmethod
    def normalize_kb_ids(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(kb_id.strip() for kb_id in value if kb_id.strip()))
        if not normalized:
            raise ValueError("至少选择一个知识库")
        return normalized


def _json_value(value: Any) -> Any:
    if isinstance(value, Evidence):
        return {
            "reference_id": value.reference_id,
            "kb_id": value.kb_id,
            "document_id": value.document_id,
            "document_name": value.document_name,
            "heading_path": list(value.heading_path),
            "chunk_index": value.chunk_index,
            "content": value.content,
            "distance": value.distance,
            "similarity": value.similarity,
            "rerank_score": value.rerank_score,
            "subquestion_id": value.subquestion_id,
            "subquestion": value.subquestion,
            "support_status": value.support_status.value,
        }
    if isinstance(value, EvidenceDecision):
        return {
            "subquestion_id": value.subquestion_id,
            "answerable": value.answerable,
            "evidence_ids": list(value.evidence_ids),
            "missing_information": value.missing_information,
            "reason": value.reason,
        }
    if isinstance(value, RetrievalResult):
        return {
            "query": value.query,
            "evidences": [_json_value(evidence) for evidence in value.evidences],
            "has_sufficient_evidence": value.has_sufficient_evidence,
            "reason": value.reason,
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if is_dataclass(value) or isinstance(value, Path):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def _document_payload(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "kb_id": document.kb_id,
        "filename": document.filename,
        "chunk_count": document.chunk_count,
        "status": document.status.value,
        "created_at": document.created_at.isoformat(),
    }


def _knowledge_base_payload(
    knowledge_base: KnowledgeBase,
    documents: Sequence[Document],
) -> dict[str, Any]:
    return {
        "id": knowledge_base.id,
        "name": knowledge_base.name,
        "created_at": knowledge_base.created_at.isoformat(),
        "document_count": len(documents),
        "chunk_count": sum(document.chunk_count for document in documents),
    }


def _known_error(error: ValueError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))


def _sse_events(events: Iterator[TraceEvent]) -> Iterator[str]:
    for event in events:
        payload = {
            "type": event.type.value,
            "stage": event.stage,
            "message": event.message,
            "payload": _json_value(event.payload),
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(
    container: AppContainer,
    settings: Any,
    frontend_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="RAG 智能客服", version="0.1.0")

    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        knowledge_bases = container.knowledge_bases.list_knowledge_bases()
        documents = [
            document
            for knowledge_base in knowledge_bases
            for document in container.knowledge_bases.list_documents(knowledge_base.id)
        ]
        return {
            "status": "ready",
            "chat_model": settings.qwen_chat_model,
            "embedding_model": settings.qwen_embedding_model,
            **({
                "rerank_model": settings.qwen_rerank_model,
                "retrieval_top_k": settings.retrieval_top_k,
            } if getattr(settings, "qwen_rerank_model", None) else {}),
            "retrieval_score_threshold": settings.retrieval_score_threshold,
            "knowledge_base_count": len(knowledge_bases),
            "document_count": len(documents),
            "chunk_count": sum(document.chunk_count for document in documents),
        }

    @app.get("/api/knowledge-bases")
    def list_knowledge_bases() -> list[dict[str, Any]]:
        return [
            _knowledge_base_payload(
                knowledge_base,
                container.knowledge_bases.list_documents(knowledge_base.id),
            )
            for knowledge_base in container.knowledge_bases.list_knowledge_bases()
        ]

    @app.post("/api/knowledge-bases", status_code=status.HTTP_201_CREATED)
    def create_knowledge_base(request: KnowledgeBaseCreate) -> dict[str, Any]:
        try:
            knowledge_base = container.knowledge_bases.create_knowledge_base(request.name)
        except ValueError as error:
            raise _known_error(error) from error
        return _knowledge_base_payload(knowledge_base, [])

    @app.delete(
        "/api/knowledge-bases/{kb_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_knowledge_base(kb_id: str) -> Response:
        container.knowledge_bases.delete_knowledge_base(kb_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/knowledge-bases/{kb_id}/documents")
    def list_documents(kb_id: str) -> list[dict[str, Any]]:
        return [
            _document_payload(document)
            for document in container.knowledge_bases.list_documents(kb_id)
        ]

    @app.post(
        "/api/knowledge-bases/{kb_id}/documents",
        status_code=status.HTTP_201_CREATED,
    )
    async def upload_documents(
        kb_id: str,
        files: list[UploadFile] = File(...),
    ) -> Response:
        uploaded: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        error_status = status.HTTP_400_BAD_REQUEST
        for file in files:
            progress: list[str] = []
            try:
                document = container.knowledge_bases.ingest_document(
                    kb_id,
                    file.filename or "",
                    await file.read(),
                    progress.append,
                )
            except ValueError as error:
                errors.append({
                    "filename": file.filename or "未命名文件",
                    "detail": str(error),
                })
                continue
            except Exception:
                logger.exception("文档处理失败：%s", file.filename or "未命名文件")
                errors.append({
                    "filename": file.filename or "未命名文件",
                    "detail": "文档处理失败，请稍后重试",
                })
                error_status = status.HTTP_500_INTERNAL_SERVER_ERROR
                continue
            uploaded.append({**_document_payload(document), "progress": progress})
        if errors and not uploaded:
            raise HTTPException(status_code=error_status, detail=errors[0]["detail"])
        return JSONResponse(
            status_code=status.HTTP_207_MULTI_STATUS if errors else status.HTTP_201_CREATED,
            content={"documents": uploaded, "errors": errors},
        )

    @app.delete(
        "/api/knowledge-bases/{kb_id}/documents/{document_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_document(kb_id: str, document_id: str) -> Response:
        try:
            container.knowledge_bases.delete_document(kb_id, document_id)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/api/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        events = container.agent.stream(
            request.question,
            [message.model_dump() for message in request.history],
            request.kb_ids,
        )
        return StreamingResponse(
            _sse_events(events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if frontend_dir is not None:
        frontend_root = frontend_dir.resolve()
        index_path = frontend_root / "index.html"
        assets_path = frontend_root / "assets"
        if assets_path.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

        @app.get("/{requested_path:path}", include_in_schema=False)
        def serve_frontend(requested_path: str) -> FileResponse:
            if requested_path == "api" or requested_path.startswith("api/"):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
            candidate = (frontend_root / requested_path).resolve()
            if candidate.is_relative_to(frontend_root) and candidate.is_file():
                return FileResponse(candidate)
            if index_path.is_file():
                return FileResponse(index_path)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    return app
