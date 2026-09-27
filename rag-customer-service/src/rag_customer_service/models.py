from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class DocumentStatus(str, Enum):
    READY = "ready"


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    id: str
    name: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    kb_id: str
    filename: str
    content_hash: str
    file_path: Path
    chunk_count: int
    status: DocumentStatus
    created_at: datetime


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    content: str
    document_name: str
    heading_path: tuple[str, ...]
    chunk_index: int


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    filename: str
    normalized_content: str
    content_hash: str
    chunks: tuple[PreparedChunk, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    kb_id: str
    document_id: str
    document_name: str
    heading_path: tuple[str, ...]
    chunk_index: int
    content: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class VectorMatch:
    chunk: Chunk
    distance: float


class EvidenceSupportStatus(str, Enum):
    RELATED = "related"
    SUPPORTING = "supporting"


@dataclass(frozen=True, slots=True)
class Evidence:
    reference_id: int
    kb_id: str
    document_id: str
    document_name: str
    heading_path: tuple[str, ...]
    chunk_index: int
    content: str
    distance: float
    similarity: float
    subquestion_id: str = ""
    subquestion: str = ""
    support_status: EvidenceSupportStatus = EvidenceSupportStatus.RELATED
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class SubQuestion:
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class QuestionDecomposition:
    standalone_question: str
    subquestions: tuple[SubQuestion, ...]
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    subquestion_id: str
    answerable: bool
    evidence_ids: tuple[int, ...] = ()
    missing_information: str = ""
    reason: str = ""
    supported_points: tuple[str, ...] = ()


class AnswerValidationStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class AnswerPart:
    subquestion_id: str
    question: str
    answer: str
    evidence_ids: tuple[int, ...]
    validation_status: AnswerValidationStatus = AnswerValidationStatus.PENDING
    validation_reason: str = ""


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    evidences: tuple[Evidence, ...]
    has_sufficient_evidence: bool
    reason: str = ""


class TraceEventType(str, Enum):
    NODE_STATUS = "node_status"
    TOOL_STATUS = "tool_status"
    EVIDENCE = "evidence"
    ANSWER_DELTA = "answer_delta"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TraceEvent:
    type: TraceEventType
    stage: str
    message: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
