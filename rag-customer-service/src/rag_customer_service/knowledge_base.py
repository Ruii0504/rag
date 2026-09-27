import logging
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from rag_customer_service.ingestion import DocumentIngestor
from rag_customer_service.models import (
    Chunk,
    Document,
    DocumentStatus,
    KnowledgeBase,
)
from rag_customer_service.qwen import QwenClient
from rag_customer_service.storage import DuplicateDocumentError, Storage
from rag_customer_service.vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)


class KnowledgeBaseIngestionError(RuntimeError):
    """文档摄取失败且自动清理未完全成功。"""


class DocumentNotFoundError(ValueError):
    """文档不存在或不属于指定知识库。"""


class KnowledgeBaseManager:
    def __init__(
        self,
        storage: Storage,
        ingestor: DocumentIngestor,
        qwen: QwenClient,
        vector_store: ChromaVectorStore,
    ) -> None:
        self._storage = storage
        self._ingestor = ingestor
        self._qwen = qwen
        self._vector_store = vector_store

    def create_knowledge_base(self, name: str) -> KnowledgeBase:
        return self._storage.create_knowledge_base(name)

    def list_knowledge_bases(self) -> list[KnowledgeBase]:
        return self._storage.list_knowledge_bases()

    def list_documents(self, kb_id: str) -> list[Document]:
        return self._storage.list_documents(kb_id)

    def ingest_document(
        self,
        kb_id: str,
        filename: str,
        content: bytes,
        on_progress: Callable[[str], None] | None = None,
    ) -> Document:
        report = on_progress or (lambda _: None)
        document_id = str(uuid4())
        file_started = False
        vector_started = False
        metadata_started = False

        try:
            report("校验")
            prepared = self._ingestor.prepare(filename, content)
            report("切分")

            if any(
                document.content_hash == prepared.content_hash
                for document in self._storage.list_documents(kb_id)
            ):
                raise DuplicateDocumentError(f"知识库 {kb_id} 中已存在相同内容")

            chunks = tuple(
                Chunk(
                    id=f"{document_id}:{prepared_chunk.chunk_index}",
                    kb_id=kb_id,
                    document_id=document_id,
                    document_name=prepared_chunk.document_name,
                    heading_path=prepared_chunk.heading_path,
                    chunk_index=prepared_chunk.chunk_index,
                    content=prepared_chunk.content,
                    content_hash=prepared.content_hash,
                )
                for prepared_chunk in prepared.chunks
            )

            report("向量化")
            embeddings = self._qwen.embed_documents(
                [chunk.content for chunk in chunks]
            )

            report("保存")
            file_started = True
            file_path = self._storage.save_document_file(
                kb_id,
                document_id,
                prepared.normalized_content,
            )

            vector_started = True
            self._vector_store.add_chunks(chunks, embeddings)

            document = Document(
                id=document_id,
                kb_id=kb_id,
                filename=filename,
                content_hash=prepared.content_hash,
                file_path=file_path,
                chunk_count=len(chunks),
                status=DocumentStatus.READY,
                created_at=datetime.now(UTC),
            )
            metadata_started = True
            self._storage.add_document(document)
            report("完成")
            return document
        except Exception as error:
            cleanup_errors = self._compensate_ingestion(
                kb_id=kb_id,
                document_id=document_id,
                metadata_started=metadata_started,
                vector_started=vector_started,
                file_started=file_started,
            )
            if cleanup_errors:
                raise KnowledgeBaseIngestionError(
                    "文档处理失败，且自动清理未完成；请检查本地数据状态"
                ) from error
            raise

    def delete_document(self, kb_id: str, document_id: str) -> None:
        if not any(
            document.id == document_id
            for document in self._storage.list_documents(kb_id)
        ):
            raise DocumentNotFoundError("文档不存在或不属于该知识库")
        self._vector_store.delete_document(document_id)
        self._storage.delete_document_file(kb_id, document_id)
        self._storage.delete_document(kb_id, document_id)

    def delete_knowledge_base(self, kb_id: str) -> None:
        documents = self._storage.list_documents(kb_id)
        self._vector_store.delete_knowledge_base(kb_id)
        for document in documents:
            self._storage.delete_document_file(kb_id, document.id)
        self._storage.delete_knowledge_base(kb_id)

    def _compensate_ingestion(
        self,
        *,
        kb_id: str,
        document_id: str,
        metadata_started: bool,
        vector_started: bool,
        file_started: bool,
    ) -> list[Exception]:
        cleanup_errors = []
        actions: list[tuple[str, Callable[[], None]]] = []
        if metadata_started:
            actions.append(
                (
                    "SQLite 元数据",
                    lambda: self._storage.delete_document(kb_id, document_id),
                )
            )
        if vector_started:
            actions.append(
                (
                    "Chroma 向量",
                    lambda: self._vector_store.delete_document(document_id),
                )
            )
        if file_started:
            actions.append(
                (
                    "原文件",
                    lambda: self._storage.delete_document_file(kb_id, document_id),
                )
            )

        for target, action in actions:
            try:
                action()
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
                logger.warning("摄取失败后未能清理%s", target)
        return cleanup_errors
