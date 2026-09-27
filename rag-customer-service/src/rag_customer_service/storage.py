import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rag_customer_service.models import (
    Document,
    DocumentStatus,
    KnowledgeBase,
)


class DuplicateKnowledgeBaseError(ValueError):
    """知识库名称已存在。"""


class DuplicateDocumentError(ValueError):
    """同一知识库中已存在相同内容。"""


class Storage:
    def __init__(self, database_path: Path, uploads_path: Path) -> None:
        self.database_path = Path(database_path)
        self.uploads_path = Path(uploads_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    kb_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id)
                        ON DELETE CASCADE,
                    UNIQUE (kb_id, content_hash)
                );
                """
            )

    def create_knowledge_base(self, name: str) -> KnowledgeBase:
        knowledge_base = KnowledgeBase(
            id=str(uuid4()),
            name=name,
            created_at=datetime.now(UTC),
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO knowledge_bases (id, name, created_at) VALUES (?, ?, ?)",
                    (
                        knowledge_base.id,
                        knowledge_base.name,
                        knowledge_base.created_at.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateKnowledgeBaseError(f"知识库名称已存在：{name}") from exc
        return knowledge_base

    def list_knowledge_bases(self) -> list[KnowledgeBase]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, created_at FROM knowledge_bases "
                "ORDER BY created_at, id"
            ).fetchall()
        return [
            KnowledgeBase(
                id=row["id"],
                name=row["name"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def delete_knowledge_base(self, kb_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))

    def add_document(self, document: Document) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO documents (
                        id, kb_id, filename, content_hash, file_path,
                        chunk_count, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.kb_id,
                        document.filename,
                        document.content_hash,
                        str(document.file_path),
                        document.chunk_count,
                        document.status.value,
                        document.created_at.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateDocumentError(
                f"知识库 {document.kb_id} 中已存在相同内容"
            ) from exc

    def list_documents(self, kb_id: str) -> list[Document]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, kb_id, filename, content_hash, file_path,
                       chunk_count, status, created_at
                FROM documents
                WHERE kb_id = ?
                ORDER BY created_at, id
                """,
                (kb_id,),
            ).fetchall()
        return [self._document_from_row(row) for row in rows]

    def delete_document(self, kb_id: str, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM documents WHERE kb_id = ? AND id = ?",
                (kb_id, document_id),
            )

    def save_document_file(
        self,
        kb_id: str,
        document_id: str,
        content: str,
    ) -> Path:
        path = self.uploads_path / kb_id / f"{document_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def delete_document_file(self, kb_id: str, document_id: str) -> None:
        path = self.uploads_path / kb_id / f"{document_id}.md"
        path.unlink(missing_ok=True)

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            kb_id=row["kb_id"],
            filename=row["filename"],
            content_hash=row["content_hash"],
            file_path=Path(row["file_path"]),
            chunk_count=row["chunk_count"],
            status=DocumentStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
