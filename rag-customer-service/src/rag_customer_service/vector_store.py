import json
from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.errors import NotFoundError

from rag_customer_service.models import Chunk, VectorMatch


class VectorStoreConfigurationError(RuntimeError):
    """向量集合配置与当前模型不一致。"""


class ChromaVectorStore:
    _collection_name = "rag_chunks"

    def __init__(
        self,
        path: Path,
        embedding_model: str,
        embedding_dimension: int,
    ) -> None:
        self._dimension = embedding_dimension
        self._client = chromadb.PersistentClient(path=path)
        expected_metadata = {
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "hnsw:space": "l2",
        }

        try:
            collection = self._client.get_collection(
                self._collection_name,
                embedding_function=None,
            )
        except NotFoundError:
            collection = self._client.create_collection(
                self._collection_name,
                metadata=expected_metadata,
                embedding_function=None,
            )
        else:
            metadata = collection.metadata or {}
            if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                raise VectorStoreConfigurationError(
                    "Embedding 模型或维度已变更，请清空索引并重新向量化"
                )

        self._collection = collection

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("切块数量与向量数量不一致")
        if not chunks:
            return
        self._validate_embeddings(embeddings)

        self._collection.add(
            ids=[chunk.id for chunk in chunks],
            embeddings=[list(embedding) for embedding in embeddings],
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )

    def search(
        self,
        query_embedding: Sequence[float],
        kb_ids: Sequence[str],
        top_k: int,
    ) -> tuple[VectorMatch, ...]:
        if not kb_ids:
            return ()
        self._validate_embeddings([query_embedding])

        where = (
            {"kb_id": kb_ids[0]}
            if len(kb_ids) == 1
            else {"kb_id": {"$in": list(kb_ids)}}
        )
        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = result["ids"][0]
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []

        return tuple(
            VectorMatch(
                chunk=self._chunk(chunk_id, document, metadata),
                distance=float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
                strict=True,
            )
        )

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def delete_knowledge_base(self, kb_id: str) -> None:
        self._collection.delete(where={"kb_id": kb_id})

    def _validate_embeddings(
        self,
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if any(len(embedding) != self._dimension for embedding in embeddings):
            raise VectorStoreConfigurationError(
                "向量维度与当前配置不一致，请重新向量化"
            )

    @staticmethod
    def _metadata(chunk: Chunk) -> dict[str, str | int]:
        return {
            "kb_id": chunk.kb_id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "heading_path": json.dumps(chunk.heading_path, ensure_ascii=False),
            "chunk_index": chunk.chunk_index,
            "content_hash": chunk.content_hash,
        }

    @staticmethod
    def _chunk(
        chunk_id: str,
        document: str,
        metadata: dict[str, object],
    ) -> Chunk:
        return Chunk(
            id=chunk_id,
            kb_id=str(metadata["kb_id"]),
            document_id=str(metadata["document_id"]),
            document_name=str(metadata["document_name"]),
            heading_path=tuple(json.loads(str(metadata["heading_path"]))),
            chunk_index=int(metadata["chunk_index"]),
            content=document,
            content_hash=str(metadata["content_hash"]),
        )
