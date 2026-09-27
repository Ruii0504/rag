from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from rag_customer_service.agent import CustomerServiceAgent
from rag_customer_service.config import Settings
from rag_customer_service.ingestion import DocumentIngestor
from rag_customer_service.knowledge_base import KnowledgeBaseManager
from rag_customer_service.qwen import QwenClient
from rag_customer_service.rerank import QwenReranker
from rag_customer_service.retrieval import Retriever
from rag_customer_service.storage import Storage
from rag_customer_service.vector_store import ChromaVectorStore


@dataclass(frozen=True, slots=True)
class AppContainer:
    knowledge_bases: KnowledgeBaseManager
    agent: CustomerServiceAgent


def build_container(
    settings: Settings,
    qwen_factory: Callable[[Settings], QwenClient] | None = None,
) -> AppContainer:
    storage = Storage(settings.sqlite_path, settings.uploads_path)
    vector_store = ChromaVectorStore(
        settings.chroma_path,
        settings.qwen_embedding_model,
        settings.qwen_embedding_dimension,
    )
    qwen = (qwen_factory or QwenClient)(settings)
    ingestor = DocumentIngestor(settings.chunk_size, settings.chunk_overlap)
    knowledge_bases = KnowledgeBaseManager(
        storage,
        ingestor,
        qwen,
        vector_store,
    )
    retriever = Retriever(
        qwen,
        vector_store,
        settings.retrieval_top_k,
        settings.retrieval_score_threshold,
        reranker=QwenReranker(
            settings.qwen_api_key,
            settings.qwen_rerank_url,
            settings.qwen_rerank_model,
        ),
    )
    agent = CustomerServiceAgent(qwen, retriever)
    return AppContainer(knowledge_bases=knowledge_bases, agent=agent)


@cache
def get_container(base_dir: Path | None = None) -> AppContainer:
    settings = Settings.from_env(base_dir)
    return build_container(settings)
