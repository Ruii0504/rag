import re
from collections.abc import Sequence

from rag_customer_service.models import Evidence, RetrievalResult, VectorMatch
from rag_customer_service.qwen import QwenClient
from rag_customer_service.rerank import QwenReranker
from rag_customer_service.vector_store import ChromaVectorStore


class Retriever:
    _candidate_pool_multiplier = 20

    def __init__(
        self,
        qwen: QwenClient,
        vector_store: ChromaVectorStore,
        top_k: int,
        score_threshold: float,
        *,
        reranker: QwenReranker | None = None,
    ) -> None:
        self._qwen = qwen
        self._vector_store = vector_store
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._reranker = reranker

    def retrieve(self, query: str, kb_ids: Sequence[str]) -> RetrievalResult:
        if not kb_ids:
            return RetrievalResult(
                query=query,
                evidences=(),
                has_sufficient_evidence=False,
                reason="未选择知识库",
            )

        query_embedding = self._qwen.embed_query(query)
        matches = self._vector_store.search(
            query_embedding,
            kb_ids,
            top_k=self._top_k * self._candidate_pool_multiplier,
        )
        if self._reranker is not None:
            documents = [
                "\n".join((match.chunk.document_name, " / ".join(match.chunk.heading_path), match.chunk.content))
                for match in matches
            ]
            ranked_results = self._reranker.rerank(query, documents, self._top_k)
            evidences = tuple(
                self._evidence(reference_id, matches[item.index], item.score)
                for reference_id, item in enumerate(ranked_results, start=1)
            )
            return RetrievalResult(
                query=query,
                evidences=evidences,
                has_sufficient_evidence=bool(evidences),
                reason="" if evidences else "未检索到相关资料",
            )

        # 未注入重排器的调用保持旧行为；生产装配始终注入，不在故障时降级。
        ranked = sorted(
            matches,
            key=lambda match: (
                self._lexical_relevance(query, match),
                self._similarity(match),
            ),
            reverse=True,
        )
        accepted = [
            match
            for match in ranked
            if self._similarity(match) >= self._score_threshold
        ][: self._top_k]
        evidences = tuple(
            self._evidence(reference_id, match)
            for reference_id, match in enumerate(accepted, start=1)
        )

        return RetrievalResult(
            query=query,
            evidences=evidences,
            has_sufficient_evidence=bool(evidences),
            reason=(
                "" if evidences else "未检索到达到相似度阈值的证据"
            ),
        )

    @staticmethod
    def _similarity(match: VectorMatch) -> float:
        return 1.0 / (1.0 + match.distance)

    @classmethod
    def _lexical_relevance(cls, query: str, match: VectorMatch) -> float:
        query_fragments = cls._character_bigrams(query)
        if not query_fragments:
            return 0.0
        candidate = " ".join((*match.chunk.heading_path, match.chunk.content))
        candidate_fragments = cls._character_bigrams(candidate)
        return len(query_fragments & candidate_fragments) / len(query_fragments)

    @staticmethod
    def _character_bigrams(value: str) -> set[str]:
        normalized = "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]+", value.casefold()))
        if len(normalized) < 2:
            return {normalized} if normalized else set()
        return {
            normalized[index : index + 2]
            for index in range(len(normalized) - 1)
        }

    @classmethod
    def _evidence(cls, reference_id: int, match: VectorMatch, rerank_score: float | None = None) -> Evidence:
        chunk = match.chunk
        return Evidence(
            reference_id=reference_id,
            kb_id=chunk.kb_id,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            heading_path=chunk.heading_path,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            distance=match.distance,
            similarity=cls._similarity(match),
            rerank_score=rerank_score,
        )
