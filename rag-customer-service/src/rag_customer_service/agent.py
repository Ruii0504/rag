from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from rag_customer_service.models import (
    AnswerValidationStatus,
    EvidenceDecision,
    EvidenceSupportStatus,
    QuestionDecomposition,
    RetrievalResult,
    TraceEvent,
    TraceEventType,
)
from rag_customer_service.answering import EvidenceAnswering
from rag_customer_service.qwen import QwenClient
from rag_customer_service.rerank import RerankError
from rag_customer_service.retrieval import Retriever


class AgentState(TypedDict, total=False):
    question: str
    history: Sequence[Mapping[str, str]]
    kb_ids: Sequence[str]
    decomposition: QuestionDecomposition
    retrievals: Mapping[str, RetrievalResult]
    decisions: tuple[EvidenceDecision, ...]


class CustomerServiceAgent:
    _refusal = "抱歉，我在所选知识库中没有找到足够证据来回答这个问题。"

    def __init__(self, qwen: QwenClient, retriever: Retriever) -> None:
        self._retriever = retriever
        self._answering = EvidenceAnswering(qwen)
        self._graph = self._build_graph()

    def stream(
        self,
        question: str,
        history: Sequence[Mapping[str, str]],
        kb_ids: Sequence[str],
    ) -> Iterator[TraceEvent]:
        state: AgentState = {
            "question": question,
            "history": tuple(history),
            "kb_ids": tuple(kb_ids),
        }
        try:
            yield from self._graph.stream(state, stream_mode="custom")
            yield TraceEvent(
                type=TraceEventType.COMPLETED,
                stage="agent",
                message="处理完成",
            )
        except RerankError as error:
            yield TraceEvent(
                type=TraceEventType.ERROR,
                stage="rerank",
                message=str(error),
            )
        except Exception:
            yield TraceEvent(
                type=TraceEventType.ERROR,
                stage="agent",
                message="客服处理失败，请稍后重试",
            )

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("retrieval", self._retrieve)
        graph.add_node("judge", self._judge)
        graph.add_node("answer", self._answer)
        graph.add_node("refuse", self._refuse)
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "retrieval")
        graph.add_edge("retrieval", "judge")
        graph.add_conditional_edges(
            "judge",
            self._route,
            {"answer": "answer", "refuse": "refuse"},
        )
        graph.add_edge("answer", END)
        graph.add_edge("refuse", END)
        return graph.compile()

    def _rewrite(self, state: AgentState) -> dict[str, QuestionDecomposition]:
        writer = get_stream_writer()
        writer(
            TraceEvent(
                type=TraceEventType.NODE_STATUS,
                stage="rewrite",
                message="问题改写与拆分开始",
            )
        )
        decomposition = self._answering.decompose(
            state["question"],
            state["history"],
        )

        writer(
            TraceEvent(
                type=TraceEventType.NODE_STATUS,
                stage="rewrite",
                message=f"问题拆分完成：{len(decomposition.subquestions)} 项",
                payload={
                    "query": decomposition.standalone_question,
                    "subquestions": [
                        {"id": item.id, "text": item.text}
                        for item in decomposition.subquestions
                    ],
                    "truncated": decomposition.truncated,
                },
            )
        )
        return {"decomposition": decomposition}

    def _retrieve(
        self,
        state: AgentState,
    ) -> dict[str, Mapping[str, RetrievalResult]]:
        writer = get_stream_writer()
        subquestions = state["decomposition"].subquestions
        writer(
            TraceEvent(
                type=TraceEventType.TOOL_STATUS,
                stage="retrieval",
                message=f"开始检索 {len(subquestions)} 个子问题",
                payload={
                    "kb_ids": list(state["kb_ids"]),
                    "subquestion_count": len(subquestions),
                },
            )
        )
        retrievals: dict[str, RetrievalResult] = {}
        next_reference_id = 1
        for subquestion in subquestions:
            result = self._retriever.retrieve(subquestion.text, state["kb_ids"])
            evidences = []
            for evidence in result.evidences:
                evidences.append(replace(
                    evidence,
                    reference_id=next_reference_id,
                    subquestion_id=subquestion.id,
                    subquestion=subquestion.text,
                    support_status=EvidenceSupportStatus.RELATED,
                ))
                next_reference_id += 1
            retrievals[subquestion.id] = replace(
                result,
                evidences=tuple(evidences),
            )
        writer(
            TraceEvent(
                type=TraceEventType.TOOL_STATUS,
                stage="retrieval",
                message=f"候选检索完成：{next_reference_id - 1} 条",
                payload={"count": next_reference_id - 1},
            )
        )
        return {"retrievals": retrievals}

    def _judge(self, state: AgentState) -> dict:
        writer = get_stream_writer()
        writer(
            TraceEvent(
                type=TraceEventType.NODE_STATUS,
                stage="judge",
                message="语义证据判定开始",
            )
        )
        subquestions = state["decomposition"].subquestions
        decisions = self._answering.judge(subquestions, state["retrievals"])
        supporting_ids = {
            evidence_id
            for decision in decisions
            if decision.answerable
            for evidence_id in decision.evidence_ids
        }
        judged_retrievals = {
            subquestion_id: replace(
                result,
                evidences=tuple(
                    replace(
                        evidence,
                        support_status=(
                            EvidenceSupportStatus.SUPPORTING
                            if evidence.reference_id in supporting_ids
                            else EvidenceSupportStatus.RELATED
                        ),
                    )
                    for evidence in result.evidences
                ),
            )
            for subquestion_id, result in state["retrievals"].items()
        }
        all_evidences = tuple(
            evidence
            for subquestion in subquestions
            for evidence in judged_retrievals[subquestion.id].evidences
        )
        answerable_count = sum(decision.answerable for decision in decisions)
        aggregate = RetrievalResult(
            query=state["decomposition"].standalone_question,
            evidences=all_evidences,
            has_sufficient_evidence=answerable_count > 0,
            reason=(
                ""
                if answerable_count
                else (
                    "找到相关资料，但缺少直接答案"
                    if all_evidences
                    else "未找到相关资料"
                )
            ),
        )
        writer(
            TraceEvent(
                type=TraceEventType.EVIDENCE,
                stage="judge",
                message=f"语义判定完成：{answerable_count}/{len(decisions)} 项可回答",
                payload={
                    "result": aggregate,
                    "count": len(all_evidences),
                    "decisions": decisions,
                },
            )
        )
        return {"retrievals": judged_retrievals, "decisions": decisions}

    @staticmethod
    def _route(state: AgentState) -> str:
        return (
            "answer"
            if any(decision.answerable for decision in state["decisions"])
            else "refuse"
        )

    def _answer(self, state: AgentState) -> dict:
        writer = get_stream_writer()
        subquestions = state["decomposition"].subquestions
        writer(TraceEvent(
            type=TraceEventType.NODE_STATUS,
            stage="generate",
            message="答案生成开始",
        ))
        parts = self._answering.generate(
            subquestions,
            state["decisions"],
            state["retrievals"],
        )
        writer(TraceEvent(
            type=TraceEventType.NODE_STATUS,
            stage="validate",
            message="答案证据校验开始",
        ))
        validated = self._answering.validate(
            parts,
            state["decisions"],
            state["retrievals"],
        )
        failed = [
            part
            for part in validated
            if part.validation_status is AnswerValidationStatus.FAILED
        ]
        if failed:
            writer(TraceEvent(
                type=TraceEventType.NODE_STATUS,
                stage="regenerate",
                message=f"{len(failed)} 项校验失败，重新生成一次",
            ))
            failed_ids = {part.subquestion_id for part in failed}
            retry_decisions = tuple(
                decision
                for decision in state["decisions"]
                if decision.subquestion_id in failed_ids
            )
            feedback = {
                part.subquestion_id: part.validation_reason
                for part in failed
            }
            retry_parts = self._answering.generate(
                subquestions,
                retry_decisions,
                state["retrievals"],
                feedback,
            )
            retry_validated = self._answering.validate(
                retry_parts,
                retry_decisions,
                state["retrievals"],
            )
            retry_by_id = {
                part.subquestion_id: part
                for part in retry_validated
            }
            validated = tuple(
                retry_by_id.get(part.subquestion_id, part)
                if part.subquestion_id in failed_ids
                else part
                for part in validated
            )

        final_answer = self._answering.compose(
            subquestions,
            state["decisions"],
            validated,
            truncated=state["decomposition"].truncated,
        )
        verified_count = sum(
            part.validation_status is AnswerValidationStatus.VERIFIED
            for part in validated
        )
        writer(
            TraceEvent(
                type=TraceEventType.ANSWER_DELTA,
                stage="answer" if verified_count else "refuse",
                message=final_answer,
                payload={"verified_count": verified_count},
            )
        )
        return {}

    def _refuse(self, state: AgentState) -> dict:
        final_answer = self._answering.compose(
            state["decomposition"].subquestions,
            state["decisions"],
            (),
            truncated=state["decomposition"].truncated,
        )
        get_stream_writer()(
            TraceEvent(
                type=TraceEventType.ANSWER_DELTA,
                stage="refuse",
                message=final_answer or self._refusal,
                payload={"reason": "所有子问题均缺少直接证据"},
            )
        )
        return {}
