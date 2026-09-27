import json

from rag_customer_service.agent import CustomerServiceAgent
from rag_customer_service.models import (
    Evidence,
    EvidenceSupportStatus,
    RetrievalResult,
    TraceEventType,
)


def response(payload) -> str:
    return json.dumps(payload, ensure_ascii=False)


def evidence(content: str) -> Evidence:
    return Evidence(
        reference_id=1,
        kb_id="kb-1",
        document_id="doc-1",
        document_name="AX3000.md",
        heading_path=("产品说明",),
        chunk_index=0,
        content=content,
        distance=0.2,
        similarity=0.83,
    )


def retrieval(query: str, content: str | None) -> RetrievalResult:
    evidences = (evidence(content),) if content is not None else ()
    return RetrievalResult(query, evidences, bool(evidences))


class ScriptedQwenClient:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.chat_calls = []
        self.stream_calls = []

    def chat(self, messages):
        self.chat_calls.append(messages)
        if not self.responses:
            raise AssertionError("发生了计划外的模型调用")
        return self.responses.pop(0)

    def stream_chat(self, messages):
        self.stream_calls.append(messages)
        raise AssertionError("验证前不得流式输出模型正文")


class MappingRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, kb_ids):
        self.calls.append((query, list(kb_ids)))
        return self.results[query]


def event_index(events, stage: str) -> int:
    return next(index for index, event in enumerate(events) if event.stage == stage)


def test_partial_answer_keeps_supported_item_and_locally_refuses_price():
    price = "AX3000 多少钱？"
    wifi = "AX3000 支持 Wi-Fi 6 吗？"
    qwen = ScriptedQwenClient(
        response({"standalone_question": f"{price}{wifi}", "subquestions": [price, wifi]}),
        response({"decisions": [{
            "subquestion_id": "q1",
            "answerable": False,
            "evidence_ids": [],
            "missing_information": "价格",
            "reason": "资料没有金额",
        }]}),
        response({"decisions": [{
            "subquestion_id": "q2",
            "answerable": True,
            "evidence_ids": [2],
            "missing_information": "",
            "reason": "资料直接说明无线标准",
        }]}),
        response({"answers": [{
            "subquestion_id": "q2",
            "answer": "AX3000 支持 Wi-Fi 6。[2]",
        }]}),
        response({"validations": [{
            "subquestion_id": "q2",
            "supported": True,
            "complete": True,
            "reason": "证据直接支持",
        }]}),
    )
    retriever = MappingRetriever({
        price: retrieval(price, "AX3000 采用双频无线设计。"),
        wifi: retrieval(wifi, "AX3000 支持 Wi-Fi 6。"),
    })

    events = list(CustomerServiceAgent(qwen, retriever).stream(
        "价格和无线标准？",
        [],
        ["kb-1"],
    ))

    answer_events = [event for event in events if event.type is TraceEventType.ANSWER_DELTA]
    assert len(answer_events) == 1
    assert answer_events[0].stage == "answer"
    assert "缺少价格信息" in answer_events[0].message
    assert "AX3000 支持 Wi-Fi 6。[2]" in answer_events[0].message
    assert event_index(events, "validate") < events.index(answer_events[0])
    evidence_event = next(event for event in events if event.type is TraceEventType.EVIDENCE)
    statuses = [item.support_status for item in evidence_event.payload["result"].evidences]
    assert statuses == [EvidenceSupportStatus.RELATED, EvidenceSupportStatus.SUPPORTING]
    assert qwen.stream_calls == []


def test_all_unanswerable_items_refuse_without_generation_or_validation():
    price = "AX3000 多少钱？"
    qwen = ScriptedQwenClient(
        response({"standalone_question": price, "subquestions": [price]}),
        response({"decisions": [{
            "subquestion_id": "q1",
            "answerable": False,
            "evidence_ids": [],
            "missing_information": "价格",
            "reason": "相关资料没有价格",
        }]}),
    )
    retriever = MappingRetriever({price: retrieval(price, "产品功能介绍")})

    events = list(CustomerServiceAgent(qwen, retriever).stream(price, [], ["kb-1"]))

    answer_event = next(event for event in events if event.type is TraceEventType.ANSWER_DELTA)
    assert answer_event.stage == "refuse"
    assert "缺少价格信息" in answer_event.message
    assert len(qwen.chat_calls) == 2
    assert all(event.stage not in {"generate", "validate"} for event in events)


def test_failed_deterministic_validation_regenerates_once_then_returns_verified_answer():
    price = "AX3000 多少钱？"
    qwen = ScriptedQwenClient(
        response({"standalone_question": price, "subquestions": [price]}),
        response({"decisions": [{
            "subquestion_id": "q1",
            "answerable": True,
            "evidence_ids": [1],
            "missing_information": "",
            "reason": "资料有价格",
        }]}),
        response({"answers": [{"subquestion_id": "q1", "answer": "价格是 299 元。[1]"}]}),
        response({"answers": [{"subquestion_id": "q1", "answer": "价格是 199 元。[1]"}]}),
        response({"validations": [{
            "subquestion_id": "q1",
            "supported": True,
            "complete": True,
            "reason": "金额和证据一致",
        }]}),
    )
    retriever = MappingRetriever({price: retrieval(price, "建议零售价为 199 元。")})

    events = list(CustomerServiceAgent(qwen, retriever).stream(price, [], ["kb-1"]))

    answer_events = [event for event in events if event.type is TraceEventType.ANSWER_DELTA]
    assert len(answer_events) == 1
    assert "199 元。[1]" in answer_events[0].message
    assert "299" not in answer_events[0].message
    assert sum(event.stage == "regenerate" for event in events) == 1
    assert "数字 299" in qwen.chat_calls[3][1]["content"]


def test_second_validation_failure_downgrades_only_item_without_leaking_drafts():
    price = "AX3000 多少钱？"
    qwen = ScriptedQwenClient(
        response({"standalone_question": price, "subquestions": [price]}),
        response({"decisions": [{
            "subquestion_id": "q1",
            "answerable": True,
            "evidence_ids": [1],
            "missing_information": "",
            "reason": "资料有价格",
        }]}),
        response({"answers": [{"subquestion_id": "q1", "answer": "价格是 299 元。[1]"}]}),
        response({"answers": [{"subquestion_id": "q1", "answer": "价格是 399 元。[1]"}]}),
    )
    retriever = MappingRetriever({price: retrieval(price, "建议零售价为 199 元。")})

    events = list(CustomerServiceAgent(qwen, retriever).stream(price, [], ["kb-1"]))

    answer_events = [event for event in events if event.type is TraceEventType.ANSWER_DELTA]
    assert len(answer_events) == 1
    assert answer_events[0].stage == "refuse"
    assert "无法通过证据校验" in answer_events[0].message
    assert "299" not in answer_events[0].message
    assert "399" not in answer_events[0].message
    assert sum(event.stage == "regenerate" for event in events) == 1


def test_invalid_validator_output_fails_closed_without_regeneration():
    warranty = "保修多久？"
    qwen = ScriptedQwenClient(
        response({"standalone_question": warranty, "subquestions": [warranty]}),
        response({"decisions": [{
            "subquestion_id": "q1",
            "answerable": True,
            "evidence_ids": [1],
            "missing_information": "",
            "reason": "资料有保修期限",
        }]}),
        response({"answers": [{"subquestion_id": "q1", "answer": "保修 2 年。[1]"}]}),
        "不是 JSON",
    )
    retriever = MappingRetriever({warranty: retrieval(warranty, "保修期为 2 年。")})

    events = list(CustomerServiceAgent(qwen, retriever).stream(warranty, [], ["kb-1"]))

    answer_event = next(event for event in events if event.type is TraceEventType.ANSWER_DELTA)
    assert answer_event.stage == "refuse"
    assert "无法通过证据校验" in answer_event.message
    assert all(event.stage != "regenerate" for event in events)
