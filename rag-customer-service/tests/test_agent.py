import json

from rag_customer_service.agent import CustomerServiceAgent
from rag_customer_service.models import (
    Evidence,
    RetrievalResult,
    TraceEventType,
)


def sufficient_result(query="如何恢复出厂？"):
    return RetrievalResult(
        query=query,
        evidences=(
            Evidence(
                reference_id=1,
                kb_id="kb-1",
                document_id="doc-1",
                document_name="恢复出厂.md",
                heading_path=("故障排查", "恢复出厂"),
                chunk_index=0,
                content="长按 Reset 键 10 秒。",
                distance=0.1,
                similarity=0.91,
            ),
        ),
        has_sufficient_evidence=True,
    )


def insufficient_result(query="问题", reason="证据不足"):
    return RetrievalResult(
        query=query,
        evidences=(),
        has_sufficient_evidence=False,
        reason=reason,
    )


class FakeQwenClient:
    def __init__(self, rewritten="恢复出厂会清空设置吗？", chunks=("会", "。[1]")):
        self.rewritten = rewritten
        self.chunks = chunks
        self.chat_calls = []
        self.stream_calls = []

    def chat(self, messages):
        self.chat_calls.append(messages)
        system = messages[0]["content"]
        user = messages[1]["content"]
        if "standalone_question" in system:
            current_question = user.rsplit("当前问题：", maxsplit=1)[-1]
            standalone = current_question if "对话历史：[]" in user else self.rewritten
            return json.dumps({
                "standalone_question": standalone,
                "subquestions": [standalone],
            }, ensure_ascii=False)
        if "证据判定器" in system:
            return json.dumps({
                "decisions": [{
                    "subquestion_id": "q1",
                    "answerable": True,
                    "evidence_ids": [1],
                    "missing_information": "",
                    "reason": "证据直接支持",
                }],
            }, ensure_ascii=False)
        if "产品客服" in system:
            return json.dumps({
                "answers": [{
                    "subquestion_id": "q1",
                    "answer": "".join(self.chunks),
                }],
            }, ensure_ascii=False)
        if "答案证据校验器" in system:
            return json.dumps({
                "validations": [{
                    "subquestion_id": "q1",
                    "supported": True,
                    "complete": True,
                    "reason": "证据直接支持",
                }],
            }, ensure_ascii=False)
        raise AssertionError("发生了计划外的模型调用")

    def stream_chat(self, messages):
        self.stream_calls.append(messages)
        yield from self.chunks


class FakeRetriever:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def retrieve(self, query, kb_ids):
        self.calls.append((query, list(kb_ids)))
        return self.result


def test_insufficient_evidence_refuses_without_answer_generation():
    qwen = FakeQwenClient()
    retriever = FakeRetriever(insufficient_result(reason="未选择知识库"))
    agent = CustomerServiceAgent(qwen, retriever)

    events = list(agent.stream("路由器支持什么？", [], []))

    assert retriever.calls == [("路由器支持什么？", [])]
    assert len(qwen.chat_calls) == 1
    assert qwen.stream_calls == []
    assert [(event.type, event.stage) for event in events] == [
        (TraceEventType.NODE_STATUS, "rewrite"),
        (TraceEventType.NODE_STATUS, "rewrite"),
        (TraceEventType.TOOL_STATUS, "retrieval"),
        (TraceEventType.TOOL_STATUS, "retrieval"),
        (TraceEventType.NODE_STATUS, "judge"),
        (TraceEventType.EVIDENCE, "judge"),
        (TraceEventType.ANSWER_DELTA, "refuse"),
        (TraceEventType.COMPLETED, "agent"),
    ]
    assert events[5].message == "语义判定完成：0/1 项可回答"
    assert "未找到相关资料" in events[6].message


def test_sufficient_evidence_streams_answer_with_numbered_source_prompt():
    qwen = FakeQwenClient(chunks=("恢复出厂", "会清空设置[1]"))
    retriever = FakeRetriever(sufficient_result())
    agent = CustomerServiceAgent(qwen, retriever)

    events = list(agent.stream("如何恢复出厂？", [], ["kb-1"]))

    assert retriever.calls == [("如何恢复出厂？", ["kb-1"])]
    assert [
        event.message
        for event in events
        if event.type is TraceEventType.ANSWER_DELTA
    ] == ["**如何恢复出厂？**\n恢复出厂会清空设置[1]"]
    generation_call = next(
        call for call in qwen.chat_calls if "产品客服" in call[0]["content"]
    )
    prompt = "\n".join(message["content"] for message in generation_call)
    assert "[1]" in prompt
    assert "恢复出厂.md" in prompt
    assert "长按 Reset 键 10 秒" in prompt
    assert "只能使用对应编号证据" in prompt
    assert events[-1].type is TraceEventType.COMPLETED


def test_follow_up_rewrite_contains_history_and_drives_retrieval():
    history = [
        {"role": "user", "content": "如何恢复出厂？"},
        {"role": "assistant", "content": "长按 Reset 键。"},
    ]
    qwen = FakeQwenClient(rewritten="恢复出厂会清空设置吗？")
    retriever = FakeRetriever(insufficient_result())
    agent = CustomerServiceAgent(qwen, retriever)

    events = list(agent.stream("这样会清空设置吗？", history, ["kb-1"]))

    rewrite_prompt = "\n".join(message["content"] for message in qwen.chat_calls[0])
    assert "如何恢复出厂" in rewrite_prompt
    assert "这样会清空设置吗" in rewrite_prompt
    assert retriever.calls == [("恢复出厂会清空设置吗？", ["kb-1"])]
    assert events[1].payload["query"] == "恢复出厂会清空设置吗？"


def test_independent_question_without_history_keeps_original_query():
    qwen = FakeQwenClient()
    retriever = FakeRetriever(insufficient_result())

    list(CustomerServiceAgent(qwen, retriever).stream("什么是 Mesh？", [], ["kb-1"]))

    assert len(qwen.chat_calls) == 1
    assert retriever.calls == [("什么是 Mesh？", ["kb-1"])]


def test_error_event_is_last_and_does_not_expose_secret_or_prompt():
    class FailingRetriever(FakeRetriever):
        def retrieve(self, query, kb_ids):
            raise RuntimeError("sk-secret 完整提示词：不可泄露")

    events = list(
        CustomerServiceAgent(
            FakeQwenClient(),
            FailingRetriever(insufficient_result()),
        ).stream("用户问题", [], ["kb-1"])
    )

    assert events[-1].type is TraceEventType.ERROR
    serialized = " ".join(event.message + str(event.payload) for event in events)
    assert "sk-secret" not in serialized
    assert "完整提示词" not in serialized
    assert all(event.type is not TraceEventType.COMPLETED for event in events)
