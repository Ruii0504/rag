from rag_customer_service.answering import EvidenceAnswering
from rag_customer_service.models import (
    AnswerPart,
    AnswerValidationStatus,
    Evidence,
    EvidenceDecision,
    RetrievalResult,
    SubQuestion,
)


class QueueQwenClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.chat_calls = []

    def chat(self, messages):
        self.chat_calls.append(messages)
        return self.responses.pop(0)


def make_evidence(reference_id: int, content: str) -> Evidence:
    return Evidence(
        reference_id=reference_id,
        kb_id="kb-1",
        document_id=f"doc-{reference_id}",
        document_name=f"资料-{reference_id}.md",
        heading_path=("产品资料",),
        chunk_index=0,
        content=content,
        distance=0.2,
        similarity=0.83,
    )


def make_result(query: str, *evidences: Evidence) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        evidences=evidences,
        has_sufficient_evidence=bool(evidences),
    )


def test_decompose_rewrites_and_limits_atomic_questions_to_five():
    qwen = QueueQwenClient([
        """{
          "standalone_question": "AX3000 的价格、无线标准、保修、接口、尺寸和重量是什么？",
          "subquestions": [
            "AX3000 的价格是多少？",
            "AX3000 支持什么无线标准？",
            "AX3000 的保修期多久？",
            "AX3000 有哪些接口？",
            "AX3000 的尺寸是多少？",
            "AX3000 的重量是多少？"
          ]
        }"""
    ])
    answering = EvidenceAnswering(qwen)

    result = answering.decompose(
        "它的价格、无线标准、保修、接口、尺寸和重量呢？",
        [{"role": "user", "content": "请介绍 AX3000"}],
    )

    assert result.standalone_question.startswith("AX3000")
    assert [item.id for item in result.subquestions] == ["q1", "q2", "q3", "q4", "q5"]
    assert result.subquestions[0].text == "AX3000 的价格是多少？"
    assert result.subquestions[-1].text == "AX3000 的尺寸是多少？"
    assert result.truncated is True
    assert "对话历史" in qwen.chat_calls[0][1]["content"]


def test_decompose_keeps_dimensions_as_one_user_intent():
    answering = EvidenceAnswering(QueueQwenClient([
        """{
          "standalone_question": "星云智联 AX6000 智能路由器的尺寸是多少？",
          "subquestions": [
            "星云智联 AX6000 智能路由器的长度是多少？",
            "星云智联 AX6000 智能路由器的宽度是多少？",
            "星云智联 AX6000 智能路由器的高度是多少？"
          ]
        }"""
    ]))

    result = answering.decompose("星云智联 AX6000 智能路由器的尺寸是多少？", [])

    assert result.subquestions == (
        SubQuestion("q1", "星云智联 AX6000 智能路由器的尺寸是多少？"),
    )


def test_decompose_keeps_single_usb_intent_and_does_not_invent_a_computer():
    answering = EvidenceAnswering(QueueQwenClient([
        """{
          "standalone_question": "计算机的 USB 接口可以连接打印机吗？",
          "subquestions": [
            "打印机是否支持通过 USB 接口连接？",
            "计算机的 USB 接口是否能识别打印机？"
          ]
        }"""
    ]))

    result = answering.decompose("USB 口能接打印机吗？", [])

    assert result.standalone_question == "USB 口能接打印机吗？"
    assert result.subquestions == (SubQuestion("q1", "USB 口能接打印机吗？"),)


def test_decompose_splits_explicit_price_and_dimensions_into_two_intents():
    answering = EvidenceAnswering(QueueQwenClient([
        """{
          "standalone_question": "星云智联 AX6000 的价格和尺寸分别是多少？",
          "subquestions": [
            "星云智联 AX6000 的价格是多少？",
            "星云智联 AX6000 的尺寸是多少？"
          ]
        }"""
    ]))

    result = answering.decompose(
        "该产品的价格和尺寸如何？",
        [{"role": "user", "content": "请介绍星云智联 AX6000。"}],
    )

    assert result.subquestions == (
        SubQuestion("q1", "星云智联 AX6000 的价格是多少？"),
        SubQuestion("q2", "星云智联 AX6000 的尺寸是多少？"),
    )


def test_decompose_invalid_json_falls_back_to_current_question():
    answering = EvidenceAnswering(QueueQwenClient(["这不是 JSON"]))

    result = answering.decompose("AX3000 多少钱？", [])

    assert result.standalone_question == "AX3000 多少钱？"
    assert result.subquestions == (SubQuestion("q1", "AX3000 多少钱？"),)
    assert result.truncated is False


def test_judge_selects_direct_evidence_for_each_subquestion():
    qwen = QueueQwenClient([
        """{
          "decisions": [{
              "subquestion_id": "q1",
              "answerable": false,
              "evidence_ids": [],
              "missing_information": "价格",
              "reason": "候选资料没有金额"
          }]
        }""",
        """{
          "decisions": [{
              "subquestion_id": "q2",
              "answerable": true,
              "evidence_ids": [2],
              "missing_information": "",
              "reason": "资料直接说明无线标准"
          }]
        }"""
    ])
    answering = EvidenceAnswering(qwen)
    subquestions = (
        SubQuestion("q1", "AX3000 多少钱？"),
        SubQuestion("q2", "AX3000 支持 Wi-Fi 6 吗？"),
    )
    retrievals = {
        "q1": make_result("AX3000 多少钱？", make_evidence(1, "AX3000 支持双频无线。")),
        "q2": make_result("AX3000 支持 Wi-Fi 6 吗？", make_evidence(2, "AX3000 支持 Wi-Fi 6。")),
    }

    decisions = answering.judge(subquestions, retrievals)

    assert decisions[0].answerable is False
    assert decisions[0].missing_information == "价格"
    assert decisions[1].answerable is True
    assert decisions[1].evidence_ids == (2,)


def test_judge_isolates_subquestions_and_accepts_direct_negative_evidence():
    qwen = QueueQwenClient([
        """{
          "decisions": [{
            "subquestion_id": "q1",
            "answerable": true,
            "evidence_ids": [1],
            "missing_information": "",
            "reason": "尺寸证据直接回答问题"
          }]
        }""",
        """{
          "decisions": [{
            "subquestion_id": "q2",
            "answerable": true,
            "evidence_ids": [2],
            "missing_information": "",
            "reason": "否定证据直接说明不支持打印机"
          }]
        }""",
    ])
    answering = EvidenceAnswering(qwen)
    subquestions = (
        SubQuestion("q1", "星云智联 AX6000 的尺寸是多少？"),
        SubQuestion("q2", "星云智联 AX6000 的 USB 口能接打印机吗？"),
    )
    retrievals = {
        "q1": make_result(
            subquestions[0].text,
            make_evidence(1, "整机长 250 毫米、宽 150 毫米、高 200 毫米。"),
        ),
        "q2": make_result(
            subquestions[1].text,
            make_evidence(2, "USB 3.0 口仅支持存储共享，不支持连接打印机。"),
        ),
    }

    decisions = answering.judge(subquestions, retrievals)

    assert [decision.answerable for decision in decisions] == [True, True]
    assert [decision.evidence_ids for decision in decisions] == [(1,), (2,)]
    assert len(qwen.chat_calls) == 2
    assert '"subquestion_id": "q2"' not in qwen.chat_calls[0][1]["content"]
    assert '"subquestion_id": "q1"' not in qwen.chat_calls[1][1]["content"]


def test_judge_accepts_complete_json_wrapped_in_markdown_code_fence():
    qwen = QueueQwenClient([
        """```json
        {
          "decisions": [{
            "subquestion_id": "q1",
            "answerable": true,
            "evidence_ids": [4],
            "missing_information": "",
            "reason": "尺寸证据直接回答问题"
          }]
        }
        ```""",
    ])
    answering = EvidenceAnswering(qwen)
    subquestion = SubQuestion("q1", "星云智联 AX6000 的尺寸是多少？")
    retrievals = {
        "q1": make_result(
            subquestion.text,
            make_evidence(4, "整机长 250 毫米、宽 150 毫米、高 200 毫米。"),
        ),
    }

    decision = answering.judge((subquestion,), retrievals)[0]

    assert decision.answerable is True
    assert decision.evidence_ids == (4,)


def test_judge_rejects_evidence_id_owned_by_another_subquestion():
    qwen = QueueQwenClient([
        """{
          "decisions": [{
            "subquestion_id": "q1",
            "answerable": true,
            "evidence_ids": [2],
            "missing_information": "",
            "reason": "错误地引用了其他问题的证据"
          }]
        }"""
    ])
    answering = EvidenceAnswering(qwen)
    subquestions = (SubQuestion("q1", "AX3000 多少钱？"),)
    retrievals = {
        "q1": make_result("AX3000 多少钱？", make_evidence(1, "没有价格。")),
    }

    decision = answering.judge(subquestions, retrievals)[0]

    assert decision.answerable is False
    assert decision.evidence_ids == ()
    assert "证据编号" in decision.reason


def test_judge_invalid_json_refuses_every_subquestion():
    answering = EvidenceAnswering(QueueQwenClient(["无法解析", "仍然无法解析"]))
    subquestions = (
        SubQuestion("q1", "价格？"),
        SubQuestion("q2", "保修？"),
    )
    retrievals = {
        "q1": make_result("价格？", make_evidence(1, "相关资料")),
        "q2": make_result("保修？", make_evidence(2, "相关资料")),
    }

    decisions = answering.judge(subquestions, retrievals)

    assert [decision.answerable for decision in decisions] == [False, False]
    assert all("判定输出无效" in decision.reason for decision in decisions)


def test_generate_sends_only_answerable_subquestions_and_allowed_evidence():
    qwen = QueueQwenClient([
        """{
          "answers": [{
            "subquestion_id": "q2",
            "answer": "AX3000 支持 Wi-Fi 6。[2]"
          }]
        }"""
    ])
    answering = EvidenceAnswering(qwen)
    subquestions = (
        SubQuestion("q1", "AX3000 多少钱？"),
        SubQuestion("q2", "AX3000 支持 Wi-Fi 6 吗？"),
    )
    decisions = (
        EvidenceDecision("q1", False, missing_information="价格"),
        EvidenceDecision("q2", True, (2,)),
    )
    retrievals = {
        "q1": make_result("价格？", make_evidence(1, "产品功能介绍")),
        "q2": make_result("无线标准？", make_evidence(2, "AX3000 支持 Wi-Fi 6。")),
    }

    parts = answering.generate(subquestions, decisions, retrievals)

    assert parts == (
        AnswerPart(
            subquestion_id="q2",
            question="AX3000 支持 Wi-Fi 6 吗？",
            answer="AX3000 支持 Wi-Fi 6。[2]",
            evidence_ids=(2,),
        ),
    )
    prompt = qwen.chat_calls[0][1]["content"]
    assert "AX3000 多少钱" not in prompt
    assert "产品功能介绍" not in prompt
    assert "AX3000 支持 Wi-Fi 6" in prompt


def test_validate_accepts_supported_citations_and_numbers():
    qwen = QueueQwenClient([
        '{"validations":[{"subquestion_id":"q2","supported":true,"complete":true,"reason":"证据直接支持"}]}'
    ])
    answering = EvidenceAnswering(qwen)
    parts = (
        AnswerPart(
            subquestion_id="q2",
            question="支持 Wi-Fi 6 吗？",
            answer="支持 Wi-Fi 6。[2]",
            evidence_ids=(2,),
        ),
    )
    decisions = (EvidenceDecision("q2", True, (2,)),)
    retrievals = {
        "q2": make_result("支持 Wi-Fi 6 吗？", make_evidence(2, "支持 Wi-Fi 6。")),
    }

    validated = answering.validate(parts, decisions, retrievals)

    assert validated[0].validation_status is AnswerValidationStatus.VERIFIED
    assert validated[0].validation_reason == "证据直接支持"


def test_validate_rejects_reference_outside_allowed_evidence_before_model_call():
    qwen = QueueQwenClient([])
    answering = EvidenceAnswering(qwen)
    parts = (
        AnswerPart("q1", "价格？", "价格是 299 元。[9]", (1,)),
    )
    decisions = (EvidenceDecision("q1", True, (1,)),)
    retrievals = {
        "q1": make_result("价格？", make_evidence(1, "价格是 299 元。")),
    }

    validated = answering.validate(parts, decisions, retrievals)

    assert validated[0].validation_status is AnswerValidationStatus.FAILED
    assert "引用编号" in validated[0].validation_reason
    assert qwen.chat_calls == []


def test_validate_rejects_number_not_present_in_allowed_evidence():
    qwen = QueueQwenClient([])
    answering = EvidenceAnswering(qwen)
    parts = (
        AnswerPart("q1", "价格？", "价格是 299 元。[1]", (1,)),
    )
    decisions = (EvidenceDecision("q1", True, (1,)),)
    retrievals = {
        "q1": make_result("价格？", make_evidence(1, "资料只介绍无线功能。")),
    }

    validated = answering.validate(parts, decisions, retrievals)

    assert validated[0].validation_status is AnswerValidationStatus.FAILED
    assert "数字 299" in validated[0].validation_reason
    assert qwen.chat_calls == []


def test_validate_invalid_model_output_fails_closed_without_retryable_status():
    answering = EvidenceAnswering(QueueQwenClient(["不是 JSON"]))
    parts = (
        AnswerPart("q1", "保修多久？", "保修 2 年。[1]", (1,)),
    )
    decisions = (EvidenceDecision("q1", True, (1,)),)
    retrievals = {
        "q1": make_result("保修多久？", make_evidence(1, "保修期为 2 年。")),
    }

    validated = answering.validate(parts, decisions, retrievals)

    assert validated[0].validation_status is AnswerValidationStatus.INVALID
    assert "校验输出无效" in validated[0].validation_reason


def test_compose_keeps_verified_answer_and_locally_refuses_missing_price():
    answering = EvidenceAnswering(QueueQwenClient([]))
    subquestions = (
        SubQuestion("q1", "AX3000 多少钱？"),
        SubQuestion("q2", "AX3000 支持 Wi-Fi 6 吗？"),
    )
    decisions = (
        EvidenceDecision("q1", False, missing_information="价格"),
        EvidenceDecision("q2", True, (2,)),
    )
    parts = (
        AnswerPart(
            "q2",
            "AX3000 支持 Wi-Fi 6 吗？",
            "AX3000 支持 Wi-Fi 6。[2]",
            (2,),
            AnswerValidationStatus.VERIFIED,
        ),
    )

    answer = answering.compose(subquestions, decisions, parts, truncated=True)

    assert "**AX3000 多少钱？**" in answer
    assert "缺少价格信息" in answer
    assert "AX3000 支持 Wi-Fi 6。[2]" in answer
    assert "仅处理前 5 个子问题" in answer
