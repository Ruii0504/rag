import json
from dataclasses import replace

import pytest

from rag_customer_service.answering import EvidenceAnswering
from rag_customer_service.models import AnswerPart, AnswerValidationStatus, EvidenceDecision, SubQuestion
from test_answering import QueueQwenClient, make_evidence, make_result


@pytest.mark.parametrize("answer", [
    "AX6000 的含天线尺寸约长 250 毫米、宽 150 毫米、高 200 毫米。[1]",
    "1. 含天线长约 250 毫米。[1]\n2. 宽约 150 毫米、高约 200 毫米。[1]",
])
def test_supported_model_in_heading_and_list_markers_are_not_fact_numbers(answer):
    evidence = replace(make_evidence(1, "含天线尺寸约长 250 毫米、宽 150 毫米、高 200 毫米。"),
                       heading_path=("星云智联 AX6000", "尺寸"))
    model = QueueQwenClient(['{"validations":[{"subquestion_id":"q1","supported":true,"complete":true,"reason":"支持"}]}'])
    validated = EvidenceAnswering(model).validate(
        (AnswerPart("q1", "尺寸？", answer, (1,)),),
        (EvidenceDecision("q1", True, (1,)),), {"q1": make_result("尺寸？", evidence)},
    )
    assert validated[0].validation_status is AnswerValidationStatus.VERIFIED


@pytest.mark.parametrize("answer,question", [
    ("电压是 24 V。[1]", "我认为是24V，对吗？"),
    ("电压是 2 V。[1]", "电压？"),
    ("电压是 6000 V。[1]", "AX6000电压？"),
])
def test_question_model_digits_and_numeric_substrings_do_not_authorize_wrong_voltage(answer, question):
    evidence = replace(make_evidence(1, "额定电压为 12 V。"), heading_path=("AX6000",))
    result = EvidenceAnswering(QueueQwenClient([])).validate(
        (AnswerPart("q1", question, answer, (1,)),), (EvidenceDecision("q1", True, (1,)),),
        {"q1": make_result(question, evidence)},
    )
    assert result[0].validation_status is AnswerValidationStatus.FAILED


def test_list_numbers_do_not_block_non_numeric_instructions():
    model = QueueQwenClient(['{"validations":[{"subquestion_id":"q1","supported":true,"complete":true,"reason":"支持"}]}'])
    result = EvidenceAnswering(model).validate(
        (AnswerPart("q1", "信号差？", "1. 调整摆放位置。[1]\n2. 调整天线角度。[1]", (1,)),),
        (EvidenceDecision("q1", True, (1,)),),
        {"q1": make_result("信号差？", make_evidence(1, "调整摆放位置和天线角度。"))},
    )
    assert result[0].validation_status is AnswerValidationStatus.VERIFIED


def test_partial_evidence_plan_keeps_known_points_and_unknown_boundary_in_generation():
    model = QueueQwenClient([
        json.dumps({"decisions": [{"subquestion_id": "q1", "answerable": True, "evidence_ids": [1],
            "supported_points": ["可用网页后台管理", "地址为 http://192.168.10.1"],
            "missing_information": "App 名称", "reason": "只能确认网页方式"}]}),
        json.dumps({"answers": [{"subquestion_id": "q1", "answer": "可以使用网页后台。[1] App 名称无法确认。"}]}),
    ])
    answering = EvidenceAnswering(model)
    questions = (SubQuestion("q1", "用哪个App管理？没有App能怎么管理？"),)
    retrievals = {"q1": make_result(questions[0].text, make_evidence(1, "主要通过网页 http://192.168.10.1 管理，App 支持未确认。"))}
    decisions = answering.judge(questions, retrievals)
    assert decisions[0].supported_points == ("可用网页后台管理", "地址为 http://192.168.10.1")
    answering.generate(questions, decisions, retrievals)
    item = json.loads(model.chat_calls[-1][1]["content"])["items"][0]
    assert item["required_points"] == ["可用网页后台管理", "地址为 http://192.168.10.1"]
    assert item["missing_information"] == "App 名称"


@pytest.mark.parametrize("complete,expected", [(False, AnswerValidationStatus.FAILED), (None, AnswerValidationStatus.INVALID)])
def test_factually_supported_but_incomplete_or_unchecked_answer_is_not_verified(complete, expected):
    payload = {"subquestion_id": "q1", "supported": True, "reason": "遗漏含天线范围", "complete": complete}
    model = QueueQwenClient([json.dumps({"validations": [payload]})])
    evidence = make_evidence(1, "含天线尺寸约长 250 毫米、宽 150 毫米、高 200 毫米。")
    result = EvidenceAnswering(model).validate(
        (AnswerPart("q1", "尺寸？", "长 250 毫米。[1]", (1,)),),
        (EvidenceDecision("q1", True, (1,)),), {"q1": make_result("尺寸？", evidence)},
    )
    assert result[0].validation_status is expected


@pytest.mark.parametrize("missing", ["价格", "价格信息", "厂商是否承诺赔偿，及其适用条件。"])
def test_refusal_does_not_append_information_to_an_entire_sentence(missing):
    answer = EvidenceAnswering(QueueQwenClient([])).compose(
        (SubQuestion("q1", "能确认吗？"),),
        (EvidenceDecision("q1", False, missing_information=missing),), (), truncated=False,
    )
    assert "信息信息" not in answer
    assert "缺少厂商是否承诺赔偿" not in answer
    assert "无法确认" in answer
