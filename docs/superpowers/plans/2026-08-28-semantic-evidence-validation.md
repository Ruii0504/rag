# RAG Semantic Evidence Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现按原子子问题检索、语义证据判定、局部回答与拒答、生成后逐项校验和一次安全重生成。

**Architecture:** 保留现有 FastAPI + LangGraph 分层单体，将结构化模型协议和确定性校验集中到新的 `answering.py`，`Retriever` 只负责候选召回，`CustomerServiceAgent` 负责工作流和事件。前端继续消费 SSE，但只有验证后的正文才能作为 `answer_delta` 输出。

**Tech Stack:** Python 3.12、LangGraph、LangChain OpenAI、FastAPI、React、TypeScript、Vitest、pytest。

**Spec:** `docs/superpowers/specs/2026-08-28-semantic-evidence-validation-design.md`

## Global Constraints

- 子问题最多 5 个。
- 每项候选 Top 7，相似度阈值 0.5。
- 相似度仅用于候选召回，不能直接代表证据充分。
- 判定和校验协议解析失败必须 fail-closed。
- 未验证正文不能通过 SSE 发送。
- 明确校验失败最多重生成一次。
- 只修改本需求涉及的代码，不删除测试、不跳过测试、不降低断言。
- 当前工作区已有用户改动；不重置、不清理无关差异、不创建 Git 提交。

---

### Task 1: 更新配置、领域模型和候选检索

**Files:**
- Modify: `rag-customer-service/.env.example`
- Modify: `rag-customer-service/src/rag_customer_service/config.py`
- Modify: `rag-customer-service/src/rag_customer_service/models.py`
- Modify: `rag-customer-service/src/rag_customer_service/retrieval.py`
- Modify: `rag-customer-service/tests/test_config.py`
- Modify: `rag-customer-service/tests/test_retrieval.py`

**Interfaces:**
- Produces: `SubQuestion`、`EvidenceDecision`、`AnswerPart`、带子问题及支持状态的 `Evidence`；`Retriever.retrieve()` 保持候选召回职责。

- [ ] 新增失败测试，证明默认 `RETRIEVAL_TOP_K == 7`、默认阈值为 `0.5`，并证明达到阈值只产生候选而不宣称证据充分。
- [ ] 运行配置和检索测试，确认因旧默认值和旧充分性语义失败。
- [ ] 实现最小领域模型与配置变更；让 `RetrievalResult` 表达候选，而非语义充分性结论。
- [ ] 运行 `pytest tests/test_config.py tests/test_retrieval.py -v`，确认通过。

### Task 2: 实现结构化改写拆分和语义证据判定

**Files:**
- Create: `rag-customer-service/src/rag_customer_service/answering.py`
- Create: `rag-customer-service/tests/test_answering.py`
- Modify: `rag-customer-service/src/rag_customer_service/agent.py`
- Modify: `rag-customer-service/tests/test_agent.py`

**Interfaces:**
- Produces: `EvidenceAnswering.decompose(question, history) -> tuple[SubQuestion, ...]`；`EvidenceAnswering.judge(subquestions, retrievals) -> tuple[EvidenceDecision, ...]`。
- Consumes: `QwenClient.chat()` 和按全局编号整理的候选证据。

- [ ] 新增失败测试：复合问题拆成两个原子问题；超过 5 项截断；非法 JSON 回退为单问题。
- [ ] 新增失败测试：价格候选只含功能资料时判为不可回答；Wi-Fi 6 项选择直接证据；越权引用和非法 JSON局部拒答。
- [ ] 运行 `pytest tests/test_answering.py -v`，确认缺少实现而失败。
- [ ] 实现严格 JSON 解析、程序重新编号、证据编号白名单和 fail-closed。
- [ ] 改造 Agent 为“拆分 → 每项检索 → 判定”，聚合并标记 supporting/related 候选。
- [ ] 运行 answering、retrieval 和 agent 测试，修复后再进入下一任务。

### Task 3: 实现局部生成、确定性校验和一次重生成

**Files:**
- Modify: `rag-customer-service/src/rag_customer_service/answering.py`
- Modify: `rag-customer-service/tests/test_answering.py`
- Modify: `rag-customer-service/src/rag_customer_service/agent.py`
- Modify: `rag-customer-service/tests/test_agent.py`

**Interfaces:**
- Produces: `EvidenceAnswering.generate()`、`EvidenceAnswering.validate()`、`EvidenceAnswering.compose()`；最终返回按子问题排序的验证后正文。

- [ ] 新增失败测试：只有可回答项进入生成；不可回答项生成固定局部拒答。
- [ ] 新增失败测试：错误引用和证据外数字被确定性校验拒绝；合法引用及数字通过。
- [ ] 新增失败测试：首次语义校验失败时只重生成失败项一次；二次失败局部拒答，其他项保留。
- [ ] 新增失败测试：验证完成前事件流没有正文；全部拒答不调用生成。
- [ ] 实现最小生成、验证、一次重试和分段合并；所有结构化解析失败均安全降级。
- [ ] 运行 answering 和 agent 测试，确认通过且没有未验证正文泄漏。

### Task 4: 扩展安全序列化与前端状态展示

**Files:**
- Modify: `rag-customer-service/src/rag_customer_service/api.py`
- Modify: `rag-customer-service/tests/test_api.py`
- Modify: `rag-customer-service/frontend/src/types.ts`
- Modify: `rag-customer-service/frontend/src/App.tsx`
- Modify: `rag-customer-service/frontend/src/App.test.tsx`
- Modify: `rag-customer-service/frontend/src/index.css`

**Interfaces:**
- Consumes: SSE 中带 `subquestion_id`、`subquestion`、`support_status` 的证据和新增阶段事件。
- Produces: 前端证据标签“回答证据”或“相关资料，不足以直接回答”。

- [ ] 新增 API 失败测试，证明新字段可序列化且本地路径、哈希和提示词仍不泄漏。
- [ ] 新增前端失败测试，证明校验阶段不显示正文，最终局部回答与局部拒答同时出现，证据标签正确。
- [ ] 运行 API 与前端目标测试并确认失败原因来自缺少新行为。
- [ ] 实现最小序列化、类型、状态展示和标签样式。
- [ ] 运行 API 与前端目标测试，确认通过。

### Task 5: 全量回归与真实场景核验

**Files:**
- Modify: `rag-customer-service/README.md`（仅在启动或行为说明需要同步时）
- Verify: all changed files

**Interfaces:**
- Consumes: 前四项完成的完整问答链路。

- [ ] 使用离线假模型覆盖：部分可答、全部拒答、拆分失败、判定失败、首次校验失败后成功、二次失败、数字越界和引用越权。
- [ ] 运行后端全量 `pytest`，任何失败先定位并修复。
- [ ] 运行前端全量 Vitest，任何失败先定位并修复。
- [ ] 运行 TypeScript `tsc --noEmit` 和 Vite 生产构建。
- [ ] 检查 `git diff --check`、配置默认值、SSE 安全字段和未验证正文泄漏边界。
- [ ] 仅在不污染正式知识库的条件下，对现有“产品价格”问题做真实模型只读验证；环境不允许时明确报告，不用模拟结果冒充真实验证。

