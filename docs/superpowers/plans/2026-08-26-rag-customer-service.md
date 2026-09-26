# RAG 智能客服系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可在本机运行、支持 Markdown 知识库管理、多知识库检索、证据优先回答、流式输出和工具过程展示的学习型 RAG 智能客服系统。

**Architecture:** 使用单进程分层单体。Streamlit 页面保持薄，知识库、摄取、检索和 LangGraph 各自通过小型 Interface 协作；SQLite、Chroma、本地文件和千问接口的实现细节封装在对应模块中。Agent 使用固定图流程，不使用自由 ReAct 决策。UI 先以 Streamlit 原生风格完成三个页面的功能原型，再通过 Codex Figma 插件写入 Figma，由用户微调并冻结节点，最后把批准的节点适配回现有 Streamlit 页面。

**Tech Stack:** Python 3.12、Streamlit、LangGraph、LangChain OpenAI 集成、Chroma、SQLite、千问官方 OpenAI 兼容接口、pytest；Codex Figma 插件仅用于 UI 设计交接，不进入应用运行时依赖。

**Spec:** `docs/superpowers/specs/2026-08-26-rag-customer-service-design.md`

## Global Constraints

- 项目目录固定为 `rag-customer-service/`。
- Python 运行时固定为 3.12（经用户确认由原计划 3.11 调整）。
- 只支持 Markdown，不实现其他文档格式。
- 单实例、无登录、少量并发、最多数万文本块。
- 千问 API Key 只能来自环境变量，不写入源码、日志或 Git。
- Chat 与 Embedding 均使用千问官方 OpenAI 兼容接口。
- 所有知识库共用一个 Chroma collection，通过 `kb_id` 过滤。
- 同一知识库内按规范化内容的 SHA-256 哈希去重。
- Agent 每次提问必须检索。
- 证据不足必须拒答。
- 对话历史只保存在 `st.session_state`。
- Figma 只调整视觉、布局和信息层级，不改变业务流程、领域 Interface 或数据持久化规则。
- Figma 是批准后 UI 的视觉基线，现有 Python 代码和自动化测试是业务行为基线；两者发生冲突时不得静默修改业务行为。
- UI 实现优先使用 Streamlit 原生组件、主题配置和少量 CSS，不引入自定义前端框架、JavaScript 注入或自定义 Streamlit Component。
- 从 Figma 回写代码前，必须取得首页、知识库页、客服页及关键状态的节点级链接和用户明确批准。
- 不引入 FastAPI、Redis、Celery、数据库迁移框架、混合检索或重排序器。
- 测试仅覆盖关键学习链路，不追求全面覆盖率。
- 每个任务完成后先通过该任务验证，再进入下一任务。

---

## 文件与 Interface 总览

### 页面文件

- `app.py`：首页和运行说明。
- `pages/1_知识库管理.py`：知识库、文档和上传操作。
- `pages/2_智能客服对话.py`：知识库选择、聊天、工具过程和证据展示。
- `.streamlit/config.toml`：保存批准后的 Streamlit 主题配置。
- `docs/ui/figma-handoff.md`：记录 Figma 文件、批准节点、页面状态和回写验收结果。

### 核心模块

- `config.py`：产生 `Settings`。
- `models.py`：定义 `KnowledgeBase`、`Document`、`PreparedDocument`、`Chunk`、`Evidence`、`RetrievalResult`、`TraceEvent`。
- `qwen.py`：产生 `QwenClient`，提供聊天、流式聊天和向量化能力。
- `storage.py`：产生 `Storage`，管理 SQLite 元数据和原始 Markdown 文件。
- `vector_store.py`：产生 `ChromaVectorStore`，管理唯一 collection。
- `ingestion.py`：产生 `DocumentIngestor`，完成校验、规范化、哈希和切分。
- `knowledge_base.py`：产生 `KnowledgeBaseManager`，协调完整知识库生命周期。
- `retrieval.py`：产生 `Retriever`，执行多知识库向量检索和证据过滤。
- `agent.py`：产生 `CustomerServiceAgent`，封装 LangGraph 和事件流。
- `bootstrap.py`：产生 `AppContainer`，集中构造并缓存上述对象。

### 约定的核心 Interface

- `QwenClient.chat(messages)`：返回完整文本。
- `QwenClient.stream_chat(messages)`：按顺序产出文本增量。
- `QwenClient.embed_documents(texts)`：批量返回文档向量。
- `QwenClient.embed_query(text)`：返回单条查询向量。
- `DocumentIngestor.prepare(filename, content)`：返回规范化后的 `PreparedDocument`。
- `KnowledgeBaseManager.ingest_document(kb_id, filename, content)`：返回已入库 `Document`。
- `Retriever.retrieve(query, kb_ids)`：返回 `RetrievalResult`。
- `CustomerServiceAgent.stream(question, history, kb_ids)`：按执行顺序产出 `TraceEvent`。
- `AppContainer` 只暴露 `knowledge_bases` 和 `agent` 两个页面入口对象。

---

### Task 1: 建立可运行的项目骨架、配置和领域模型

**Files:**
- Create: `rag-customer-service/requirements.txt`
- Create: `rag-customer-service/pyproject.toml`
- Create: `rag-customer-service/.env.example`
- Create: `rag-customer-service/.gitignore`
- Create: `rag-customer-service/src/rag_customer_service/__init__.py`
- Create: `rag-customer-service/src/rag_customer_service/config.py`
- Create: `rag-customer-service/src/rag_customer_service/models.py`
- Create: `rag-customer-service/tests/test_config.py`

**Consumes:** 无。

**Produces:** `Settings` 以及后续任务共享的数据模型；一个可 editable install 的 `src/` 包。

- [ ] **Step 1: 创建虚拟环境与依赖清单**

  使用 Python 3.12 创建 `.venv`。依赖清单只包含 Streamlit、LangGraph、LangChain Core、LangChain OpenAI、LangChain Text Splitters、Chroma、python-dotenv 和 pytest。安装成功后固定实际兼容版本，不同时维护第二套依赖声明。

- [ ] **Step 2: 配置包发现**

  在 `pyproject.toml` 中只配置构建后端、项目名称和 `src` 包发现。完成 editable install 后，从项目根目录验证可以导入 `rag_customer_service`。

- [ ] **Step 3: 先定义配置失败场景**

  在 `test_config.py` 中定义两个验收场景：环境变量齐全时路径及检索参数被正确解析；缺少 `QWEN_API_KEY` 时抛出明确配置错误。执行 `pytest tests/test_config.py -v`，确认实现前失败。

- [ ] **Step 4: 实现最小配置模型**

  `Settings` 只包含千问连接信息、模型名、Embedding 维度、数据路径、切分参数、Top-K 和相似度阈值。默认值记录在 `.env.example`，真实 Key 留空。

- [ ] **Step 5: 定义领域数据结构**

  每个模型只保存跨模块必要字段；禁止把 Chroma 或 OpenAI SDK 的原始对象放入模型。`TraceEvent` 至少区分节点状态、工具状态、证据、回答增量、完成和错误。

- [ ] **Step 6: 验证并提交**

  执行配置测试和包导入检查。预期全部通过。初始化 Git 仓库后，以“建立项目骨架和配置”为独立提交。

**Exit criteria:** 新环境按 README 中的准备命令安装后能导入包；配置错误不会延迟到第一次模型调用才出现。

---

### Task 2: 实现 SQLite 元数据和本地文件存储

**Files:**
- Create: `rag-customer-service/src/rag_customer_service/storage.py`
- Create: `rag-customer-service/tests/test_storage.py`
- Modify: `rag-customer-service/src/rag_customer_service/models.py`

**Consumes:** `Settings`、`KnowledgeBase`、`Document`。

**Produces:** `Storage`，供知识库模块使用。

- [ ] **Step 1: 定义数据库验收场景**

  覆盖知识库新建、列表、删除；文档新增、列表、删除；同一 `kb_id` 下重复 `content_hash` 被拒绝；不同 `kb_id` 下相同哈希允许保存。

- [ ] **Step 2: 定义文件验收场景**

  使用 pytest 临时目录验证：文件按 `uploads/<kb_id>/<document_id>.md` 保存；删除文档后文件消失；文件名不能决定真实存储路径。

- [ ] **Step 3: 执行测试并确认失败**

  运行 `pytest tests/test_storage.py -v`。预期因为 `Storage` 尚不存在而失败。

- [ ] **Step 4: 初始化 SQLite schema**

  创建 `knowledge_bases` 与 `documents` 两张表；启用外键；为知识库名称建立唯一约束；为 `(kb_id, content_hash)` 建立联合唯一约束。所有 ID 使用应用生成的字符串 ID。

- [ ] **Step 5: 实现元数据和文件操作**

  SQLite 查询只返回领域模型。文件保存前创建目标目录。所有删除操作使用明确的 `kb_id` 和 `document_id`，禁止根据用户上传文件名拼接删除路径。

- [ ] **Step 6: 验证并提交**

  执行 `pytest tests/test_storage.py -v`。预期全部通过。提交信息聚焦“增加知识库元数据与文件存储”。

**Exit criteria:** 元数据约束与文件路径规则均由自动化测试证明，测试不写入真实 `data/`。

---

### Task 3: 封装千问聊天与 Embedding 调用

**Files:**
- Create: `rag-customer-service/src/rag_customer_service/qwen.py`
- Create: `rag-customer-service/tests/test_qwen.py`
- Modify: `rag-customer-service/README.md`

**Consumes:** `Settings`。

**Produces:** `QwenClient` 的四个固定方法：完整聊天、流式聊天、批量文档向量、单条查询向量。

- [ ] **Step 1: 定义离线 Adapter 测试**

  用可替换的底层客户端验证 Base URL、API Key、聊天模型、Embedding 模型和维度被正确传递；流式增量保持原顺序；批量输入不会被静默丢失。

- [ ] **Step 2: 执行测试并确认失败**

  运行 `pytest tests/test_qwen.py -v`，确认 `QwenClient` 尚未实现。

- [ ] **Step 3: 实现最小千问封装**

  模块只负责协议适配，不包含 RAG 提示词、知识库逻辑或 Streamlit 输出。异常转换成简短、可展示且不含 API Key 的错误。

- [ ] **Step 4: 运行离线测试**

  执行 `pytest tests/test_qwen.py -v`，预期全部通过且不产生真实 API 费用。

- [ ] **Step 5: 执行一次人工连通性检查**

  在本地设置真实环境变量，分别验证一条聊天请求、一条流式请求和一条 Embedding 请求。记录返回向量维度，并确认与 `QWEN_EMBEDDING_DIMENSION` 一致。测试输出不得打印 Key。

- [ ] **Step 6: 更新 README 并提交**

  记录千问环境变量配置、连通性错误排查和模型/维度变更后必须重建向量的限制。提交信息聚焦“封装千问模型接口”。

**Exit criteria:** 业务模块不需要知道 OpenAI 兼容客户端的构造参数；真实连通性检查成功。

---

### Task 4: 实现 Markdown 校验、哈希和标题感知切分

**Files:**
- Create: `rag-customer-service/src/rag_customer_service/ingestion.py`
- Create: `rag-customer-service/tests/test_ingestion.py`
- Modify: `rag-customer-service/src/rag_customer_service/models.py`

**Consumes:** `Settings` 中的 chunk 大小和 overlap。

**Produces:** `DocumentIngestor.prepare()` 与 `PreparedDocument`。

- [ ] **Step 1: 定义切分验收样例**

  覆盖 UTF-8 Markdown、多级标题、超长章节、没有标题的正文、空文件、非 `.md` 文件、Windows 与 Unix 换行。每个块必须带文档名、标题路径和稳定序号。

- [ ] **Step 2: 定义哈希验收样例**

  相同语义内容仅换行风格不同时，应在规范化后得到同一 SHA-256；内容真正变化时哈希必须变化。

- [ ] **Step 3: 执行测试并确认失败**

  运行 `pytest tests/test_ingestion.py -v`。

- [ ] **Step 4: 实现最小处理流程**

  固定顺序为：扩展名检查、UTF-8 解码、换行规范化、空内容检查、哈希、按 Markdown 标题分段、超长段递归切分、生成块元数据。此模块不调用 SQLite、Chroma 或 Streamlit。

- [ ] **Step 5: 使用现有产品文档做只读抽查**

  对 `product data/01-产品介绍与技术规格.md` 和 `product data/04-Mesh组网、固件升级与恢复出厂.md` 执行切分，确认三级标题路径和长章节处理符合预期，不写回原文件。

- [ ] **Step 6: 验证并提交**

  执行 `pytest tests/test_ingestion.py -v`，预期全部通过。提交信息聚焦“增加 Markdown 摄取准备流程”。

**Exit criteria:** 切分结果稳定、可引用、可重复生成；非法文件在调用 Embedding 前被拒绝。

---

### Task 5: 实现单 collection 的 Chroma 向量存储

**Files:**
- Create: `rag-customer-service/src/rag_customer_service/vector_store.py`
- Create: `rag-customer-service/tests/test_vector_store.py`
- Modify: `rag-customer-service/src/rag_customer_service/models.py`

**Consumes:** `Settings`、`Chunk`、外部生成的向量。

**Produces:** `ChromaVectorStore` 的新增、检索、按文档删除和按知识库删除能力。

- [ ] **Step 1: 定义隔离临时 Chroma 的测试**

  使用固定小向量覆盖：添加多个知识库的数据；通过多个 `kb_id` 查询；返回文档、元数据和距离；删除单文档；删除整知识库。

- [ ] **Step 2: 定义 collection 不变量测试**

  保存 Embedding 模型标识与维度。再次初始化时如果配置不一致，必须明确拒绝启动并提示重新向量化。

- [ ] **Step 3: 执行测试并确认失败**

  运行 `pytest tests/test_vector_store.py -v`。

- [ ] **Step 4: 实现唯一 collection**

  collection 名固定。块 ID 必须全局唯一。查询使用 `kb_id` 的集合过滤，删除使用明确的 `document_id` 或 `kb_id` 过滤。

- [ ] **Step 5: 统一检索距离输出**

  `vector_store.py` 返回原始距离及块数据，不在此处决定证据是否充分；距离到业务相似度的解释由 `retrieval.py` 集中处理。

- [ ] **Step 6: 验证并提交**

  执行 `pytest tests/test_vector_store.py -v`，预期全部通过。提交信息聚焦“增加 Chroma 单集合存储”。

**Exit criteria:** 多知识库过滤和两种删除路径均有自动化证据；模型维度不一致不会污染 collection。

---

### Task 6: 组合完整知识库生命周期

**Files:**
- Create: `rag-customer-service/src/rag_customer_service/knowledge_base.py`
- Create: `rag-customer-service/tests/test_knowledge_base.py`

**Consumes:** `Storage`、`DocumentIngestor`、`QwenClient`、`ChromaVectorStore`。

**Produces:** `KnowledgeBaseManager`，作为知识库页面的唯一业务入口。

- [ ] **Step 1: 定义成功摄取测试**

  使用假 Embedding 验证一次上传按顺序完成规范化、同库查重、向量化、原文件保存、Chroma 写入和 SQLite 登记，并返回 chunk 数量。

- [ ] **Step 2: 定义失败补偿测试**

  分别模拟 Embedding、文件写入、Chroma 写入和 SQLite 登记失败。每种失败后均验证本次文档没有可检索向量、没有文件残留、没有成功文档元数据。

- [ ] **Step 3: 定义删除测试**

  删除文档后验证三层数据全部消失；删除知识库后验证其全部文档、文件和向量消失，其他知识库不受影响。

- [ ] **Step 4: 执行测试并确认失败**

  运行 `pytest tests/test_knowledge_base.py -v`。

- [ ] **Step 5: 实现协调流程和进度事件**

  上传过程至少报告“校验、切分、向量化、保存、完成”五类进度。补偿操作按已完成步骤逆序执行；补偿失败要保留原始错误并记录安全的清理提示。

- [ ] **Step 6: 验证并提交**

  执行知识库、存储、切分和向量存储测试。预期全部通过。提交信息聚焦“完成知识库生命周期”。

**Exit criteria:** 页面未来只需调用 `KnowledgeBaseManager`；任何已覆盖的失败点不会产生半完成文档。

---

### Task 7: 实现多知识库检索与证据判定

**Files:**
- Create: `rag-customer-service/src/rag_customer_service/retrieval.py`
- Create: `rag-customer-service/tests/test_retrieval.py`

**Consumes:** `QwenClient.embed_query()`、`ChromaVectorStore.search()`、Top-K 和相似度阈值配置。

**Produces:** `Retriever.retrieve()` 与 `RetrievalResult`。

- [ ] **Step 1: 定义选择范围测试**

  验证未选择知识库时返回明确的无证据结果；选择一个知识库时不泄漏其他知识库结果；选择多个知识库时统一排序。

- [ ] **Step 2: 定义阈值与 Top-K 测试**

  使用固定距离数据验证最多返回 Top-K 条证据，低于阈值的项被移除，全部移除时 `has_sufficient_evidence` 为假。

- [ ] **Step 3: 执行测试并确认失败**

  运行 `pytest tests/test_retrieval.py -v`。

- [ ] **Step 4: 实现检索 Interface**

  检索顺序固定为：校验知识库选择、生成查询向量、带 `kb_id` 过滤查询、统一相似度、排序、截断、阈值过滤、构造编号证据。不要在这里生成客服答案。

- [ ] **Step 5: 验证并提交**

  执行 `pytest tests/test_retrieval.py tests/test_vector_store.py -v`。预期全部通过。提交信息聚焦“增加证据优先检索”。

**Exit criteria:** `RetrievalResult` 足以让 Agent 分支，不需要理解 Chroma 原始结构。

---

### Task 8: 构建确定性 LangGraph 和统一事件流

**Files:**
- Create: `rag-customer-service/src/rag_customer_service/agent.py`
- Create: `rag-customer-service/tests/test_agent.py`
- Modify: `rag-customer-service/src/rag_customer_service/models.py`

**Consumes:** `QwenClient`、`Retriever`、当前会话历史和知识库 ID。

**Produces:** `CustomerServiceAgent.stream()` 的 `TraceEvent` 序列。

- [ ] **Step 1: 定义图分支测试**

  验证每次问题都调用一次检索；证据不足进入拒答节点且不调用答案生成；证据充分进入生成节点；未选择知识库直接形成可解释拒答。

- [ ] **Step 2: 定义问题改写测试**

  会话包含“如何恢复出厂？”后，追问“这样会清空设置吗？”时，改写请求必须包含必要历史；没有历史且问题已独立时保持原意。

- [ ] **Step 3: 定义事件顺序测试**

  最小顺序为：问题改写开始与完成、检索工具开始、证据结果、证据判定、回答增量或拒答、完成。错误时最后事件为安全错误，不包含 Key 或完整提示词。

- [ ] **Step 4: 执行测试并确认失败**

  运行 `pytest tests/test_agent.py -v`。

- [ ] **Step 5: 构建固定图**

  图只包含问题改写、检索、证据判定、生成回答和拒答节点。检索节点始终执行；证据判定只读取 `RetrievalResult`，不调用额外模型。

- [ ] **Step 6: 约束回答提示词**

  生成节点只接收编号证据，要求回答内使用来源编号，不允许使用证据之外的产品事实。拒答节点使用固定、简短的中文话术。

- [ ] **Step 7: 验证并提交**

  执行 `pytest tests/test_agent.py tests/test_retrieval.py -v`。预期全部通过。提交信息聚焦“构建确定性客服 Agent”。

**Exit criteria:** Agent 的两条业务分支、工具调用次数和事件顺序均可重复验证。

---

### Task 9: 集中组合依赖并缓存资源

**Files:**
- Create: `rag-customer-service/src/rag_customer_service/bootstrap.py`
- Create: `rag-customer-service/tests/test_bootstrap.py`

**Consumes:** 所有已完成模块。

**Produces:** `AppContainer` 与统一 `get_container()` 入口。

- [ ] **Step 1: 定义组合测试**

  在临时数据目录和假 Qwen 客户端下构造容器，验证页面入口只暴露知识库管理器和 Agent；构造过程会初始化 SQLite 与 Chroma。

- [ ] **Step 2: 执行测试并确认失败**

  运行 `pytest tests/test_bootstrap.py -v`。

- [ ] **Step 3: 实现唯一组合入口**

  `bootstrap.py` 是唯一允许创建具体 `Storage`、`ChromaVectorStore`、`QwenClient`、`Retriever` 和 `CustomerServiceAgent` 的位置。使用 Streamlit 资源缓存保证脚本重跑不会重复创建客户端。

- [ ] **Step 4: 验证并提交**

  执行 `pytest tests/test_bootstrap.py -v`。预期全部通过。提交信息聚焦“集中应用依赖组合”。

**Exit criteria:** 两个页面不需要知道模块构造顺序，也不会各自创建一套数据库或 Chroma 客户端。

---

### Task 10: 实现首页和知识库管理功能原型

**Files:**
- Create: `rag-customer-service/app.py`
- Create: `rag-customer-service/pages/1_知识库管理.py`
- Modify: `rag-customer-service/README.md`

**Consumes:** `AppContainer.knowledge_bases`。

**Produces:** 使用 Streamlit 原生组件、可操作且可写入 Figma 的首页和知识库管理功能原型。

- [ ] **Step 1: 建立首页和运行入口**

  首页只展示项目用途、两个页面的入口、当前 Chat/Embedding 模型名称以及配置状态。不得展示 Key。此阶段只使用 Streamlit 原生布局和组件，不做品牌化视觉精修。

- [ ] **Step 2: 实现知识库区域**

  展示知识库列表、当前选择、新建输入和删除确认。禁止删除未明确选择的知识库；删除后立即刷新列表。

- [ ] **Step 3: 实现文档区域**

  只允许 `.md` 文件。上传按钮触发同步摄取，按进度事件更新界面。重复文档、非法编码、空文档和模型错误分别显示明确中文信息。

- [ ] **Step 4: 实现文档列表与删除**

  列表至少显示原文件名、状态、chunk 数量和上传时间。删除前要求确认，完成后刷新列表。

- [ ] **Step 5: 覆盖原型关键状态**

  首页至少可观察正常配置和缺少配置两种状态。知识库页至少可观察空列表、已有知识库、上传处理中、重复文档、非法文档、模型失败和删除确认状态。状态通过真实业务条件或测试数据触发，不额外实现仅供展示的假业务分支。

- [ ] **Step 6: 人工验收知识库生命周期**

  创建“路由器产品知识”知识库，上传 `product data/` 中 6 份文档；再次上传其中一份，确认被同库哈希去重；删除一份文档后确认列表、文件和向量同步消失。

- [ ] **Step 7: 更新 README 并提交**

  写明启动命令、页面入口和知识库操作流程。提交信息聚焦“增加首页和知识库功能原型”。

**Exit criteria:** 首页和知识库页使用 Streamlit 原生风格运行；不使用命令行即可完成知识库与文档的完整生命周期；关键状态足以作为 Figma UI 微调的输入。

---

### Task 11: 实现客服对话功能原型

**Files:**
- Create: `rag-customer-service/pages/2_智能客服对话.py`
- Modify: `rag-customer-service/README.md`

**Consumes:** `AppContainer.agent`、知识库列表、`TraceEvent`。

**Produces:** 使用 Streamlit 原生组件、当前会话内可多轮使用且可写入 Figma 的客服功能原型。

- [ ] **Step 1: 实现知识库多选和输入约束**

  页面顶部多选知识库。没有选中知识库时禁用或拒绝提问，并说明原因。传给 Agent 的只能是稳定 `kb_id`，不能是名称。

- [ ] **Step 2: 实现当前会话历史**

  在 `st.session_state` 保存用户消息和最终助手消息。刷新或服务重启后不恢复历史，不写入 SQLite。

- [ ] **Step 3: 实现工具过程区域**

  按 `TraceEvent` 顺序显示问题改写、检索开始、启用知识库、命中数量、证据判定和错误。证据项展示文档名、标题路径、相似度和片段。

- [ ] **Step 4: 实现回答流**

  回答增量进入同一个占位区域，结束后把完整答案写入会话历史。证据不足时展示固定拒答，并仍保留检索过程供用户理解。

- [ ] **Step 5: 覆盖原型关键状态**

  客服页至少可观察未选择知识库、空会话、问题改写、检索中、流式回答、回答完成、证据不足和安全错误状态。此阶段不添加 Figma 尚未批准的自定义视觉组件。

- [ ] **Step 6: 人工多轮验收**

  先问“如何恢复出厂设置？”，再问“这样会清空原来的网络设置吗？”。确认第二问经过问题改写并检索维护文档。答案必须展示对应文档和章节证据。

- [ ] **Step 7: 人工越界验收**

  询问“今天北京天气怎么样？”。确认系统执行检索后拒答，不借助模型常识回答天气。

- [ ] **Step 8: 提交**

  提交信息聚焦“增加客服对话功能原型”。

**Exit criteria:** 首页、知识库页和客服页均具备可运行的功能原型；用户能够看到 Agent 执行过程、证据和流式回答；全部关键 UI 状态已准备好进入 Figma。

---

### Task 12: 通过 Codex Figma 插件微调并冻结 UI

**Files:**
- Create: `rag-customer-service/docs/ui/figma-handoff.md`
- Read only: `rag-customer-service/app.py`
- Read only: `rag-customer-service/pages/1_知识库管理.py`
- Read only: `rag-customer-service/pages/2_智能客服对话.py`
- External artifact: 同一个 Figma Design 文件中的页面与状态节点。

**Consumes:** Task 10～11 的可运行 Streamlit 功能原型及其关键状态。

**Produces:** 用户批准的 Figma 节点级设计基线，以及记录文件链接、节点链接、状态覆盖和批准结果的 `figma-handoff.md`。

**Required Figma workflow:** 写入页面前使用 `figma-generate-design` 与 `figma-use`；需要新建 Figma 文件时先使用 `figma-create-new-file`。

- [ ] **Step 1: 准备同一个 Figma 目标文件**

  确认 Codex 已连接 Figma 插件。已有目标文件时复用该文件；没有目标文件时，通过 Codex Figma 插件创建名为“RAG 智能客服 UI”的 Figma Design 文件。后续页面、状态和微调全部在该文件中完成。

- [ ] **Step 2: 将功能原型写入 Figma**

  启动 Streamlit 原型，读取三个页面源码和实际页面状态。通过 Codex Figma 插件把首页、知识库页和客服页写入同一个 Figma 文件；保留当前 Streamlit 信息结构和交互限制，优先复用文件内或已发布设计系统的组件、变量与样式。

- [ ] **Step 3: 建立页面状态矩阵**

  在 Figma 中至少保留以下可单独定位的节点：首页默认态；知识库空态、列表态、上传中、错误态和删除确认态；客服空态、检索中、流式回答、回答完成、证据不足和错误态。重复元素使用组件或实例，禁止把所有状态做成无法复用的扁平图层。

- [ ] **Step 4: 由用户完成 UI 微调**

  暂停代码开发，把同一 Figma 文件交给用户。用户可以直接编辑 Figma，也可以继续在 Codex 中使用 Figma 插件调整视觉、布局和信息层级；不得在此阶段新增业务流程、接口、持久化需求或非 Streamlit 原生能力。

- [ ] **Step 5: 执行 Streamlit 可实现性复核**

  对用户微调后的节点逐项检查。布局必须能由 Streamlit 原生组件、主题配置和少量 CSS 实现；若节点依赖复杂动画、自由 DOM、JavaScript 或自定义组件，先在 Figma 中收敛设计，不把这些能力静默加入代码。

- [ ] **Step 6: 冻结节点级交接物**

  用户明确确认 UI 后，在 `docs/ui/figma-handoff.md` 记录 Figma 文件链接、每个批准节点的名称和节点级链接、对应页面状态、批准日期，以及“仅调整 UI、不改变业务行为”的约束。只有文件级链接或未批准的草稿节点不满足交接条件。

- [ ] **Step 7: 验证并提交**

  逐一打开交接文档中的节点级链接，确认都指向同一 Figma 文件且覆盖状态矩阵。提交信息聚焦“记录批准的 Figma UI 基线”。

**Exit criteria:** 用户已在 Codex Figma 插件或 Figma 中完成微调并明确批准；三个页面及关键状态都有可读取的节点级链接；设计在既定 Streamlit 技术边界内可实现。

---

### Task 13: 将批准的 Figma UI 适配回 Streamlit

**Files:**
- Create: `rag-customer-service/.streamlit/config.toml`
- Modify: `rag-customer-service/app.py`
- Modify: `rag-customer-service/pages/1_知识库管理.py`
- Modify: `rag-customer-service/pages/2_智能客服对话.py`
- Modify: `rag-customer-service/docs/ui/figma-handoff.md`
- Modify: `rag-customer-service/README.md`

**Consumes:** `figma-handoff.md` 中用户批准的节点级链接，以及 Task 10～11 已验收的业务行为。

**Produces:** 视觉上匹配批准节点、行为上保持原验收结果的 Streamlit 页面。

**Required Figma workflow:** 读取节点并开始编码前使用 `figma-design-to-code`，以每个节点的设计上下文和截图作为实现参考。

- [ ] **Step 1: 读取批准节点的设计上下文**

  通过 Codex Figma 插件逐个读取 `figma-handoff.md` 中的批准节点，获取设计上下文和截图。插件返回的 React/Tailwind 代码只作为视觉参考，必须适配到现有 Python/Streamlit 结构，禁止直接引入另一套前端运行时。

- [ ] **Step 2: 建立设计到 Streamlit 的映射清单**

  为颜色、字体、间距、圆角、页面区块和状态展示建立映射。优先映射到 `.streamlit/config.toml` 和 Streamlit 原生组件；只有主题配置无法表达的已批准差异才使用局部 CSS。

- [ ] **Step 3: 实现统一 Streamlit 主题**

  在 `.streamlit/config.toml` 中写入批准的基础主题。不得加入 API Key、模型配置或业务参数，也不创建第二套设计令牌系统。

- [ ] **Step 4: 适配首页**

  只修改布局、文案呈现和视觉样式，保持模型名称、配置状态和页面入口的原有来源及行为。

- [ ] **Step 5: 适配知识库页**

  实现批准节点对应的空态、列表态、上传进度、错误和删除确认视觉。知识库创建、上传、去重、删除以及刷新行为不得改变。

- [ ] **Step 6: 适配客服页**

  实现批准节点对应的空态、检索过程、证据、流式回答、拒答和错误视觉。`TraceEvent` 顺序、检索调用次数、历史范围和拒答逻辑不得改变。

- [ ] **Step 7: 执行逐状态视觉比对**

  在与 Figma 对应的页面状态下截取本地 Streamlit 页面，逐项核对字体、颜色、间距、文本裁切、重叠和组件状态。发现差异时只做定向修正，不重写已通过验收的业务模块。

- [ ] **Step 8: 执行行为回归验证**

  重新执行 `pytest -v`，并重复 Task 10～11 的知识库生命周期、多轮追问和越界拒答验收。把视觉比对结果和未影响业务行为的结论记录到 `figma-handoff.md`。

- [ ] **Step 9: 更新 README 并提交**

  README 记录 Figma 交接文件位置和 UI 修改边界。提交信息聚焦“应用批准的 Figma UI”。

**Exit criteria:** 三个页面及关键状态与批准的 Figma 节点一致；全部自动化测试和 Task 10～11 的行为验收仍通过；没有引入新的业务流程或前端框架。

---

### Task 14: 端到端验收、阈值调优和交付文档

**Files:**
- Create: `rag-customer-service/tests/evaluation_cases.md`
- Modify: `rag-customer-service/README.md`
- Modify: `rag-customer-service/.env.example`
- Modify: `rag-customer-service/docs/ui/figma-handoff.md`

**Consumes:** 完整系统、批准的 Figma UI 基线和 `product data/` 中 6 份现有 Markdown。

**Produces:** 可重复执行的验收清单、初始 Top-K/阈值结果、Figma UI 验收记录和完整运行说明。

- [ ] **Step 1: 建立最小评测集**

  至少记录 12 个问题：产品规格、首次联网、WiFi 修改、Mesh、恢复出厂、指示灯故障、保修各至少一题；多轮追问至少两题；知识库外问题至少三题。每题记录预期命中文档和是否应拒答。

- [ ] **Step 2: 调整 Top-K 与阈值**

  使用同一套评测问题比较少量候选配置。选择能覆盖应回答问题且不会让越界问题通过的最简单配置，并把最终值写入 `.env.example` 和评测记录。

- [ ] **Step 3: 执行全部自动化测试**

  运行 `pytest -v`。预期无失败。真实千问连通性检查与单元测试分开，避免每次测试产生费用。

- [ ] **Step 4: 执行数据一致性验收**

  上传全部 6 份产品文档，记录 SQLite 文档数和 Chroma chunk 数；删除单文档后确认其向量数归零；删除知识库后确认 SQLite、Chroma 和上传目录均无相关数据。

- [ ] **Step 5: 执行最终 UI 和业务验收**

  从空数据目录启动，完整走通新建知识库、上传、重复拦截、文档删除、多库选择、多轮对话、证据引用、拒答和知识库删除；同时按 `figma-handoff.md` 复核三个页面及关键状态仍匹配批准节点。

- [ ] **Step 6: 完成 README**

  README 必须包含：Python 版本、安装步骤、环境变量、启动命令、目录说明、数据重置方式、Embedding 模型变更限制、Figma UI 交接边界、已知 MVP 限制和验收步骤。

- [ ] **Step 7: 最终提交**

  确认 `.env`、`data/app.db`、`data/chroma/` 和 `data/uploads/` 未被 Git 跟踪。提交评测记录、Figma UI 验收记录和交付文档。

**Exit criteria:** 新开发者仅按 README 即可从空目录启动并复现业务与 UI 验收流程；所有核心需求都有明确验收证据；批准的 Figma 节点能够追溯到最终页面。

---

## 推荐执行节奏

- 第一阶段：Task 1～5，得到配置、存储、模型、切分和向量库底座。
- 第二阶段：Task 6～9，得到可测试的知识库、检索、Agent 和组合入口。
- 第三阶段：Task 10～11，使用 Streamlit 原生风格完成首页、知识库页和客服页的功能原型。
- 第四阶段：Task 12，在 Codex Figma 插件中写入三个页面，由用户微调并冻结节点级 UI 基线；此阶段暂停代码视觉精修。
- 第五阶段：Task 13～14，把批准的 Figma 节点适配回 Streamlit，再完成端到端验证和交付文档。
- 每个阶段结束后做一次人工复盘；不要在底层链路未通过时提前制作完整 UI，也不要在 Figma 节点未获用户批准时开始设计回写。

## 完成定义

- `streamlit run app.py` 能从项目根目录启动。
- 6 份现有产品 Markdown 能全部摄取。
- 同库重复文档被拦截，不同知识库允许相同内容。
- 多知识库过滤正确。
- 工具调用过程和证据可见。
- 最终回答逐步流式显示。
- 有证据时只依据证据回答，无证据时明确拒答。
- 多轮指代问题经过改写后能检索到正确章节。
- 删除文档或知识库后不存在可召回残留。
- 首页、知识库页和客服页先完成可运行的 Streamlit 功能原型，再进入 Figma 微调。
- `docs/ui/figma-handoff.md` 记录用户批准的节点级链接和关键状态，最终页面能追溯到这些节点。
- Figma 回写只改变 UI；知识库生命周期、`TraceEvent` 顺序、强制检索、拒答和会话历史范围均保持不变。
- 最终三个页面及关键状态与批准的 Figma 节点一致，并保持 Streamlit 原生组件、主题配置和少量 CSS 的实现边界。
- 所有自动化测试通过，真实 API Key 未进入仓库或日志。
