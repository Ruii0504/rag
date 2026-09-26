# RAG 客服 React UI 迁移设计

## 目标

以 `design/网页功能原型设计` 和三张深色界面截图为视觉基线，用 React/Vite 替换现有 Streamlit 页面，同时保留已经验证的 Python RAG 领域逻辑、SQLite、Chroma、千问适配器和证据优先回答规则。

## 已确认范围

- 页面保持三个：概览、知识库、智能客服。
- 视觉采用用户提供的深色 Figma Make 导出代码；本次不调用 Figma 插件。
- Figma 只提供 UI 参考，不改变领域模型、摄取、检索、证据判定或拒答规则。
- 先让新前后端通过自动化测试和浏览器验收，再删除 Streamlit 页面与依赖。
- 不上传 `product data/` 中的文档，也不在测试中产生真实千问费用。

## 架构

应用继续保持单进程分层单体。Python 核心前增加一个薄 FastAPI 适配层，React 只通过 JSON API 和 SSE 消费业务能力。

```text
React/Vite
  ├─ JSON：状态、知识库、文档、上传与删除
  └─ SSE：执行事件、证据、回答增量与完成状态
            ↓
FastAPI 适配层
            ↓
AppContainer
  ├─ KnowledgeBaseManager
  └─ CustomerServiceAgent
```

生产构建后由 FastAPI 提供 `frontend/dist` 静态文件；开发时 Vite 将 `/api` 代理到 FastAPI。浏览器会话历史保存在 React 内存中，刷新即清空，与原 Streamlit 会话级语义一致。

## API 合约

- `GET /api/status`：返回系统状态、模型名和汇总存储数据，不返回 API Key。
- `GET /api/knowledge-bases`：返回知识库及文档数、文本块数。
- `POST /api/knowledge-bases`：创建知识库。
- `DELETE /api/knowledge-bases/{kb_id}`：删除知识库及其文档。
- `GET /api/knowledge-bases/{kb_id}/documents`：列出文档。
- `POST /api/knowledge-bases/{kb_id}/documents`：上传一个或多个 UTF-8 Markdown 文档。
- `DELETE /api/knowledge-bases/{kb_id}/documents/{document_id}`：删除文档。
- `POST /api/chat/stream`：接收问题、会话历史和知识库 ID，以 `text/event-stream` 顺序返回原有 `TraceEvent`。

所有响应只序列化前端所需字段。领域对象不直接暴露，`Evidence` 在 API 边界转换为 JSON；已知输入错误返回 4xx，未知错误返回不含密钥和内部提示词的安全消息。

## UI 状态

- 概览：首页按钮可进入客服或知识库，模型与存储状态来自 API。
- 知识库：加载、空列表、创建、选中、上传中、上传成功/失败、删除确认。
- 智能客服：知识库多选、空会话、发送中、执行事件、流式回答、证据卡片、错误/拒答、新会话。
- 保留截图中的响应式行为：窄屏隐藏侧栏详情，主要操作仍可完成。

## 风险与取舍

- React 原型的模拟计时器不能进入最终实现，否则只能得到视觉演示而不是可用系统；所有业务数据改由 API 提供。
- 新增 FastAPI 是替换 Streamlit 后最小的 HTTP 边界，不引入独立服务、队列、认证或全局状态库。
- 上传摄取目前是同步业务调用。首版由请求内执行并在 UI 展示忙碌状态，不增加后台任务系统。
- SSE 只负责服务端到浏览器的事件流，用户请求仍使用普通 POST，避免 WebSocket 的额外复杂度。

## 验收标准

1. Python API 测试证明 CRUD、错误映射和 SSE 事件顺序。
2. React 测试证明三个页面导航、知识库操作和流式对话状态由 API 驱动。
3. 前端生产构建成功，浏览器核对三页布局与截图一致且无控制台错误。
4. 删除 Streamlit 页面、配置和依赖后，后端与前端完整测试仍通过。
5. 本次实现没有调用 Figma 插件，也没有向外部服务上传产品资料。
