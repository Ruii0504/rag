# RAG 客服 React UI 迁移实施计划

**目标：** 将用户提供的深色 React 原型接入现有 Python RAG 核心，并在替代界面验证通过后移除 Streamlit。

**设计：** `docs/superpowers/specs/2026-08-27-react-ui-migration-design.md`

**约束：** 不调用 Figma 插件；每个阶段先写失败测试，再做最小实现；阶段测试失败必须定位并修复；最后运行所有测试和生产构建。

## 阶段 1：建立 FastAPI 合约

**文件：**

- 新建 `rag-customer-service/tests/test_api.py`
- 新建 `rag-customer-service/src/rag_customer_service/api.py`
- 修改 `rag-customer-service/src/rag_customer_service/bootstrap.py`
- 修改 `rag-customer-service/requirements.txt`

1. 为状态、知识库列表/创建/删除、文档列表/上传/删除写 API 测试并确认失败。
2. 为聊天 SSE 的事件顺序、证据序列化、安全错误写 API 测试并确认失败。
3. 实现只依赖 `AppContainer` 的 FastAPI 工厂和最小序列化函数。
4. 用 `pytest tests/test_api.py tests/test_bootstrap.py -v` 验证。

**成功标准：** API 测试不访问真实模型或真实数据目录，且覆盖 React 所需业务入口。

## 阶段 2：迁入 React 原型并建立前端测试

**文件：**

- 新建 `rag-customer-service/frontend/`
- 以 `design/网页功能原型设计/` 的 Vite、TypeScript、React 和 CSS 为视觉基线
- 新建 `frontend/src/api.ts`、`frontend/src/types.ts` 与组件测试

1. 迁入不含缓存和构建产物的源码。
2. 添加 Vitest 与 Testing Library，先写导航和 API 驱动状态测试并确认失败。
3. 保持原 CSS 和页面信息层级，移除模拟 `setTimeout` 数据源。
4. 用 `pnpm test --run` 和 `pnpm build` 验证。

**成功标准：** 三页可导航；知识库页和概览页显示测试 API 数据；无模拟业务计时器。

## 阶段 3：接入知识库操作

**文件：**

- 修改 `frontend/src/App.tsx`、`frontend/src/api.ts`
- 修改相应前端测试

1. 先测试创建、选择、上传、删除以及失败提示。
2. 实现 API 调用和刷新策略；保留原型的弹窗、确认和 toast 样式。
3. 运行知识库相关前端测试和后端 API 测试。

**成功标准：** 知识库与文档状态全部来自 Python 核心，上传仅接受 Markdown。

## 阶段 4：接入 SSE 客服会话

**文件：**

- 修改 `frontend/src/App.tsx`、`frontend/src/api.ts`
- 修改相应前端测试

1. 先测试发送请求、事件时间线、增量拼接、证据卡片、拒答/错误和新会话。
2. 实现 POST SSE 解析与会话内历史。
3. 运行聊天相关前端测试和 `tests/test_api.py tests/test_agent.py`。

**成功标准：** UI 按服务端事件顺序展示执行过程，答案和证据不由前端伪造。

## 阶段 5：浏览器验收与 Streamlit 清理

**文件：**

- 删除 `app.py`、`pages/`、`.streamlit/`、`tests/test_pages.py`
- 修改 `bootstrap.py`、`requirements.txt`、`README.md`
- 添加 FastAPI 静态资源启动入口

1. 本地启动 FastAPI 与 Vite，浏览三页并检查布局、交互、响应式和控制台。
2. 浏览器验收通过后，删除 Streamlit 页面和依赖，移除 `st.cache_resource`。
3. 更新启动说明，验证生产构建可由 FastAPI 提供。
4. 运行后端全量 `pytest`、前端全量测试与生产构建。

**成功标准：** 仓库中不再存在 Streamlit 运行代码或依赖，新界面完整替代旧页面，所有自动化验证通过。
