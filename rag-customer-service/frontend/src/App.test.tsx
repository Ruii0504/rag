import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import App from "./App"


const statusPayload = {
  status: "ready",
  chat_model: "qwen-live-test",
  embedding_model: "embedding-live-test",
  retrieval_score_threshold: 0.72,
  knowledge_base_count: 1,
  document_count: 1,
  chunk_count: 7,
}

const knowledgeBasePayload = [
  {
    id: "kb-api",
    name: "API 产品知识",
    created_at: "2026-08-27T10:00:00+00:00",
    document_count: 1,
    chunk_count: 7,
  },
  {
    id: "kb-secondary",
    name: "备用知识库",
    created_at: "2026-08-27T10:01:00+00:00",
    document_count: 0,
    chunk_count: 0,
  },
]

const documentPayload = [
  {
    id: "doc-api",
    kb_id: "kb-api",
    filename: "来自API的说明.md",
    chunk_count: 7,
    status: "ready",
    created_at: "2026-08-27T10:00:00+00:00",
  },
]

function jsonResponse(payload: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(payload), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  )
}

function sseResponse(events: unknown[]) {
  const body = events.map(event => `data: ${JSON.stringify(event)}\n\n`).join("")
  return Promise.resolve(new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  }))
}

describe("应用外壳与知识库页面", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        if (url === "/api/chat/stream") {
          const request = JSON.parse(String(init?.body)) as { question: string }
          if (request.question === "价格和无线标准？") {
            return sseResponse([
              { type: "node_status", stage: "rewrite", message: "问题拆分完成：2 项", payload: {} },
              { type: "tool_status", stage: "retrieval", message: "候选检索完成：2 条", payload: {} },
              {
                type: "evidence",
                stage: "judge",
                message: "语义判定完成：1/2 项可回答",
                payload: {
                  count: 2,
                  result: {
                    query: "AX3000 的价格和无线标准是什么？",
                    has_sufficient_evidence: true,
                    reason: "",
                    evidences: [
                      {
                        reference_id: 1,
                        kb_id: "kb-api",
                        document_id: "doc-api",
                        document_name: "产品介绍.md",
                        heading_path: ["产品功能"],
                        chunk_index: 0,
                        content: "AX3000 采用双频设计。",
                        distance: 0.2,
                        similarity: 0.83,
                        subquestion_id: "q1",
                        subquestion: "AX3000 多少钱？",
                        support_status: "related",
                      },
                      {
                        reference_id: 2,
                        kb_id: "kb-api",
                        document_id: "doc-api",
                        document_name: "产品介绍.md",
                        heading_path: ["无线标准"],
                        chunk_index: 1,
                        content: "AX3000 支持 Wi-Fi 6。",
                        distance: 0.21,
                        similarity: 0.83,
                        subquestion_id: "q2",
                        subquestion: "AX3000 支持 Wi-Fi 6 吗？",
                        support_status: "supporting",
                      },
                    ],
                  },
                },
              },
              { type: "node_status", stage: "generate", message: "答案生成开始", payload: {} },
              { type: "node_status", stage: "validate", message: "答案证据校验开始", payload: {} },
              {
                type: "answer_delta",
                stage: "answer",
                message: "**AX3000 多少钱？**\n当前知识库找到相关资料，但缺少价格信息，因此无法确认。\n\n**AX3000 支持 Wi-Fi 6 吗？**\nAX3000 支持 Wi-Fi 6。[2]",
                payload: { verified_count: 1 },
              },
              { type: "completed", stage: "agent", message: "处理完成", payload: {} },
            ])
          }
          if (request.question === "无证据问题") {
            return sseResponse([
              { type: "node_status", stage: "judge", message: "证据不足", payload: {} },
              { type: "answer_delta", stage: "refuse", message: "抱歉，我没有找到足够证据。", payload: {} },
              { type: "completed", stage: "agent", message: "处理完成", payload: {} },
            ])
          }
          if (request.question === "触发错误") {
            return sseResponse([
              { type: "error", stage: "agent", message: "客服处理失败，请稍后重试", payload: {} },
            ])
          }
          return sseResponse([
            { type: "node_status", stage: "rewrite", message: "问题改写完成", payload: {} },
            { type: "tool_status", stage: "retrieval", message: "在 1 个知识库中检索", payload: { kb_ids: ["kb-api"] } },
            {
              type: "evidence",
              stage: "retrieval",
              message: "命中 1 条候选片段",
              payload: {
                count: 1,
                result: {
                  query: "如何恢复出厂？",
                  has_sufficient_evidence: true,
                  reason: "",
                  evidences: [{
                    reference_id: 1,
                    kb_id: "kb-api",
                    document_id: "doc-api",
                    document_name: "来自API的说明.md",
                    heading_path: ["使用与维护", "恢复出厂设置"],
                    chunk_index: 0,
                  content: "长按 Reset 按钮 8 秒。",
                  distance: 0.09,
                  similarity: 0.91,
                  subquestion_id: "q1",
                  subquestion: "如何恢复出厂？",
                  support_status: "supporting",
                }],
                },
              },
            },
            { type: "node_status", stage: "judge", message: "证据充分", payload: {} },
            { type: "answer_delta", stage: "answer", message: "长按 Reset ", payload: {} },
            { type: "answer_delta", stage: "answer", message: "8 秒。[1]", payload: {} },
            { type: "completed", stage: "agent", message: "处理完成", payload: {} },
          ])
        }
        if (url === "/api/status") return jsonResponse(statusPayload)
        if (url === "/api/knowledge-bases" && init?.method === "POST") {
          return jsonResponse(
            {
              ...knowledgeBasePayload[0],
              id: "kb-new",
              name: "新知识库",
              document_count: 0,
              chunk_count: 0,
            },
            201,
          )
        }
        if (url === "/api/knowledge-bases") return jsonResponse(knowledgeBasePayload)
        if (url === "/api/knowledge-bases/kb-api" && init?.method === "DELETE") {
          return Promise.resolve(new Response(null, { status: 204 }))
        }
        if (url === "/api/knowledge-bases/kb-api/documents" && init?.method === "POST") {
          return jsonResponse({
            documents: [{
              ...documentPayload[0],
              id: "doc-upload",
              filename: "新上传资料.md",
              progress: ["校验", "切分", "向量化", "保存", "完成"],
            }],
            errors: [{ filename: "失败.txt", detail: "仅支持 Markdown (.md) 文件" }],
          }, 207)
        }
        if (url === "/api/knowledge-bases/kb-api/documents/doc-api" && init?.method === "DELETE") {
          return Promise.resolve(new Response(null, { status: 204 }))
        }
        if (url === "/api/knowledge-bases/kb-api/documents") {
          return jsonResponse(documentPayload)
        }
        if (url === "/api/knowledge-bases/kb-secondary/documents") return jsonResponse([])
        if (url === "/api/knowledge-bases/kb-new/documents") return jsonResponse([])
        throw new Error(`未处理的测试请求：${url}`)
      }),
    )
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it("默认显示概览并使用状态 API 数据", async () => {
    render(<App />)

    expect(screen.getByRole("heading", { name: "每一次回答，都有据可循。" })).toBeTruthy()
    expect(await screen.findByText('"qwen-live-test"')).toBeTruthy()
    expect(screen.getByText("7 chunks indexed")).toBeTruthy()
  })

  it("启用重排时展示 Top-K 与独立重排分数，不显示旧阈值", async () => {
    vi.mocked(fetch).mockImplementation((input) => {
      if (String(input) === "/api/status") return jsonResponse({ ...statusPayload, rerank_model: "qwen3-rerank", retrieval_top_k: 7 })
      if (String(input) === "/api/knowledge-bases") return jsonResponse(knowledgeBasePayload)
      if (String(input) === "/api/chat/stream") return sseResponse([
        { type: "evidence", stage: "judge", message: "语义判定完成", payload: { result: { evidences: [{
          reference_id: 1, kb_id: "kb-api", document_id: "doc-api", document_name: "USB.md",
          chunk_index: 0, heading_path: ["USB"], content: "不支持打印机", similarity: 0.49,
          rerank_score: 0.96, support_status: "supporting", subquestion: "能接打印机吗？",
        }] } } },
        { type: "answer_delta", stage: "answer", message: "不支持。[1]", payload: {} },
        { type: "completed", stage: "agent", message: "处理完成", payload: {} },
      ])
      return jsonResponse(documentPayload)
    })
    render(<App />)
    await screen.findByText('"qwen-live-test"')
    fireEvent.click(screen.getByRole("button", { name: "智能客服" }))
    expect(screen.getByText("重排保留")).toBeTruthy()
    expect(screen.getByText("Top 7")).toBeTruthy()
    expect(screen.getByText("qwen3-rerank · 能否回答仍由语义判定确认")).toBeTruthy()
    expect(screen.queryByText("候选阈值")).toBeNull()
    fireEvent.change(screen.getByLabelText("客户问题"), { target: { value: "能接打印机吗？" } })
    fireEvent.click(screen.getByRole("button", { name: "发送问题" }))
    expect(await screen.findByText("重排 0.96")).toBeTruthy()
    expect(screen.getByText("向量 0.49")).toBeTruthy()
  })

  it("知识库页显示 API 返回的知识库与文档", async () => {
    render(<App />)
    fireEvent.click(screen.getByRole("button", { name: "知识库" }))

    const baseButton = await screen.findByRole("button", { name: /API 产品知识/ })
    expect(baseButton).toBeTruthy()
    fireEvent.click(baseButton)
    expect(await screen.findByText("来自API的说明.md")).toBeTruthy()
    expect(screen.queryByText("AX3000_产品说明.md")).toBeNull()
  })

  it("创建知识库后刷新 API 数据", async () => {
    render(<App />)
    fireEvent.click(screen.getByRole("button", { name: "知识库" }))
    await screen.findByRole("button", { name: /API 产品知识/ })
    fireEvent.click(screen.getByRole("button", { name: "新建知识库" }))
    fireEvent.change(screen.getByLabelText("知识库名称"), {
      target: { value: "新知识库" },
    })
    fireEvent.click(screen.getByRole("button", { name: "创建知识库" }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /新知识库/ })).toBeTruthy()
    })
    expect(fetch).toHaveBeenCalledWith(
      "/api/knowledge-bases",
      expect.objectContaining({ method: "POST" }),
    )
  })

  it("选择 Markdown 文件后上传并显示服务端结果", async () => {
    render(<App />)
    fireEvent.click(screen.getByRole("button", { name: "知识库" }))
    await screen.findByRole("button", { name: /API 产品知识/ })
    fireEvent.click(screen.getByRole("button", { name: "上传 Markdown" }))
    fireEvent.change(screen.getByLabelText("Markdown 文件"), {
      target: { files: [new File(["# 新资料"], "新上传资料.md", { type: "text/markdown" })] },
    })
    fireEvent.click(screen.getByRole("button", { name: "开始上传" }))

    expect(await screen.findByText("新上传资料.md")).toBeTruthy()
    expect(screen.getByText("1 份成功，1 份失败")).toBeTruthy()
    expect(fetch).toHaveBeenCalledWith(
      "/api/knowledge-bases/kb-api/documents",
      expect.objectContaining({ method: "POST", body: expect.any(FormData) }),
    )
  })

  it("确认后通过 API 删除当前知识库", async () => {
    render(<App />)
    fireEvent.click(screen.getByRole("button", { name: "知识库" }))
    await screen.findByRole("button", { name: /API 产品知识/ })
    fireEvent.click(screen.getByRole("button", { name: "删除" }))
    fireEvent.click(screen.getByRole("button", { name: "确认删除" }))

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /API 产品知识/ })).toBeNull()
    })
    expect(fetch).toHaveBeenCalledWith(
      "/api/knowledge-bases/kb-api",
      expect.objectContaining({ method: "DELETE" }),
    )
  })

  it("确认后删除单份文档并同步页面计数", async () => {
    render(<App />)
    fireEvent.click(screen.getByRole("button", { name: "知识库" }))
    await screen.findByText("来自API的说明.md")
    fireEvent.click(screen.getByRole("button", { name: "删除 来自API的说明.md" }))
    fireEvent.click(screen.getByRole("button", { name: "确认删除文档" }))

    await waitFor(() => {
      expect(screen.queryByText("来自API的说明.md")).toBeNull()
    })
    expect(fetch).toHaveBeenCalledWith(
      "/api/knowledge-bases/kb-api/documents/doc-api",
      expect.objectContaining({ method: "DELETE" }),
    )
  })

  it("知识库管理单选不会被客服多选状态污染", async () => {
    const { container } = render(<App />)
    await screen.findByText('"qwen-live-test"')
    fireEvent.click(screen.getByRole("button", { name: "智能客服" }))
    fireEvent.click(screen.getByRole("checkbox", { name: /备用知识库/ }))
    fireEvent.click(screen.getByRole("button", { name: "知识库" }))
    await screen.findByText("来自API的说明.md")

    expect(container.querySelectorAll(".base-row.selected")).toHaveLength(1)
  })

  it("按 SSE 事件顺序展示回答、过程与证据，并可新建会话", async () => {
    render(<App />)
    await screen.findByText('"qwen-live-test"')
    fireEvent.click(screen.getByRole("button", { name: "智能客服" }))
    fireEvent.change(screen.getByLabelText("客户问题"), {
      target: { value: "如何恢复出厂？" },
    })
    fireEvent.click(screen.getByRole("button", { name: "发送问题" }))

    expect(await screen.findByText("长按 Reset 8 秒。[1]")).toBeTruthy()
    expect(screen.getByText("问题改写完成")).toBeTruthy()
    expect(screen.getByText("证据充分")).toBeTruthy()
    expect(screen.getByRole("heading", { name: "来自API的说明.md" })).toBeTruthy()
    expect(screen.getByText("使用与维护 / 恢复出厂设置")).toBeTruthy()
    const streamCall = vi.mocked(fetch).mock.calls.find(([url]) => url === "/api/chat/stream")
    expect(JSON.parse(String(streamCall?.[1]?.body))).toEqual({
      question: "如何恢复出厂？",
      history: [],
      kb_ids: ["kb-api"],
    })

    fireEvent.click(screen.getByRole("button", { name: "新建会话" }))
    expect(screen.queryByText("长按 Reset 8 秒。[1]")).toBeNull()
    expect(screen.getByText("等待提问")).toBeTruthy()
  })

  it("多轮提问后保留并展示完整会话历史", async () => {
    render(<App />)
    await screen.findByText('"qwen-live-test"')
    fireEvent.click(screen.getByRole("button", { name: "智能客服" }))
    fireEvent.change(screen.getByLabelText("客户问题"), {
      target: { value: "第一轮问题" },
    })
    fireEvent.click(screen.getByRole("button", { name: "发送问题" }))
    await screen.findByText("长按 Reset 8 秒。[1]")
    fireEvent.change(screen.getByLabelText("客户问题"), {
      target: { value: "第二轮问题" },
    })
    fireEvent.click(screen.getByRole("button", { name: "发送问题" }))

    await waitFor(() => {
      expect(screen.getByText("第一轮问题")).toBeTruthy()
      expect(screen.getByText("第二轮问题")).toBeTruthy()
    })
    const streamCalls = vi.mocked(fetch).mock.calls.filter(([url]) => url === "/api/chat/stream")
    expect(JSON.parse(String(streamCalls[1]?.[1]?.body)).history).toEqual([
      { role: "user", content: "第一轮问题" },
      { role: "assistant", content: "长按 Reset 8 秒。[1]" },
    ])
  })

  it("同时展示局部回答、局部拒答和两类检索资料标签", async () => {
    render(<App />)
    await screen.findByText('"qwen-live-test"')
    fireEvent.click(screen.getByRole("button", { name: "智能客服" }))
    fireEvent.change(screen.getByLabelText("客户问题"), {
      target: { value: "价格和无线标准？" },
    })
    fireEvent.click(screen.getByRole("button", { name: "发送问题" }))

    expect(await screen.findByText(/缺少价格信息/)).toBeTruthy()
    expect(screen.getByText(/AX3000 支持 Wi-Fi 6。\[2]/)).toBeTruthy()
    expect(screen.getByText("回答证据")).toBeTruthy()
    expect(screen.getByText("相关资料，不足以直接回答")).toBeTruthy()
    expect(screen.getByText("答案证据校验开始")).toBeTruthy()
    expect(screen.getByText("回答完成 · 1 条引用")).toBeTruthy()
  })

  it("请求进行中创建新会话会取消旧流且不回写旧内容", async () => {
    render(<App />)
    await screen.findByText('"qwen-live-test"')
    let signal: AbortSignal | undefined
    vi.mocked(fetch).mockImplementation((_input, init) => {
      signal = init?.signal ?? undefined
      return new Promise((_resolve, reject) => {
        signal?.addEventListener("abort", () => reject(new DOMException("已取消", "AbortError")))
      })
    })
    fireEvent.click(screen.getByRole("button", { name: "智能客服" }))
    fireEvent.change(screen.getByLabelText("客户问题"), {
      target: { value: "不应回写的问题" },
    })
    fireEvent.click(screen.getByRole("button", { name: "发送问题" }))
    fireEvent.click(screen.getByRole("button", { name: "新建会话" }))

    expect(signal?.aborted).toBe(true)
    expect(screen.queryByText("不应回写的问题")).toBeNull()
    expect(screen.getByText("等待提问")).toBeTruthy()
  })

  it.each([
    ["无证据问题", "抱歉，我没有找到足够证据。", "证据不足 · 已拒答"],
    ["触发错误", "客服处理失败，请稍后重试", "处理失败"],
  ])("展示拒答或安全错误状态：%s", async (question, answer, state) => {
    render(<App />)
    await screen.findByText('"qwen-live-test"')
    fireEvent.click(screen.getByRole("button", { name: "智能客服" }))
    fireEvent.change(screen.getByLabelText("客户问题"), { target: { value: question } })
    fireEvent.click(screen.getByRole("button", { name: "发送问题" }))

    expect((await screen.findAllByText(answer)).length).toBeGreaterThan(0)
    expect(screen.getByText(state)).toBeTruthy()
  })
})
