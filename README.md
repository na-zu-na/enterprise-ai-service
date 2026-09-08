# Enterprise AI Service

`enterprise-ai-service` 是企业知识库 AI 服务，基于 FastAPI 提供文档解析、文本切块、向量化、混合检索、Reranker、RAG 问答和 LangGraph Agent。

本仓库只包含 Python AI 服务。用户认证、知识库权限、文档业务数据、任务管理和业务会话由外部 Spring 服务提供，Python 服务通过 HTTP 与其协作。

## 功能

- 解析 TXT 和 Markdown 文档。
- 按固定长度和 Markdown 章节进行文本切块。
- 使用 BGE-M3 生成 1024 维文本向量。
- 使用 PostgreSQL + pgvector 完成 Dense Retrieval。
- 使用 Elasticsearch 完成 BM25 Retrieval。
- 使用 RRF 融合 Dense 与 BM25 结果。
- 使用 BGE Reranker 重排并按阈值过滤候选。
- 使用 OpenAI 兼容接口生成 RAG 答案。
- 使用 LangGraph 编排知识查询、任务和日历工具。
- 使用 PostgreSQL Checkpointer 保存多轮对话和 HITL 中断状态。
- 使用 MCP 接入 Google Calendar。
- 使用 Langfuse 记录 Agent、LLM 和检索链路。
- 提供 Vector、Hybrid + RRF、Hybrid + RRF + Reranker 离线评测。

## 技术栈

| 分类 | 技术 |
|---|---|
| Web API | FastAPI、Uvicorn、Pydantic |
| Agent | LangChain、LangGraph、ToolNode、interrupt/resume |
| LLM | OpenAI Python SDK、`langchain-openai` |
| Embedding | FlagEmbedding、`BAAI/bge-m3` |
| Reranker | FlagEmbedding、`BAAI/bge-reranker-v2-m3` |
| 向量检索 | PostgreSQL、pgvector、SQLAlchemy |
| 关键词检索 | Elasticsearch BM25 |
| Checkpointer | Psycopg 3、LangGraph PostgreSQL Checkpointer |
| 外部工具 | MCP、Google Calendar MCP |
| 可观测性 | Langfuse、OpenTelemetry |
| 测试与评测 | unittest、人工标注数据集、Hit@K、Recall@K、MRR |

[`requirements.txt`](requirements.txt) 是完整环境冻结文件，其中包含部分当前主链路尚未使用的包，例如 Celery、Redis、Pandas、Datasets、Accelerate 和 PEFT。

## 项目结构

```text
enterprise-ai-service/
├── agents/                  # LangGraph Agent
│   ├── config/              # Agent 系统提示词
│   ├── schemas/             # Agent 工具参数模型
│   ├── tools/               # 知识、任务、时间等工具
│   ├── graph.py             # Graph 构建与工具注册
│   └── state.py             # Agent 状态与 reducer
├── api/
│   ├── routes/              # FastAPI 路由
│   └── exception_handlers.py
├── chunkers/                # TXT、Markdown 切块
├── clients/                 # Spring、Elasticsearch、MCP 客户端
├── core/                    # 配置和统一响应
├── db/                      # SQLAlchemy Session 和 Checkpointer
├── embeddings/              # BGE-M3 Embedding
├── evals/                   # 检索评测数据、指标和报告
├── llm/                     # OpenAI 兼容 LLM 客户端
├── models/                  # SQLAlchemy 数据模型
├── observability/           # Langfuse 客户端与上下文
├── parsers/                 # TXT、Markdown 解析器
├── rerankers/               # BGE Reranker
├── retrieval/               # BM25、RRF 和检索数据模型
├── schemas/                 # API 请求与响应模型
├── services/                # 文档、检索、RAG 等服务编排
├── tests/                   # 自动化测试
├── .env.example             # 环境变量模板
├── main.py                  # FastAPI 应用入口
└── requirements.txt         # Python 依赖冻结文件
```

## 核心流程

### 文档处理

```text
文档信息和本地文件路径
        ↓
TXT / Markdown Parser
        ↓
Chunker（默认 800 字符，重叠 100 字符）
        ↓
BGE-M3 Embedding
        ↓
返回 Chunk、metadata 和 1024 维向量
```

Python 服务不负责接收 multipart 文件或长期保存上传文件。调用方传入服务可访问的 `storagePath`，Python 读取文件并返回解析结果。

### 混合检索

```text
用户查询
  ├── PostgreSQL pgvector Dense Retrieval
  └── Elasticsearch BM25 Retrieval
                    ↓
                 RRF 融合
                    ↓
              BGE Reranker
                    ↓
                阈值过滤
                    ↓
             RAG / Agent Context
```

默认参数：

| 参数 | 默认值 |
|---|---:|
| Dense candidate K | 30 |
| BM25 candidate K | 30 |
| RRF Top K | 20 |
| RRF constant | 60 |
| Reranker Top K | 5 |
| Reranker threshold | 0.5 |

Reranker 推理失败时，在线服务回退到 RRF 排名；离线评测会直接失败，避免把降级结果误认为 Reranker 结果。

### Agent

```text
START → Agent/LLM → 是否调用工具？
                       ├── 否 → END
                       └── 是 → ToolNode → Agent/LLM
```

基础工具：

- `knowledge_search`：企业知识库检索。
- `query_tasks`：查询当前用户任务。
- `create_tasks`：经用户审批后创建任务。
- `get_time`：读取服务器当前时间。

Google Calendar MCP 在应用启动时动态提供日历查询和创建工具。创建任务及创建日历事件使用 LangGraph HITL：工具调用通过 `interrupt()` 暂停，用户确认后通过 `Command(resume=...)` 恢复。

Agent 使用 PostgreSQL Checkpointer 保存执行状态，thread ID 格式为：

```text
conversation:<conversation_id>
```

## 环境要求

- Python 3.10 或兼容版本。
- PostgreSQL，并安装 pgvector 扩展。
- Elasticsearch。
- 可访问的 OpenAI 兼容 LLM API。
- 可访问的外部 Spring 业务服务。
- Google Calendar MCP Python 解释器和 Server 脚本。
- 首次启动需要下载或预先缓存 BGE-M3 与 BGE Reranker 模型。

## 安装

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖中包含 PyTorch、Transformers 和 FlagEmbedding，安装及首次模型加载可能需要较长时间。

## 配置

从模板创建本地配置：

```powershell
Copy-Item .env.example .env
```

不要提交包含密钥、密码或访问令牌的 `.env`。

### LLM

| 变量 | 说明 |
|---|---|
| `LLM_API_KEY` | OpenAI 兼容服务 API Key |
| `LLM_BASE_URL` | OpenAI 兼容 API 地址 |
| `LLM_MODEL` | 模型名称 |

### PostgreSQL 与 Elasticsearch

| 变量 | 说明 |
|---|---|
| `DATABASE_URL` | SQLAlchemy/Psycopg PostgreSQL URL |
| `ELASTICSEARCH_URL` | Elasticsearch 地址 |
| `ELASTICSEARCH_INDEX` | Chunk 索引名，默认 `document_chunk` |
| `ELASTICSEARCH_API_KEY` | 可选 Elasticsearch API Key |
| `ELASTICSEARCH_CA_CERTS` | 可选 CA 证书路径 |

`DATABASE_URL` 示例：

```dotenv
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/enterprise_agent
```

数据库中的 `document_chunk.embedding` 必须是 `vector(1024)`。

### Reranker

| 变量 | 说明 |
|---|---|
| `RERANKER_MODEL` | Reranker 模型名称 |
| `RERANKER_USE_FP16` | 是否启用 FP16 |
| `RERANKER_QUERY_MAX_LENGTH` | Query 最大长度 |
| `RERANKER_PASSAGE_MAX_LENGTH` | Passage 最大长度 |
| `RERANKER_MIN_SCORE` | 最终候选最低分数 |

### 外部业务服务

| 变量 | 说明 |
|---|---|
| `SPRING_REQUEST_URL` | 外部 Spring 服务地址 |
| `SPRING_KNOWLEDGE_BASE_IDS_PATH` | 查询当前用户可访问知识库 ID 的路径 |
| `SPRING_USER_NAME` | 预留的服务用户名配置 |
| `SPRING_USER_PASSWORD` | 预留的服务密码配置 |

请求外部 Spring 服务时，Python 会透传用户 Bearer Token。Agent 内部调用还使用由可信上游传递的 `X-Authenticated-User-Id`。

### Google Calendar MCP

| 变量 | 说明 |
|---|---|
| `GOOGLE_CALENDAR_MCP_PYTHON` | 运行 MCP Server 的 Python 路径 |
| `GOOGLE_CALENDAR_MCP_SERVER` | MCP Server 脚本路径 |

这两个变量缺失时，当前应用会在启动阶段失败。

### Langfuse

| 变量 | 说明 |
|---|---|
| `LANGFUSE_ENABLED` | 是否启用观测 |
| `LANGFUSE_PUBLIC_KEY` | Langfuse Public Key |
| `LANGFUSE_SECRET_KEY` | Langfuse Secret Key |
| `LANGFUSE_BASE_URL` | Langfuse 服务地址 |
| `LANGFUSE_ENVIRONMENT` | 环境名称 |
| `LANGFUSE_RELEASE` | Release 标识 |
| `LANGFUSE_SAMPLE_RATE` | 采样率 |

未启用 Langfuse，或启用后没有配置密钥时，业务仍可运行，观测会被关闭。

## 启动

确保 PostgreSQL、Elasticsearch、外部 Spring 服务和 Google Calendar MCP 配置可用，然后执行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

开发模式：

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

启动后可访问：

- Swagger UI：`http://localhost:8000/docs`
- OpenAPI JSON：`http://localhost:8000/openapi.json`

## API

成功响应统一使用：

```json
{
  "code": 200,
  "message": "success",
  "data": {}
}
```

| Method | Endpoint | 用途 |
|---|---|---|
| POST | `/document/parse` | 解析、切块并向量化文档 |
| POST | `/embedding/documents` | 批量生成 Chunk Embedding |
| POST | `/embedding/retrieval` | 仅执行 Dense Retrieval |
| POST | `/rag/query` | 完整检索、重排并生成 RAG 答案 |
| GET | `/knowledge-bases/accessible` | 查询当前用户可访问的知识库 ID |
| POST | `/retrieval/retrieve` | Dense + BM25 + RRF，不执行 Reranker |
| GET | `/chunks/{chunk_id}` | 查询完整 Chunk 和 metadata |
| GET | `/documents/{document_id}` | 查询文档 metadata |
| POST | `/agent/chat` | 调用 Production Agent |
| POST | `/agent/approvals/respond` | 恢复 HITL 审批流程 |

权限相关接口需要：

```http
Authorization: Bearer <access-token>
```

`/agent/chat` 和 `/agent/approvals/respond` 还需要可信业务服务传入：

```http
X-Authenticated-User-Id: <user-id>
```

检索接口的详细请求和响应格式见 [`docs/retrieval-api.md`](docs/retrieval-api.md)。

### Agent 请求示例

```http
POST /agent/chat
Authorization: Bearer <access-token>
X-Authenticated-User-Id: 1
Content-Type: application/json
```

```json
{
  "message": "请查询公司的年假申请规则",
  "knowledge_base_ids": [1, 2],
  "conversation_id": 1001
}
```

当返回 `status=approval_required` 时，使用响应中的 `approval.interrupt_id` 恢复：

```json
{
  "conversation_id": 1001,
  "interrupt_id": "<interrupt-id>",
  "approved": true
}
```

## 测试

运行全部测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

部分测试会导入 Embedding、Reranker 或数据库模块，需要完整依赖和有效的 `DATABASE_URL`。

## 检索评测

默认评测数据位于 `evals/datasets/retrieval_eval.json`，运行：

```powershell
.\.venv\Scripts\python.exe -m evals.retrieval_eval
```

指定参数：

```powershell
.\.venv\Scripts\python.exe -m evals.retrieval_eval `
  --dataset evals/datasets/retrieval_eval.json `
  --candidate-k 30 `
  --rrf-top-k 20 `
  --top-k 5 `
  --output evals/results/retrieval_eval_results.json
```

评测比较 Vector Only、Hybrid + RRF 和 Hybrid + RRF + Reranker，并输出 Hit@K、Recall@K、MRR 以及分类结果。完整说明见 [`evals/README.md`](evals/README.md)。

## 注意事项

- FastAPI 服务与外部业务服务必须连接到一致的知识库和文档数据。
- PostgreSQL Chunk 与 Elasticsearch 索引需要保持同步，否则 Dense 和 BM25 结果可能不一致。
- BGE-M3 在模块初始化阶段加载，启动时间和内存占用会高于普通 Web 服务。
- BGE Reranker 在首次需要重排时延迟加载。
- `get_time` 读取服务器真实本地时间，目前没有固定时钟注入能力。
- `/agent/*` 更适合作为内部服务接口；对外调用建议经过负责认证、权限和业务会话的上游服务。
