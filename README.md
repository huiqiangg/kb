# 知识问答 Agent

基于 FastAPI、`langchain-openai` 和 LLMOps 知识库的流式 RAG 服务。接口为 `POST /api/kb/chat/completions`，根路径提供简单的调试网页，Swagger 位于 `/docs`。

## 设计取向：不过度封装，直接实现对应功能

判断标准是「**能一眼读完、改一处就生效**」优先于「层次整齐」，宁可长一点、直白一点：

- **同形状的重复就地写第二遍**，不为复用抽基类或公共工具模块 —— `ChatOpenAI(...)` 的组装在
  `RagAgent._ensure_rewrite_model()` 与 `_ensure_answer_model()` 各写一遍（六行构造参数重复两次，
  比多一层包装好维护）；`_stream_tokens()` 同时服务 FAQ 润色与最终答案生成，但不会为此再包一个
  `ChatStreamAgent` 类。
- **不为「以后可能换实现」预留接口 / 工厂 / 包装层**：`chat_model.py`、`stream_agent.py`、
  以及改写侧的 `QueryRewriteAgent` + `create_agent`（不带工具的纯包装，模型 `ainvoke` 一行就够）
  都已删除，不要再加回来。
- **不加「可能有用」的配置开关与上限**：如「术语映射开关」（`DB_ENABLED` 已覆盖离线语义）、
  每环节的历史轮次上限（全量传入）。
- **只有真正需要收敛的顺序 / 规则才抽成一个方法**（FAQ 直返的完整事件序列
  `RagAgent._faq_direct`、召回的空间分组与降级 `KnowledgeBaseClient._retrieve`）——
  抽的是**语义**，不是代码形状。

代码量很小（`app/` 约 1100 行），按「入口 → 编排 → 外部依赖」三块读即可：
`app/main.py` + `app/routers/chat.py`（HTTP/SSE）→ `app/services/rag_agent.py`（整条问答链路的编排）
→ `knowledge_base.py` / `repository.py` / `rewrite.py` / `prompting.py`（外部依赖与纯函数）。

## 启动

```bash
cp .env.example .env
python3.12 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/uvicorn app.main:app --reload
```

必须配置 `KB_PLATFORM_BASE_URL`、`KB_PROJECT_ID` 和银行环境需要的鉴权信息。模型密钥请放入 `MODEL_ACCESS_KEY`，不要提交 `.env`。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `KB_PLATFORM_BASE_URL` | `http://127.0.0.1:8080` | LLMOps 平台根地址 |
| `KB_PROJECT_ID` | `assets` | 全局兜底 `project_id`（空间）查询参数；`ranges` 自带 `project_id` 时以 range 的为准 |
| `KB_TENANT_ID` | 空 | 全局兜底 `tenantId`（租户）查询参数，留空则不拼该参数 |
| `KB_AUTHORIZATION` | 空 | 召回接口鉴权，留空则不拼 `Authorization` 头 |
| `KB_MIX_RETRIEVE_URL` | 空 | **跨库召回**接口完整地址（最终答案检索用）；留空按 `{KB_PLATFORM_BASE_URL}/applet/api/v1/knowlhub/kbs:mix-retrieve` 推导 |
| `KB_FAQ_RETRIEVE_URL` | 空 | **FAQ 探测**接口完整地址；留空复用 `KB_MIX_RETRIEVE_URL` |

**接口 IP 与路径尚未最终确定**，所以召回地址能整条覆盖：定下地址后只改环境变量（或 `.env`），不用动代码。覆盖时 `project_id` / `tenantId` 仍会作为查询参数拼到该地址上。

FAQ 探测与最终答案检索是**同一个接口的两处部署**（请求体与响应结构完全一致），但 FAQ 知识库不保证挂在同一个网关地址上，因此两个地址各自可整条覆盖；两通道同地址时只配 `KB_MIX_RETRIEVE_URL` 即可。

### 空间与知识库层级

平台侧层级是「租户(`tenantId`) → 空间(`project_id`) → 知识库」，**知识库不支持跨空间检索**。
门户在每条 `ranges` 项里带上这三个信息：

| 字段 | 含义 |
| --- | --- |
| `project_id` | 空间 id |
| `tenantId` / `tenant_id` | 租户 id（两种写法都收） |
| `knowledgeType` / `knowledge_type` | 知识库类型：`1` 切片库，`2` 标准问答（QA）库；**只有显式传 `2` 才算 QA 库**，不传即不参与 FAQ 直返 |

`project_id` / `tenantId` 是**空间级**查询参数而不是全局配置，而 FAQ 探测与最终答案检索走的是
**同一个召回客户端**，所以「按 `(project_id, tenantId)` 把 ranges 分组、每组各发一次跨库检索」
这条规则只写一遍（`KnowledgeBaseClient._retrieve`）：**两条通道都按 range 自带的空间走**，
组内多个知识库共享一次请求与一次重排，组与组之间并发。
range 不带这两个字段时回落 `KB_PROJECT_ID` / `KB_TENANT_ID`，老请求行为不变。

FAQ 直返只对**显式标注 `knowledgeType=2`** 的 range 生效：切片库（1）与未标类型的 range 都不会触发
FAQ 探测，直接进入改写与跨库检索 —— 「没标类型」不等于「是 QA 库」，放宽会让切片库白跑一次永远不命中的召回。

## 数据库

术语映射与提示词配置存放在 MySQL。`db.sql` 只有 `CREATE TABLE` + `INSERT`，**不含建库与清表语句**：

```bash
# 首次：先建库，再整份执行
mysql -h127.0.0.1 -uroot -p -e "CREATE DATABASE IF NOT EXISTS kb DEFAULT CHARSET utf8mb4"
mysql -h127.0.0.1 -uroot -p kb < db.sql

# 已有库想按 db.sql 重来：备份 -> 删表 -> 重建
mysqldump -h127.0.0.1 -uroot -p --databases kb > /tmp/kb_backup_$(date +%Y%m%d_%H%M%S).sql
mysql -h127.0.0.1 -uroot -p -e "DROP TABLE IF EXISTS kb.rag_keywords_mapping, kb.rag_prompt"
mysql -h127.0.0.1 -uroot -p kb < db.sql
```

重建后应为 **47 条术语 + 3 条提示词**。

| 表 | 用途 |
| --- | --- |
| `rag_keywords_mapping` | 口语术语 -> 银行标准术语，问答前对 query 做替换 |
| `rag_prompt` | 各环节提示词，按 `prompt_key` 取用；每条拆成 `system_content`（固定指令，不含变量）+ `user_content`（含变量的正文） |

**提示词以 `db.sql` 为准**（库里现存内容仅作参考）。改提示词请改 `db.sql`，再同步进库：

```bash
.venv/bin/python scripts/sync_rag_prompt.py --dry-run   # 只预览差异
.venv/bin/python scripts/sync_rag_prompt.py             # 预览后确认写入
```

同步工具只替换 `db.sql` 中出现的 `prompt_key`，库中多出来的记录保持原样并给出提示；写入后会读回与 `db.sql` 逐字符比对，确认没有被二次转义。老库（单列 `prompt_content`）会先被就地迁移成 `system_content` + `user_content` 再写入。

**别用 `mysql` 命令行手动拼、或截取 `db.sql` 的片段执行**：提示词正文含大量 `\n` / `\"` 转义，手写命令行会在长字符串处断开（报 `ERROR 1064 ... near '' at line N`），且每条记录只是 `( ... )` 的 values 元组、缺 `INSERT INTO ... VALUES` 前缀。**整份脚本经管道执行没问题**（转义会正确还原、中文不乱码），出问题的只是「截一小段手拼」。只想同步某几条提示词时用上面的 `sync_rag_prompt.py`。

配置表的读取层（`app/services/repository.py`）一律 `.mappings()` 之后**按列名取值**，不用 `for a, b, c in rows` 这类位置解包 —— SELECT 的列顺序一调整（或中间插一列），位置解包会**静默把两列读串**：不报错、不 warning，直到模型拿到的提示词少了变量才会发现。回归脚本：

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_repository_loading.py
```

相关配置项（见 `.env.example`）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DB_ENABLED` | `true` | 关闭后不连库，全部走内置实现 |
| `DB_HOST` / `DB_PORT` / `DB_USER` / `DB_PASSWORD` / `DB_NAME` | `127.0.0.1` / `3306` / `root` / 空 / `kb` | 连接信息 |
| `DB_CACHE_TTL_SECONDS` | `60` | 两张表的进程内缓存时长，改库后最多等这么久生效 |
| `PROMPT_KEY_QUERY_UNDERSTANDING` | `query_understanding` | 改写环节取哪条提示词 |
| `PROMPT_KEY_ANSWER_SUMMARY` | `answer_summary` | 最终答案生成取哪条提示词 |
| `PROMPT_KEY_ANSWER_POLISH` | `answer_polish` | FAQ 直返答案润色取哪条提示词 |

三个 `PROMPT_KEY_*` 对应 `db.sql` 中 `rag_prompt` 的记录。每条提示词拆成两段：`system_content` 是固定的角色与规则（**不含变量**，原样作为 system 消息），`user_content` 是含占位符的正文（渲染后作为 user 消息）。三个场景各有独立的装配函数：

| 场景 | prompt key | user 段占位符 | 装配函数 | 对话历史 |
| --- | --- | --- | --- | --- |
| 改写 | `query_understanding` | `{query}` | `build_rewrite_messages` | **带**（提示词要求结合上下文改写） |
| 最终答案生成 | `answer_summary` | `{query}` + `{ragkm}` | `build_answer_messages` | 不带（单轮） |
| FAQ 直返润色 | `answer_polish` | `{query}` + `{answer}` | `build_polish_messages` | 不带（单轮） |

1. **改写** —— 对术语映射后的 query 做意图识别、改写与关键词提取；
2. **生成** —— 用检索资料生成最终答案，**一次调用直接流式下发**（不再有独立的润色步骤），检索资料注入 `{ragkm}`；
3. **FAQ 直返润色** —— 仅 FAQ 高置信命中时走，库中标准答案注入 `{answer}`。

生成与 FAQ 润色是**单轮**调用：`[system, user]` 两条消息，`user` 里同时带着问题和资料。改写要「结合上下文」，因此额外带上对话历史，消息形如 `[system, user, assistant, …, user]`，当前问题只出现在最后那条 user 里（`system_content` 按定义不含变量，也不做占位符替换——里面若有 JSON 示例花括号会原样保留）。

> **占位符写在库里的 `user_content` 里，改提示词时只改 `db.sql`**：注入哪个占位符由各场景的装配函数决定（如生成场景固定注入 `{ragkm}`）。若把库里的占位符名改掉（例如 `{ragkm}` → `{context}`），必须同步改名对应的装配函数与内置兜底提示词，否则那个 `{context}` 会**原样发给模型**——`fill()` 只替换声明过且有值的占位符，不做静默剔除。
>
> 同理，`user_content` 里漏写 `{query}` 时问题不会被带出来（装配层不自动补到末尾，免得提示词和实际输入对不上）；`user_content` 为空的行会被跳过并回落内置提示词，避免发一条残缺消息出去。
>
> `fill()` 同时认 `{name}` 与 `{{name}}` 两种花括号写法（`db.sql` 现行版本用的是双花括号）。

装配规则的回归脚本：

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_message_assembly.py
```

改写环节走 **OpenAI 兼容协议**（`ChatOpenAI`）**非流式一次调用**（`ainvoke`）同时产出改写与关键词，输出约定为纯 JSON 文本：

```json
{"rewritten_query": "数字人民币单日转账限额", "keywords": ["数字人民币", "单日", "转账限额"]}
```

`keywords` 替代了原来单独的「标签提取」模型调用，SSE 的 `tags_extracted` 事件数据来自这里。解析做容错（兼容旧字段 `query`/`tags`），调用失败或输出不可解析时回落原始 query 继续检索，不中断问答。JSON 的解析契约在 `app/services/rewrite.py`（纯函数，没有模型客户端）。

改写模型**惰性创建**：`RagAgent` 构造时不会初始化 `ChatOpenAI`，FAQ 首轮直返等走不到改写环节的请求也不会触发创建，只有真正进入改写才建（`RagAgent._ensure_rewrite_model`）。调用是非流式的，因此改写阶段不再下发 `rewrite_token` 增量事件。

模型客户端**就地组装，没有公共包装层**：`ChatOpenAI(...)` 只在两个用到它的地方各写一遍 —— 改写在 `RagAgent._ensure_rewrite_model()`，回答在 `RagAgent._ensure_answer_model()`（FAQ 润色与最终答案生成共用同一个实例）。组装时把 `/chat/completions` 后缀去掉（openai 客户端会自己补），鉴权用 `MODEL_ACCESS_KEY`：`api_key` 位置与真实鉴权头 `accessKey` 都用它。模型配置：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `REWRITE_MODEL_URL` / `REWRITE_MODEL_NAME` | 见 `.env.example` | 改写模型地址与模型名 |
| `ANSWER_MODEL_URL` / `ANSWER_MODEL_NAME` | 空 / `Qwen3-32B` | 答案模型（流式生成 + FAQ 润色）地址与模型名 |
| `MODEL_ACCESS_KEY` | 空 | 模型鉴权：Bearer 与 `accessKey` 头都用它；留空时 `api_key` 位置给占位串 |

**每个模型各配一条完整地址（含 `/chat/completions`），互不覆盖** —— 不做「全局 base + 局部覆盖」那种双层配置：少一层优先级要记，也不会出现全局变量把某个模型的专用路径（如答案模型带模型 uuid 的 `/api/model/{uuid}/chat/completions`）悄悄顶掉、还不报错的情况。两个模型的地址留空时取用都会抛错（否则 `base_url` 为空会静默打到 api.openai.com，而两侧调用失败都只吞成一条 warning，很难发现）。地址与密钥的组装有独立回归：`scripts/verify_model_config.py`。

生成与 FAQ 润色都**直接用 `ChatOpenAI` 流式调用**（`astream`，不套 agent）：两者都是「喂 messages、要一段流式文本」，用的是同一个回答模型，差别只在提示词，因此共用 `RagAgent._stream_tokens()`，由调用方决定装配哪条提示词。RAG 链路的生成一步到位（取 `answer_summary`，资料以 `{ragkm}` 注入 user 段），生成即最终答案，不再润色 / 总结；FAQ 高置信命中则用 `answer_polish` 润色后流式下发，只装配 `{query}` 与 `{answer}`。

该模型同样**惰性创建**：`RagAgent` 构造时不会初始化 `ChatOpenAI`，检索无结果等走不到生成的请求也不会触发创建。

**降级策略**：数据库未启用或连接失败时不阻断问答——术语映射退化为不替换，提示词回落代码内置版本（见 `app/services/prompting.py`），日志会打一条 warning。若取不到对应 `prompt_key`，或该行的 `user_content` 为空，生成环节回落内置答案提示词，保证模型仍能拿到检索资料。生成或润色调用失败时一个字都不吐，由调用方降级：RAG 链路发一句「未能生成答案，请稍后重试。」，FAQ 直返回落库中原文。

术语替换采用「最长优先 + 单次扫描」，不会出现替换结果被二次替换的级联问题；ASCII 术语（如 `LPR`、`ATM`）大小写不敏感。

## 检索链路

1. **FAQ 直返探测**：先按 `knowledgeType=2` 把 `ranges` 收窄到**标准问答库**（切片库里没有可直接回答的 QA 对，探它只会白跑），再按 `(project_id, tenantId)` 分组，每轮对两个 query 变体（原问题、术语映射后的问题）各发一次**跨库检索** `kbs:mix-retrieve`。最高分 ≥ `FAQ_SIMILARITY_THRESHOLD` 时直接返回库中标准答案（`chunk.qa_pairs[].answer` 优先于 `chunk.content`，经 `answer_polish` 润色后流式下发，`answer_type` 为 `faq`）；改写成功后再探一轮，命中则 `answer_type` 为 `faq_rewrite`。命中时还会把库里**其他相近问题**（同一次召回的 `chunk.qa_pairs[].question`，排除作答那条）凑满 3 条、在 token 流结束后作为一条 `related_queries` 事件下发。
2. **跨库混合检索**：未命中 FAQ 时先改写 query，随后调用**跨库召回接口** `kbs:mix-retrieve`——同一空间内的全部知识库由一次请求覆盖，结果由平台统一重排（跨空间时按空间拆成多次，见上文）：

```json
{
  "query": "个人住房贷款可以提前还款吗",
  "keywords": ["房贷", "提前还款"],
  "ranges": [{"knowledge_base_id": "37c5dwtg4ufbw332", "doc_range": ["ds3523"]}],
  "disable_rerank": false,
  "rerank_params": {"top_k": 12, "score_threshold": 0, "weight_type": 1, "full_text_rerank_weight": 0.4}
}
```

`query` 用改写结果（改写失败则回落到术语映射结果），`keywords` 是改写同一次调用产出的业务关键词，`ranges` 直接来自请求体。跨库时各知识库相似度度量体系不一致，接口要求重排必须开启，因此固定 `disable_rerank=false` 并携带 `rerank_params`（`weight_type=1` 为动态权重 WRRF，按向量/全文命中名次加权，无需再传 rerank 模型对象）。接口地址取 `KB_MIX_RETRIEVE_URL`（留空则按 `KB_PLATFORM_BASE_URL` 推导），地址定下来前不必改代码。

响应取 `result[].chunk.content` 作为正文、`result[].score` 作为得分，另有 `doc_id` / `doc_name` / `knowledge_base_id`，按分数截取 `FINAL_CONTEXT_TOP_K` 条拼成带编号的上下文交给模型。**标准问答库的答案优先取 `chunk.qa_pairs[].answer`**，其次才是 `chunk.content`。检索失败（含知识库不可达、非 2xx）只记一条 warning 并降级为「未检索到内容」（`answer_type` 为 `no_context`），不让检索故障把整条问答链路带崩。

> ⚠️ **`FAQ_SIMILARITY_THRESHOLD` 需要按真实数据重新标定**。它原本是按**单库相似度**调的（默认 0.98），
> 现在 FAQ 也走跨库重排（`weight_type=1` 的动态权重 WRRF），分数量纲已经不同。阈值不合适会表现为
> 「FAQ 永远不直返」；`RagAgent._faq_match` 会在最高分没过阈值时打一条 `INFO` 日志带上实际分数，
> 照日志回调 `FAQ_SIMILARITY_THRESHOLD` 即可。

3. **单步生成最终答案**：`answer_generate` 阶段把资料拼成参考上下文，用 `answer_summary` 提示词（留空回落内置 `DEFAULT_SUMMARY_PROMPT`）经 `ChatOpenAI` 流式调用一次产出最终答案，资料注入该提示词 user 段的 `{ragkm}` 占位符，`token` 事件逐字下发后发 `done`。不再有 `answer_polish` 事件。

对应的回归脚本（进程内 mock 出网，覆盖正常链路、提示词 key、FAQ 直返与相近问题、FAQ 未达阈值、
range 自带空间、只有切片库、未标类型不探 FAQ、跨空间分组（**两条通道都断言**）、检索失败、
生成失败、租户未配置、两个召回地址各自覆盖、以及**用真 `ChatOpenAI` 跑一遍流式**共十六个场景）：

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_mix_retrieve.py
```

## 请求示例

```json
{
  "messages": [{"role": "user", "content": "个人贷款提前还款怎么办理？"}],
  "ranges": [
    {
      "knowledge_base_id": "37c5dwtg4ufbw332",
      "doc_range": ["ds3523"],
      "project_id": "space-a",
      "tenantId": "tenant-001",
      "knowledgeType": 2
    }
  ]
}
```

`ranges` 里只有 `knowledge_base_id` 必填；`project_id` / `tenantId` / `knowledgeType` 由门户带上，
用于定位空间与判断是否标准问答库。都不传时回落全局 `KB_PROJECT_ID` / `KB_TENANT_ID`；
`knowledgeType` 必须是 `2` 才会走 FAQ 直返，不传或传 `1` 一律跳过 FAQ 探测。

响应为 SSE：`status` 表示处理阶段、`token` 为最终答案增量、`sources` 为检索来源、`related_queries` 为其他相近问题、`done` 表示结束。FAQ 高置信命中的 `done` 事件 `answer_type` 为 `faq`（首轮命中）或 `faq_rewrite`（改写后命中），其答案经润色后流式下发；润色调用失败时回落库中原文。

FAQ 直返时，**等 token 流全部下发完**（`done` 之前）会补发**一条** `related_queries` 事件，三条一次带走：

```
event: related_queries
data: {"queries":["提前还款需要预约吗","提前还款收违约金吗","线上能办提前还款吗"]}
```

内容是本次召回里**除选中作答那条以外**的其他 QA 问题原文，按相似度排序取 3 条 —— 数组顺序即相似度序。**凑不满 3 条就整段不发**（不会出现 1~2 条的情况），未命中 FAQ 的普通检索回答也不发该事件。

## 本地联调（mock 召回网关）

知识库 / 模型网关在联调环境常常不可达，用 `scripts/mock_kb_gateway.py` 喂假数据就能跑通整条链路，**不用改一行代码**（零依赖，标准库 `http.server`）：

```bash
# 终端 1：假召回网关（FAQ 通道默认返回一条 0.99 的 QA 命中 + 3 条相近问题）
.venv/bin/python scripts/mock_kb_gateway.py

# 终端 2：主服务把两个召回地址指过去（DB_ENABLED=false 可顺带绕开数据库）
DB_ENABLED=false \
KB_FAQ_RETRIEVE_URL=http://127.0.0.1:8099/applet/api/v1/knowlhub/kbs:mix-retrieve \
KB_MIX_RETRIEVE_URL=http://127.0.0.1:8099/applet/api/v1/knowlhub/kbs:mix-retrieve \
.venv/bin/uvicorn app.main:app --port 8000

# 终端 3：调接口（-N 关掉 curl 缓冲，否则看不到流式）
curl -N -X POST http://127.0.0.1:8000/api/kb/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"房贷能提前还款吗"}],
       "ranges":[{"knowledge_base_id":"kb-qa-001","doc_range":["ds3523"],"project_id":"assets","tenantId":"tenant-001","knowledgeType":2}]}'
```

浏览器打开 `http://127.0.0.1:8000/` 是内置调试页，比 curl 更直观。

mock 网关按请求体 `keywords` 是否为空区分通道（空 = FAQ 探测、非空 = 最终答案检索）。但两条通道的请求体形状本来完全一样，**改写失败时会话 keywords 恒空**，所以精确验证某条通道要显式设 `MOCK_CHANNEL`：

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `MOCK_PORT` | `8099` | 监听端口 |
| `MOCK_CHANNEL` | `auto` | `auto` / `faq` / `chunk` |
| `MOCK_FAQ_SCORE` | `0.99` | 作答那条的分数；设 `0.5` 可测「未达阈值 → 走完整链路」 |
| `MOCK_RELATED_COUNT` | `3` | 相近问题候选条数；设 `2` 可测「凑不满三条 → 一条都不发」 |

FAQ 命中后的答案会走 `answer_polish` 润色，**模型不可达时自动回落库中原文**（按 12 字切块下发），所以没有模型网关也能验证「FAQ 直返 + `related_queries`」的完整事件序列。
