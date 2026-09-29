# 项目长期记忆 — bank/kb（农商行知识问答 Agent）

## 技术栈
- 后端：FastAPI + LangChain 1.2，SSE 流式 RAG
- 接口：`POST /api/kb/chat/completions`，根路径为调试网页，`/docs` 为 Swagger
- 配置：`app/core/config.py`（pydantic-settings），读 `.env`

## 数据库约定
- 通过 OrbStack 容器 `intent-mysql`（mysql:8.0）提供，未单独建容器。
- 连接：`127.0.0.1:3306`，用户 `root`，密码 `19921112kxw`，库 `kb`（utf8mb4）。
- 表：`rag_keywords_mapping`（术语映射）、`rag_prompt`（提示词配置，`system_content` +
  `user_content` 两列，2026-09-28 由单列 `prompt_content` 迁移而来）。
- 初始化脚本：`db.sql`（**无 `CREATE DATABASE`/`DROP TABLE`，需先建库并清空**）。
- **整库重建（2026-09-28 实测可行）**：
  ```bash
  docker exec intent-mysql mysqldump -uroot -p'***' --databases kb > /tmp/kb_backup_<ts>.sql
  docker exec intent-mysql mysql -uroot -p'***' -e "DROP TABLE IF EXISTS kb.rag_keywords_mapping, kb.rag_prompt;"
  docker exec -i intent-mysql mysql -uroot -p'***' --default-character-set=utf8mb4 kb < db.sql
  ```
  整份文件经 stdin 管道喂给 mysql 是**可行的**（`\n` / `\"` 都会按预期转义，中文不乱码）；
  重建后应得 **47 条术语 + 3 条提示词**、无重复术语。核对用
  `sync_rag_prompt.py --dry-run`（提示词）+ 应用层连接读一遍（术语）。
- 访问方式：SQLAlchemy async + aiomysql，惰性建池，`pool_pre_ping`。
- **读表一律 `.mappings()` 后按列名取值**（`_text_column(row, "system_content")`），
  **不要写 `for key, system, user in rows` 这类位置解包** —— 用户 2026-09-28 指出这种写法
  「不健壮」。位置解包的失败是**静默**的：SELECT 列顺序一调整就把两列读串（不报错不 warning），
  多一列则报 `ValueError` 且报错点离原因很远。按列名取 → 列名写错直接 `KeyError`，即刻可见。
  `scripts/sync_rag_prompt.py` 里的 `row[0]` / `tuple(stored)` 也一并改成了 mappings 写法。

## 配置表读取策略
- 术语映射与提示词**从 DB 读取**，进程内 TTL 缓存 60s（`DB_CACHE_TTL_SECONDS`），
  库不可用或查不到记录时**回落代码内置**，不阻断问答。
- `DB_ENABLED=false` 可整体绕开数据库（离线/单测用）。
- **缓存并发语义（`_TtlStore`，2026-09-28）**：同一 repository 的并发 `get()` **只触发一次
  loader、只查一次库** —— 未命中时抢 `_lock`，锁内双重检查，只有第一个协程真正加载；
  且 `_load()` 一次读整张表，所以 `asyncio.gather(prompts.get(k1), get(k2), get(k3))`
  最终就是一条 `SELECT`。**前提是共用 store**：术语表与提示词表是两个 store，跨表 gather 会各开一个连接。
  失败窗口是独立字段 `_retry_after`（**不能复用 `_expires_at`**：加载失败时 `_value` 仍是
  `None`，双重检查恒假，静默窗口会失效，导致冷启动+库故障时并发调用串行各试一次、
  把首请求卡到多次 `connect_timeout` 叠加）。`invalidate()` 同时清 `_retry_after`。
- 提示词按环节配置 key，**与 `db.sql` 中 `rag_prompt` 的三条记录一一对应**（用户要求字段名与 DB key 同名）：
  `PROMPT_KEY_QUERY_UNDERSTANDING=query_understanding`（**只用于改写**）、
  `PROMPT_KEY_ANSWER_SUMMARY=answer_summary`（**最终答案生成**）、
  `PROMPT_KEY_ANSWER_POLISH=answer_polish`（FAQ 直返润色）。
- 回答链路（2026-09-28 改版后）：**FAQ 直返探测 → 改写 → 跨库混合检索 → 单步生成最终答案（流式）**。
  生成一步到位、即最终答案，**已去掉润色与总结两步**；生成提示词取 `answer_summary`，
  检索资料注入它的 **`{ragkm}`** 占位符（2026-09-28 用户重写该行时由 `{answer}` 改名而来）。
  `DEFAULT_SUMMARY_PROMPT` 是生成环节的内置兜底，占位符必须与库中保持一致（`{query}`+`{ragkm}`）。
- **三个场景的提示词装配各自一个函数**（用户明确要求「分别单独实现」，不要再合成一个带
  `**values` 的通用函数，也不要共用渲染函数）：
  `build_rewrite_messages(prompt, query, history)`（填 `{query}`，**带历史**）、
  `build_answer_messages(prompt, query, context)`（填 `{query}`+`{ragkm}`，**单轮**）、
  `build_polish_messages(prompt, query, answer)`（填 `{query}`+`{answer}`，**单轮**）。
  三者**只共用 `fill()` 一个原语**，system 段的拼装、user 段的填充都写在各自函数体里
  —— 用户明确否掉了共用的 `_render()` / `_system_message()`（「每个方法单独处理」）。
- 入参是 `Prompt`（`app/models/prompt.py` 的 frozen dataclass，`system` + `user`），
  不是裸字符串模板。`PromptRepository.get()` 返回 `Prompt | None`。
- `app/services/model_client.py` 已删（httpx 直调模型被 langchain 取代）。
  `answer_summary` 这条 DB 记录**仍在用**（生成环节取它），别再当死配置删掉。
- **提示词以 `db.sql` 为准**，库（`rag_prompt`）里现存内容只作参考；用户明确过这一点。
  两边曾不一致（库版 `answer_polish` 多出「检索知识：{context}」段），已按 `db.sql` 覆写库。
  同步库时用 **`scripts/sync_rag_prompt.py`**（预览差异 → 确认写入 → 读回逐字符比对）。
  **整份 `db.sql` 经 stdin 管道执行没问题**（见上方「整库重建」）；
  **别用 mysql CLI 手动拼/截取 db.sql 的片段**：内容含大量 `\"` / `\n` 转义，
  手写命令行会打断解析（且各条只是 values 元组，需补 `INSERT INTO ... VALUES` 前缀）。
- **提示词表结构（2026-09-28 定稿）：一条提示词拆两列** ——
  `system_content`（固定的角色与规则，**不含变量**，原样作 system 消息，不做占位符替换）
  + `user_content`（含 `{query}`/`{ragkm}`/`{answer}` 占位符的正文，渲染后作 user 消息）。
  用户原话：「固定的提示词放在 system 里面，里面不带变量，带有变量的放在 user 里面」。
- 装配结果的消息形状：
  - 改写 = `[system, 历史轮次…, user]`；
  - 生成 / FAQ 润色 = `[system, user]`（单轮）。
  system 段为空时不产生 system 消息。
- **`user_content` 里漏写 `{query}` 时问题不会被带出来** —— 装配层**不自动补到末尾**（用户
  在「整段进 user」那版之后要求「每个方法单独处理」，连带去掉了这层兜底）。`user_content`
  **为空的行**在仓库层直接被跳过并回落内置，避免发一条残缺消息。
- `fill()` 同时认 `{name}` 与 `{{name}}`（先双后单：`{{query}}` 内含 `{query}`，
  先按单花括号替换会把双花括号啃成半截）。**db.sql 现行版本用的是双花括号**
  （用户写 user 段时就是这个写法），代码已兼容。
- 回归脚本：`scripts/verify_message_assembly.py`（9 组用例，含「当前问题只出现一次」不变量）。
- **FAQ 直返（`if direct`，两个分支都算）同样走 `answer_polish` 润色，且流式下发**（用户要求）。
  走 `app/services/stream_agent.py` 的 `ChatStreamAgent`，**不传 context**，
  只装配 `{query}` + `{answer}`（`db.sql` 的 `answer_polish` 也只有这两个占位符）。
  润色失败回落库中原文；这条链路多一次模型调用，已不是「0 次调用秒回」。
- 标签提取仍用内置提示词（DB 中无对应记录，未给它配 key）。
- DB 中的提示词含 JSON 输出示例（花括号），**不要用 `str.format` / `ChatPromptTemplate` 渲染**，
  须用 `app/services/prompting.py` 的显式占位符替换。

## 检索链路
- 完整链路：**术语映射 → FAQ 直返探测（跨库 `kbs:mix-retrieve`，只探 `knowledgeType==2` 的库）→
  改写 → FAQ 再探 → 跨库混合检索 → 单步生成最终答案（流式）**。
- **空间层级（2026-09-29 起）**：平台侧是「租户(`tenantId`) → 空间(`project_id`) → 知识库」，
  且**知识库不支持跨空间检索**。门户在每条 `ranges` 项里带 `project_id` / `tenantId` /
  `knowledgeType`（1 切片，2 标准问答），字段名两种写法都收（`AliasChoices`）。
  `project_id`/`tenantId` 是**空间级 query 参数**，不是全局配置 —— 所以 ranges 跨空间时
  必须按 `(project_id, tenantId)` 分组，每组各发一次请求。
  **`knowledgeType` 是门户入参字段，两份 LLMOps 平台文档里都没有，不要放进出网 body**。
- **`knowledgeType` 必须严格等于 `2` 才是 QA 库（用户 2026-09-29 明确更正）**：
  不传（`None`）与传 `1` 都**不参与 FAQ 直返**，一次 FAQ 探测都不发。
  曾按「未知保留为候选」实现，用户否掉 —— `None` 是被当成 QA 库塞进跨库检索、白跑一次
  且永远不可能命中。过滤点只有 `RagAgent._faq_probe` 一处。
- `hybrid_retrieval` 阶段之后调用的**不再是本项目的单库检索**，而是第三方**跨库召回接口**
  `POST {KB_PLATFORM_BASE_URL}/applet/api/v1/knowlhub/kbs:mix-retrieve?project_id=…&tenantId=…`
  （用户指定，接口文档「知识库召回接口（跨库检索）」）。**FAQ 探测与最终答案检索共用这一个端点**；
- 单库 `kbs:retrieve` 已不使用，`FAQ_TARGET_RANGE` 一并删除（不留死配置）。
  **但 `KB_FAQ_RETRIEVE_URL` / `faq_retrieve_url` 保留** —— 用户 2026-09-29 明确
  「FAQ 库地址可能与 `KB_MIX_RETRIEVE_URL` 不是一个 url，只是返回结构一样」：
  两条通道是同一接口的两处部署，地址各自可整条覆盖；FAQ 地址留空则**回落 mix 地址**。
- 跨库请求体：`query`（用改写结果，失败回落术语映射结果）、`keywords`（改写同一次调用产出的
  业务关键词；**FAQ 探测传空数组**，回归脚本就靠这个区分两类请求）、`ranges`、`disable_rerank=false`
  + `rerank_params`（跨库必须重排；`weight_type=1` 动态权重 WRRF 就不必传 rerank 模型对象）。
- 响应结构与单库检索一致：`result[].chunk.content` + `result[].score` / `doc_id` / `doc_name` /
  `knowledge_base_id`；取分最高的 `FINAL_CONTEXT_TOP_K` 条拼上下文。
  **标准问答库的答案优先取 `chunk.qa_pairs[].answer`，其次才是 `chunk.content`。**
- 跨库检索失败**只记 warning 返回空列表**，走 `answer_type=no_context` 降级，不抛异常。
- `KB_TENANT_ID`（留空则不带 `tenantId` 参数）。
- **召回接口 IP/路径待定**，因此地址支持整条覆盖，且**两个通道各自一个变量**：
  `KB_MIX_RETRIEVE_URL`（最终检索）与 `KB_FAQ_RETRIEVE_URL`（FAQ 探测，留空复用前者）。
  **不要再把 `{base}/applet/api/v1/...` 硬拼在客户端里** —— 用户明确说过地址还没定；
  客户端内部也别写死用 `mix_retrieve_url`，地址作为参数传（`_retrieve(url, payload, scope)`）。
- ⚠️ **待办**：`mix_retrieve()` 仍用全局 `KB_PROJECT_ID` / `KB_TENANT_ID`，**未按空间分组**，
  跨空间 ranges 会落到错误的空间（用户本轮明确「先只改 `_faq_probe`」）。修法照
  `faq_retrieve` 的 `_group_by_space()`，各组并发后合并再统一取 top_k。
- ⚠️ **`FAQ_SIMILARITY_THRESHOLD`（0.98）是按单库相似度调的**，现在 FAQ 也过跨库重排
  （WRRF 分数量纲不同），需要按真实数据重新标定；`_faq_match` 会在未达阈值时 INFO 打出实际最高分。
- **FAQ 直返附带「其他相近问题」（2026-09-29）**：命中后把本次召回的**其他 QA 命中的
  `question`** 按分数序去重取 **3 条**，在 **token 流全部下发之后、`done` 之前**
  发**一条** `related_queries` 事件（`{"queries": [...]}`，数组顺序即相似度序）。
  **凑不满 3 条就一条都不发**（不会出现 1~2 条）。
  （事件形状改过两轮：数组 → 逐条 `related_query` → 定稿回数组。**用户最终要的是「一条事件带三条」**，
  别再做逐条。改动只落在 `RagAgent._faq_direct()` 一处，所以来回改成本很低。）
  数据复用同一次召回，**不额外发请求**；未命中 FAQ（不直返）时也不发。
  `FaqProbeResult(hit, related_queries)` 是 `_faq_probe` 的新返回类型（原来直接返回 hit），
  `RetrievalHit.question` 承载 QA 问题原文（切片库命中为 None）。
  两个直返分支的事件序列抽在 `RagAgent._faq_direct()` 里 —— 顺序规则只写一遍，避免漏改。
- 回归脚本：`scripts/verify_mix_retrieve.py`（`httpx.MockTransport` 进程内 mock 出网，
  覆盖正常链路 / 生成提示词取 `answer_summary` / FAQ 直返 / **直返附三条相近问题** /
  **相近问题不足三条不发** / FAQ 未达阈值 / range 自带空间 /
  只有切片库不探 FAQ / **未标 knowledge_type 不探 FAQ** / 跨空间分组 / 检索 500 /
  生成模型失败 / 租户未配置 / 整条覆盖 mix 地址 / **FAQ 与检索地址各自覆盖** 共**十五组场景**）。
  **脚本的默认 ranges 必须标 `knowledgeType=2`**，否则 FAQ 探测为 0 次、大量断言假过。
  注意 FAQ 的请求次数期望：**每轮探测并发两个 query 变体、共两轮**，
  单库单空间 = 4 次请求，跨 2 空间 = 8 次。

## 关键文件
- `app/core/db.py` — 异步引擎
- `app/models/prompt.py` — `Prompt`（`system` + `user` 的 frozen dataclass）
- `scripts/sync_rag_prompt.py` — 把 `db.sql` 的 rag_prompt 同步进库（唯一受支持的同步方式），
  并负责老表结构（单列 `prompt_content`）就地迁移
- `app/services/repository.py` — 术语/提示词仓库（TTL 缓存 + 降级）；`PromptRepository.get()`
  返回 `Prompt | None`，`user_content` 为空的行跳过
- `app/services/knowledge_base.py` — 知识库召回客户端：`faq_retrieve()`（FAQ 探测，按空间分组走
  跨库检索，打到 `settings.faq_retrieve_url`）+ `mix_retrieve()`（最终答案跨库检索，打到
  `settings.mix_retrieve_url`），共用 `_payload()` / `_retrieve(url, payload, scope)` /
  `_post()` / `_to_hit()`；`_group_by_space()` 是空间分组规则（range 自带优先，缺省回落 settings）；
  `KNOWLEDGE_TYPE_QA = 2`；`RetrievalHit.question` 存 QA 命中的问题原文（切片库为 None），
  供 FAQ 直返的「其他相近问题」用。
  **不再有通用的 `retrieve(query, ranges, target_range)`**（那是死参数），
  **也不再有单库 `_retrieve_faq()`**；
  方法名从 `_mix_payload`/`_mix_request` 改成了中性的 `_payload`/`_retrieve` —— 两通道共用。
- `app/models/chat.py` — `KnowledgeRange`（`knowledge_base_id` / `doc_range` / `project_id` /
  `tenant_id` / `knowledge_type`，后两个用 `AliasChoices` 兼容驼峰与下划线入参）
- `scripts/verify_mix_retrieve.py` — 跨库检索链路回归（进程内 mock 出网，11 组场景）
- `scripts/verify_message_assembly.py` — 三个场景的消息装配回归（9 组用例）
- `scripts/verify_repository_loading.py` — 配置表读取层回归（3 组用例：按列名取值、
  跳过残缺行、术语表装配）。用假 `session`/`Database` 同形替身，不出网。
- `app/services/prompting.py` — 内置兜底提示词（都是 `Prompt`）+ `fill()` +
  **三个场景各自的装配函数**（见上文「配置表读取策略」）
- `app/services/text_normalizer.py` — `TermMapper`（最长优先单次扫描）
- `app/services/rewrite_agent.py` — query 改写（langchain `create_agent` + `ChatOpenAI`
  OpenAI 兼容协议）。模型输出纯 JSON 文本（`rewritten_query`+`keywords`），容错解析；
  不用 `response_format`/ToolStrategy（内部网关不认 tools）。一次调用同时出改写与关键词，
  没有单独的标签提取调用。调用是**非流式** `rewrite()`（`ainvoke`），不再有 `rewrite_token` 事件。
- `app/services/stream_agent.py` — `ChatStreamAgent`：**唯一的流式生成 agent**
  （langchain `create_agent` + 回答模型 + `astream(stream_mode="messages")`）。
  **FAQ 直返润色与 RAG 最终答案生成共用它** —— 两者都是「喂 messages、要一段流式文本」，
  只是提示词不同，拆两个类纯属重复实现（原 `faq_agent.py` 已删）。
  失败只记 warning 结束，把「降级成什么」留给调用方（FAQ 回落库中原文，RAG 回落一句提示语）。
- `app/services/rag_agent.py` 的 `_ensure_streamer()` — 惰性取上面那个 agent；
  原 `_ensure_rewriter` / `_ensure_faq_polisher` 合并成 `_ensure_streamer`（一个就够）。
- `app/services/chat_model.py` — `build_chat_model()`：ChatOpenAI 的唯一组装处，
  收敛网关怪癖（base_url 去 `/chat/completions`、api_key 占位、accessKey 头）。
  **模型地址只有一个来源：每个模型自己的 `*_MODEL_URL`（2026-09-29 定稿）。**
  曾经有过 `OPENAI_BASE_URL` / `OPENAI_API_KEY` 两个「全局网关地址 / 密钥」变量，
  **用户要求直接删掉，不要再加回来**：一层「每个模型一条完整 URL」就够，多一层全局覆盖
  只会制造优先级问题（曾把答案模型带模型 uuid 的路径顶换成改写模型的路径，且不报错）。
  鉴权：`MODEL_ACCESS_KEY` 同时作 `api_key` 与 `accessKey` 头，留空时 api_key 给 `"EMPTY"`。
  `ChatStreamAgent.build()` 未配 `ANSWER_MODEL_URL` 即抛 RuntimeError，由调用方降级。
  回归 `scripts/verify_model_config.py`（4 组）。

## 约定
- 助手风格：用户期望直接指出遗漏并执行，不要反复确认。
- 改动同一文件的多处内容须串行编辑，避免并行 Edit 互相覆盖。
- **不要给单个环节加功能开关**：用户明确去掉了 `TERM_MAPPING_ENABLED`，
  认为 `DB_ENABLED` 已覆盖「离线/降级」这一个语义，再叠一层只是徒增配置面。
- **不要写死每环节的魔数上限**：用户去掉了 `REWRITE_HISTORY_LIMIT` / `ANSWER_HISTORY_LIMIT`，
  改写拿到全量历史轮次（最终答案生成是单轮，本来不带历史；FAQ 直返润色按设计也不带）。
- **重资源客户端一律惰性创建**：用户不接受在构造函数里提前建（如 `RagAgent.__init__`
  里 new 出改写 agent）。构造函数只存注入值，首次真正用到才创建
  （`_ensure_rewriter()` / `_ensure_streamer()`）。
- **一个概念只用一个名字**：惰性创建用**显式方法** `_ensure_xxx()`，不要用与存储字段
  同名的 `@property`。曾写成 property `_rewrite_agent` + 字段 `_rewriter`，
  两个名字指同一对象，用户指出「有点绕」，已改成 `_ensure_*`。
- **`_` 前缀是 Python 的私有约定**（无 `private` 关键字）：只在本类/本模块内使用的
  方法、状态、模块级类加 `_`；会被其它文件调用的（`mapper()`、`apply()`、`get()`、
  `build()`、`invalidate()`、`stream()`、`rewrite()`）不加。判断就一条：
  会不会被别的文件调用。
- **不要为「过程展示」保留无意义的流式**：改写结果必须完整 JSON 才有用，因此用 `ainvoke`
  非流式一次拿结果；只有最终答案才流式下发。
- **模型调用一律走 langchain**（用户要求）：不要再写 httpx 直调 `chat/completions` 的
  模型客户端（原 `model_client.py` 已删）。所有 agent 的 `ChatOpenAI` 组装只经
  `chat_model.build_chat_model()` 一处。
- **同类 agent 不要各写一份**：形状相同（喂 messages、拿流式文本）的环节共用一个 agent 类，
  差异只体现在提示词上；环节数量不等于 agent 数量。
- **解包数据库行一律按列名，不用位置**（用户明确要求）：`for a, b, c in rows` 在列顺序变化时
  会静默读串，属「能跑但不可靠」的写法。见上文「数据库约定」。
