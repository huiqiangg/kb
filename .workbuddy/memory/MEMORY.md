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
  重建后条数**与 `db.sql` 一致**为准（2026-09-29 词表精简后是 **44 条术语 + 3 条提示词**，
  之前是 47 条 —— 别死记数字，按文件数）。核对用
  `sync_rag_prompt.py --dry-run`（提示词）+ 应用层连接读一遍（术语）。
- **本机没有 `mysql` CLI**（MySQL 只在 OrbStack 容器 `intent-mysql` 里），所以
  「按 `db.sql` 重建」的常规做法是**用项目自己的 async 引擎**跑脚本，不要为它装 CLI：
  `scripts/reload_term_mapping.py`（**2026-09-29 新增**，只重建术语表：DROP + 建表 + INSERT
  全部从 `db.sql` 原样抽出，做完打印行数并用 `TermMapper` 跑替换自检）。
- **`db.sql` 的 `CREATE TABLE` 不能有尾逗号**（MySQL 语法，`ERROR 1064 near ')'`）：
  2026-09-29 原始文件两处 `enabled ... '...,'\n)` 都是坏的，**整份执行必失败** ——
  已修掉。以后再改初始化脚本，最后一行列定义不要带逗号；验证手段是「临时库整份跑一遍」：
  建 `<db>_sqlcheck` → 逐条执行 → 核对行数 → `DROP DATABASE`，不碰真实库。
- 访问方式：SQLAlchemy async + aiomysql，惰性建池，`pool_pre_ping`。
- **读表一律 `.mappings()` 后按列名取值**（`_text_column(row, "system_content")`），
  **不要写 `for key, system, user in rows` 这类位置解包** —— 用户 2026-09-28 指出这种写法
  「不健壮」。位置解包的失败是**静默**的：SELECT 列顺序一调整就把两列读串（不报错不 warning），
  多一列则报 `ValueError` 且报错点离原因很远。按列名取 → 列名写错直接 `KeyError`，即刻可见。
  `scripts/sync_rag_prompt.py` 里的 `row[0]` / `tuple(stored)` 也一并改成了 mappings 写法。

## 配置表读取策略
- 术语映射与提示词**从 DB 读取**，进程内 TTL 缓存 300s（`DB_CACHE_TTL_SECONDS`），
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
  走 `RagAgent._stream_tokens()`（直接用 `ChatOpenAI.astream`），**不传 context**，
  只装配 `{query}` + `{answer}`（`db.sql` 的 `answer_polish` 也只有这两个占位符）。
  润色失败回落库中原文；这条链路多一次模型调用，已不是「0 次调用秒回」。
- 标签提取仍用内置提示词（DB 中无对应记录，未给它配 key）。
- DB 中的提示词含 JSON 输出示例（花括号），**不要用 `str.format` / `ChatPromptTemplate` 渲染**，
  须用 `app/services/prompting.py` 的显式占位符替换。
- **术语映射的冷加载会算在「关键词替换」耗时里**（日志紧跟在 `术语映射表已加载：N 条` 之后）：
  首次 ~25–60ms（容器网络冷时可到 400ms），TTL 内 0.0x ms。用户问「为什么 211ms」时按此解释，
  别答成「替换算法慢」。是否启动预热是取舍问题（与「启动不主动连库」冲突），不默认加。
- **术语表数据坑（2026-09-29 已由用户改词表解决）**：原表里
  `房贷→个人住房贷款` + `提前还款→贷款提前还款` 同时命中会拼出
  「个人住房贷款能贷款提前还款吗」（两条替换都对，是标准词自带公共前缀）。
  用户已把 `提前还款` / `部分提前还款` 两条**删掉**，改成 `还贷→偿还贷款`、`提前还贷→提前还款`，
  现在「房贷能提前还款吗」→「个人住房贷款能提前还款吗」。
  排查这类「替换后读不通」的问题，先逐条打印命中的 `source→standard` 再判断是词表还是逻辑；
  **词表是用户业务数据，不要擅自改**（这次也是等用户改完再重建）。

## 检索链路
- 完整链路：**术语映射 → FAQ 直返探测（跨库 `kbs:mix-retrieve`，只探 `knowledgeType==2` 的库）→
  改写 → FAQ 再探 → 跨库混合检索 → 单步生成最终答案（流式）**。
- **空间层级（2026-09-29 起）**：平台侧是「租户(`tenantId`) → 空间(`project_id`) → 知识库」，
  且**知识库不支持跨空间检索**。门户在每条 `ranges` 项里带 `project_id` / `tenantId` /
  `knowledgeType`（1 切片，2 标准问答），字段名两种写法都收（`AliasChoices`）。
  `project_id`/`tenant_id` 的位置**两条通道不同**（用户 2026-09-29 两次定稿）：
  - `mix_retrieve`（外部方给的接口）→ **普通 POST**：`client.post(url, json=payload)`，
    **不带查询参数、不带任何请求头**（不要 `Authorization`），**body 只有三项**：
    `{"query": …, "keywords": […], "ranges": […]}`，**不要 disable_rerank / rerank_params**。
    `ranges` **直接用本次请求的 ranges**（`model_dump(by_alias=True, exclude_unset=True)`，
    门户传什么发什么、含 `project_id`/`tenantId`/`knowledgeType`，没传的字段不补空值），
    **一次请求全部发完**、不按空间分组。
    用户原话：「这是外部接口，只吃一个空间，不要分组、就发一次」+
    「ranges 直接使用 request 请求的 ranges，请求头不需要加 Authorization，就是一个普通的 post 请求」+
    「请求 body 是这样 {query, keywords, ranges}，ranges 就是本次 http 请求的 ranges」。
  - `faq_retrieve`（平台接口）→ `project_id`/`tenantId` 查询参数
    （`project_id` 取 range 自带的、缺省回落 `KB_PROJECT_ID`；`tenantId` 只认 range 自带的），
    body = 上述三项 +
    `disable_rerank=false` + `rerank_params`（`_faq_body()`），ranges 只带
    `knowledge_base_id`/`doc_range`，跨空间时按 `(project_id, tenantId)` 分组、每组各发一次。
  **出网 ranges 的键名靠 `serialization_alias` 还原成门户驼峰**（`tenantId` / `knowledgeType`）——
  只加 `validation_alias` 时序列化仍是模型字段名（`tenant_id`），2026-09-29 踩过一次；
  `project_id` / `knowledge_base_id` / `doc_range` 门户本来就叫这个，无需别名。
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
- 跨库请求体两条通道不同：`query`（用改写结果，失败回落术语映射结果）、`keywords`（改写同一次
  调用产出的业务关键词；**FAQ 探测传空数组**）两边都有；`disable_rerank=false` + `rerank_params`
  （跨库必须重排；`weight_type=1` 动态权重 WRRF 就不必传 rerank 模型对象）**只有 FAQ 那条有**，
  最终检索（外部接口）body 只有三项。
  **回归脚本按「body 里有没有 `disable_rerank`」区分两类请求**（原来按 keywords 是否为空，
  改写失败时 keywords 恰好为空会误判）。
- 响应结构与单库检索一致：`result[].chunk.content` + `result[].score` / `doc_id` / `doc_name` /
  `knowledge_base_id`；取分最高的 `FINAL_CONTEXT_TOP_K` 条拼上下文。
  **标准问答库的答案优先取 `chunk.qa_pairs[].answer`，其次才是 `chunk.content`。**
- 跨库检索失败**只记 warning 返回空列表**，走 `answer_type=no_context` 降级，不抛异常。
- **召回接口不配鉴权、租户没有全局兜底（2026-09-29 用户定稿）**：`.env` 里的
  `KB_AUTHORIZATION` 与 `KB_TENANT_ID` **已删**，代码里也没有这两个字段了 ——
  `settings.kb_authorization` / `settings.kb_tenant_id` 都不要再加回来。
  即两条通道**都不拼 `Authorization` 头**；`project_id` 保留全局兜底 `KB_PROJECT_ID`
  （只对 FAQ 探测生效），`tenantId` 只认 `ranges` 自带的（取不到就不拼该查询参数）。
  回归脚本里有一条「两条通道都不带 Authorization」守着，防以后被顺手加回来。
- **召回接口 IP/路径待定**，因此地址支持整条覆盖，且**两个通道各自一个变量**：
  `KB_MIX_RETRIEVE_URL`（最终检索）与 `KB_FAQ_RETRIEVE_URL`（FAQ 探测，留空复用前者）。
  **不要再把 `{base}/applet/api/v1/...` 硬拼在客户端里** —— 用户明确说过地址还没定；
  两条通道各自从 `settings.mix_retrieve_url` / `settings.faq_retrieve_url` 取地址，
  不再有「把 url 当参数传」的通用方法（一个通道一个地址来源，传参只是死参数）。
- **两条通道：请求形状各一份、响应解析共用一份（2026-09-29 用户定稿，晚间再确认响应同构）**。
  `mix_retrieve()` 调的是**外部方提供的接口**，请求流程**自带一份**、不借道 FAQ 那条 —— 用户原话
  「这个接口是别人提供给我的」，意思是契约变化要能只落在一处。
  所以**没有 `_retrieve()` 这个共用方法了**（曾在 2026-09-29 上午合并出来，下午按用户要求拆开）。
  现在：请求体与发请求各有一份 —— FAQ 走 `_faq_body()`（前三项 + rerank 参数 + `_kb_ranges()` 投影）
  + `_faq_request(payload, scope)`（**只有 FAQ 用**：拼 project_id/tenantId 查询参数）；
  mix 两项都直接写在 `mix_retrieve()` 里（普通 POST、三项 body）。
  **`_body()` 那个「共用骨架」已删**（两边字段已经不同）；原 `_request(url, payload, scope)` 的
  `url` 是死参数（只剩一个调用点），2026-09-29 收成 `_faq_request()` 并去掉该参数。
  用户晚间确认「**mix_retrieve 与 faq_retrieve 单次请求的 response 结构相同**」→ 所以
  **响应解析只有一份**：`_parse()` / `_to_hit()`，两条通道都调它（已实测：同一份同构响应
  分别走两条通道，命中的 7 个字段逐个一致，含 QA 库的 `chunk.qa_pairs`）。
  **判据：请求形状可以复制，响应解析不能复制。**
  注意**请求形状（发几次、body 几个字段、带不带参数/请求头、ranges 怎么发）也属于接口性质**、
  不是共用规则：mix = 普通 POST + 三项 body + 原样 ranges（见上文），
  FAQ = 查询参数 + rerank 参数 + 按空间分组。
  **别为了「两条路长一样」把任一边的形状抄到另一边。**
  历史坑：两条路各写一份时只给 FAQ 做了按空间分组、`mix_retrieve` 用全局
  `KB_PROJECT_ID`/`KB_TENANT_ID`，跨空间请求落到错误空间（那时它还是本项目的接口）。
  `asyncio.gather(return_exceptions=True)` 的顺序与传入一致，`zip(groups, outcomes)`
  就能把失败组与它的 project_id 一起打进 warning；`mix_retrieve` 只发一次，
  失败用 `try/except` 降级成空列表（同样只 warning，不抛）。
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
  只有切片库不探 FAQ / **未标 knowledge_type 不探 FAQ** / 跨空间（FAQ 分组 8 次 / 检索只发 1 次）/
  检索 500 /
  生成模型失败 / 租户未配置 / 整条覆盖 mix 地址 / **FAQ 与检索地址各自覆盖** /
  **用真 `ChatOpenAI` 流式出 token** 共**十六组场景**）。
  **脚本的默认 ranges 必须标 `knowledgeType=2`**，否则 FAQ 探测为 0 次、大量断言假过。
  注意 FAQ 的请求次数期望：**每轮探测并发两个 query 变体、共两轮**，
  单库单空间 = 4 次请求，跨 2 空间 = 8 次。
  **测试里替换 `rag_agent.httpx` 要换成副本（`types.SimpleNamespace(**vars(httpx))`），
  不能直接改 `httpx.AsyncClient` 属性** —— openai SDK 会 `isinstance(client, httpx.AsyncClient)`，
  把模块属性换成 lambda 会让真模型场景炸 `TypeError: isinstance() arg 2 must be a type`。

## 关键文件
- `app/core/db.py` — 异步引擎
- `app/models/prompt.py` — `Prompt`（`system` + `user` 的 frozen dataclass）
- `scripts/mock_kb_gateway.py` — **本地假召回网关（联调用）**，零依赖标准库 `http.server`。
  把 `KB_FAQ_RETRIEVE_URL` / `KB_MIX_RETRIEVE_URL` 指到 `http://127.0.0.1:8099/applet/api/v1/knowlhub/kbs:mix-retrieve`
  即可让 `_faq_probe` / `mix_retrieve` 拿到假数据，**不用改生产代码**（比硬编码 return 干净，也不会忘删）。
  默认给「1 条 0.99 的 QA 作答 + 3 条相近问题」，触发 FAQ 直返 + `related_queries`。
  `MOCK_CHANNEL=auto|faq|chunk`、`MOCK_FAQ_SCORE`、`MOCK_RELATED_COUNT`、`MOCK_PORT` 可调。
- `scripts/sync_rag_prompt.py` — 把 `db.sql` 的 rag_prompt 同步进库（唯一受支持的同步方式），
  并负责老表结构（单列 `prompt_content`）就地迁移
- `scripts/reload_term_mapping.py` — **按 `db.sql` 重建 `rag_keywords_mapping`（2026-09-29 新增）**。
  从 `db.sql` 原样抽出该表的 `CREATE TABLE` + `INSERT` 两条语句 → DROP + 建表 + 灌数据 →
  打印行数并用 `TermMapper` 跑替换自检。用项目自己的 async 引擎（本机没 mysql CLI）。
  跑法 `PYTHONPATH=. .venv/bin/python scripts/reload_term_mapping.py`；**只动术语表**，
  提示词走 `sync_rag_prompt.py`。跑前先备份（脚本不自动备份）。
- `app/services/repository.py` — 术语/提示词仓库（TTL 缓存 + 降级）；`PromptRepository.get()`
  返回 `Prompt | None`，`user_content` 为空的行跳过
- `app/services/knowledge_base.py` — 知识库召回客户端（2026-09-29 拆成两份流程）。
  `mix_retrieve(query, keywords, ranges)`：**普通 POST**（`client.post(url, json=payload)`，
  无 header、无 query），body 就三项 `query` / `keywords` / `ranges`，其中 `ranges` 用
  `model_dump(by_alias=True, exclude_unset=True)` **原样带上门户字段（含 knowledgeType）**，
  只发一次；失败 `try/except` → warning → `[]`；返回 `list[RetrievalHit]`（按分数降序）。
  `faq_retrieve(query, ranges)`：`keywords` 恒空、地址 `faq_retrieve_url`，body 由
  `_faq_body()` 组装（前三项 + `disable_rerank` / `rerank_params`），仍走
  「`_group_by_space()` 分组 → 组间并发（`_faq_request(payload, scope)` 拼 project_id/tenantId
  查询参数）→ 单组失败只 warning → 合并排序」。
  **已无 `_retrieve()` 共用方法**（见上文判据）。**响应解析两条通道共用** ——
  `_parse()` / `_to_hit()`（用户确认两边 response 结构相同）；其余按通道各一份：
  `_faq_body()` / `_kb_ranges()`（平台 range 投影）/ `_faq_request()`。
  `KNOWLEDGE_TYPE_QA = 2`；`RetrievalHit.question` 存 QA 命中的问题原文（切片库为 None）。
  已删的冗余：`SpaceScope` 别名（只出现两次，直接用 `tuple[str, str]`）、
  `RetrievalHit.raw`（只写不读）、`_space_of()`（mix 不再需要收敛空间）、
  `_payload()` → `_faq_body()`、`_body()`（两条通道字段已不同，共用骨架名不副实）、
  `_request(url, payload, scope)` → `_faq_request(payload, scope)`
  （`url` 是死参数，且名字看着通用、实际只服务 FAQ）。
  **不再有通用的 `retrieve(query, ranges, target_range)`**（那是死参数），
  **也不再有单库 `_retrieve_faq()`**。
- `app/models/chat.py` — `KnowledgeRange`（`knowledge_base_id` / `doc_range` / `project_id` /
  `tenant_id` / `knowledge_type`）。后两个**入参**用 `AliasChoices` 收驼峰与下划线两种写法，
  **出网**再用 `serialization_alias` 还原成门户驼峰（`tenantId` / `knowledgeType`）——
  pydantic 的 `validation_alias` 只管入参、不影响序列化（踩过）。
- `scripts/verify_mix_retrieve.py` — 跨库检索链路回归（进程内 mock 出网，11 组场景）
- `scripts/verify_message_assembly.py` — 三个场景的消息装配回归（9 组用例）
- `scripts/verify_repository_loading.py` — 配置表读取层回归（3 组用例：按列名取值、
  跳过残缺行、术语表装配）。用假 `session`/`Database` 同形替身，不出网。
- `app/services/prompting.py` — 内置兜底提示词（都是 `Prompt`）+ `fill()` +
  **三个场景各自的装配函数**（见上文「配置表读取策略」）
- `app/services/text_normalizer.py` — `TermMapper`（最长优先单次扫描）
- `app/services/rewrite.py` — 只剩**纯函数**（2026-09-29 拍平后）：`RewriteResult`（dataclass）
  + `parse_rewrite_json()` + `KEYWORD_LIMIT`。模型输出纯 JSON 文本
  （`rewritten_query`+`keywords`），容错解析兼容旧字段；不用 `response_format`/ToolStrategy
  （内部网关不认 tools）。一次调用同时出改写与关键词，没有单独的标签提取调用。
  **没有客户端、没有类、没有单例** —— 模型与调用都在 `RagAgent` 里
  （原 `rewrite_agent.py` 的 `QueryRewriteAgent` 类 + `build()` + `create_agent` + 模块单例
  已全部删除，不要再加回来）。
- `app/services/rag_agent.py` 的 `_ensure_answer_model()` —— **回答模型的唯一组装点**，
  惰性创建；`_stream_tokens(messages)` 直接 `ChatOpenAI.astream()` 逐段出文本，
  **FAQ 直返润色与 RAG 最终答案生成共用它**（两者都是「喂 messages、要一段流式文本」，
  只是提示词不同）。失败只记 warning 结束，把「降级成什么」留给调用方
  （FAQ 回落库中原文，RAG 回落一句提示语）。
- **模型客户端就地组装，没有公共包装层（2026-09-29 定稿，用户明确要求）**：
  `app/services/chat_model.py`（`build_chat_model()`）、`app/services/stream_agent.py`
  （`ChatStreamAgent`）、`app/services/rewrite_agent.py`（`QueryRewriteAgent` + `build()` +
  `create_agent` + 模块单例）**全部已删，不要再加回来**。`ChatOpenAI(...)` 只在两处各写一遍，
  都在 `RagAgent`：改写在 `_ensure_rewrite_model()`、回答在 `_ensure_answer_model()`。
  组装规则：base_url 去 `/chat/completions` 后缀；鉴权用**各自模型的密钥**
  `REWRITE_MODEL_KEY` / `ANSWER_MODEL_KEY`（同时作 `api_key` 与 `accessKey` 头），
  留空时 api_key 给 `"EMPTY"`、不带 `accessKey` 头。
  **两侧地址留空时都要抛 RuntimeError 并点名变量**（`REWRITE_MODEL_URL` / `ANSWER_MODEL_URL`）——
  空 base_url 会静默打到 api.openai.com，而两侧失败都只吞成 warning。
  **地址与密钥都只有一个来源：每个模型自己那组 `*_MODEL_URL` / `*_MODEL_KEY`。**
  曾有 `OPENAI_BASE_URL`/`OPENAI_API_KEY` 两个「全局网关地址/密钥」变量，**用户要求删掉**；
  后来又收敛出来的全局 `MODEL_ACCESS_KEY` 也按同一条逻辑拆成两个 per-model key 并删除
  （2026-09-29，用户原话「env 配置中缺少 `REWRITE_MODEL_KEY` 和 `ANSWER_MODEL_KEY`」）——
  **不要再加回任何一种全局覆盖变量**（曾把答案模型带模型 uuid 的路径顶换成改写模型的路径，且不报错）。
  回归 `scripts/verify_model_config.py`（4 组，含「两模型地址与密钥互不覆盖」）。
  剥离 `create_agent` 后 `langchain` / `langgraph*` 不再是直接依赖，
  `pyproject.toml` 已去掉 `langchain`，`uv lock` 一并清掉 6 个包（111 行）。

## 约定
- 助手风格：用户期望直接指出遗漏并执行，不要反复确认。
- 改动同一文件的多处内容须串行编辑，避免并行 Edit 互相覆盖。
- **不要给单个环节加功能开关**：用户明确去掉了 `TERM_MAPPING_ENABLED`，
  认为 `DB_ENABLED` 已覆盖「离线/降级」这一个语义，再叠一层只是徒增配置面。
- **不要写死每环节的魔数上限**：用户去掉了 `REWRITE_HISTORY_LIMIT` / `ANSWER_HISTORY_LIMIT`，
  改写拿到全量历史轮次（最终答案生成是单轮，本来不带历史；FAQ 直返润色按设计也不带）。
- **重资源客户端一律惰性创建，构造参数照旧保留**：用户不接受在构造函数里提前建
  `ChatOpenAI`，但**接受（且要求）构造函数接收并暂存注入的模型实例**：
  `self._rewrite_model = rewrite_model` / `self._answer_model = answer_model`
  （默认 `None` = 尚未创建），由 `_ensure_rewrite_model()` / `_ensure_answer_model()`
  首次取用时创建。曾把它们改成类属性默认 `None`、`__init__` 不收参数，被用户否掉
  （原话「算了还是放在构造函数里面吧」）—— 不要再动这两个参数。
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
  模型客户端（原 `model_client.py` 已删）。`ChatOpenAI` 就地组装，**没有公共组装模块**。
- **抹平重复 ≠ 抽公共模块**（2026-09-29 用户明确否掉）：形状相同的环节共用**方法**
  （`RagAgent._stream_tokens` 同时服务 FAQ 润色与最终答案生成），不要为此保留一个**类/模块**。
  用户原话「也不需要包装 ChatStreamAgent，直接使用 `ChatOpenAI(...)` 这种方式」——
  六行构造参数重复两次，比多一层包装更好维护。
- **解包数据库行一律按列名，不用位置**（用户明确要求）：`for a, b, c in rows` 在列顺序变化时
  会静默读串，属「能跑但不可靠」的写法。见上文「数据库约定」。
- **本项目不需要过度封装，直接实现对应功能即可**（2026-09-29 用户原话，总纲）：
  这是贯穿全项目的默认取向，上面几条「不抽公共模块 / 不包 agent / 不为复用叠抽象层」
  都是它的具体化。落笔时先问「这层抽象是为谁加的」——
  - 同形状的重复**就地写第二遍**优于抽基类/工具模块（`ChatOpenAI` 构造、
    `_faq_direct` 里的润色流、`_TtlStore.get` 里两次「不用打库」的判定）；
  - 不为「以后可能换实现」预留接口/protocol/工厂/包装类；
  - 不加「可能有用」的配置开关与魔数上限（见上文两条）；
  - 真正需要收敛的**顺序/规则**（直返事件序列 `_faq_direct`、FAQ 召回的分组与降级）
    才抽成一个方法，抽的是**语义**不是**代码形状**；
  - **请求形状（发几次、带不带参数/请求头、ranges 怎么发）属接口性质，不跟着别人抄**：
    外部接口就按它说的写成普通 POST + 原样 ranges（`mix_retrieve`），
    别为了两条路对称把 FAQ 的分组/查询参数搬过去，反之亦然；
  - 只写不读的字段、声明了没人用的参数、只出现在一两处的类型别名，都直接删。
  **2026-09-29 整体重构已逐条落地**，删掉的东西不要再加回来：
  `app/services/chat_model.py`、`app/services/stream_agent.py`、
  `app/services/rewrite_agent.py`（类 + `build()` + `create_agent` + 模块单例）、
  8 行的 `app/core/logging.py`、仓库根目录那个 PyCharm 样例 `main.py`、
  `KnowledgeBaseClient` 的两套同形召回方法、`RetrievalHit.raw`、`SpaceScope` 别名。
  判断标准：**能一眼读完、改一处就生效** > 层次整齐。宁可长一点、直白一点。
  重构的配套要求：**先有行为断言脚本**（本项目 4 个脚本 34 组用例），改完原地复跑 +
  起服务打一次 SSE 冒烟，再交给用户；否则「瘦身」会变成改坏行为的借口。
- ⚠️ **用户会自己改代码并提交**（commit message 一律「第N版」，作者是他本人）。
  断言变红时**先判断是哪一侧的意图落后，不要默认「测试是对的、代码是坏的」**：
  - 代码新写、测试还是旧契约 → **更新测试**（2026-09-29 16:05 就是这个情况，我改反了方向，
    被用户当场抓回：「我修改的yield，你为什么又改回去了」）；
  - 代码漏实现、测试是文档先行 → 才改代码。
  判据：`git log -p` / `git show HEAD:<file>` 看改动**是否成套**（整套 SSE 契约一起变 =
  有意重构；单点遗漏 = 多半是漏实现），并核对时间戳——**提交时间晚于我的编辑的，
  一律视为用户的有意改动，不碰**。
  「覆盖 HEAD 跑一遍确认归属」这一步没错，**错在确认完归属后反手把用户拉回旧状态**。
- **提交前先看 `git status` / `git diff`**：工作区里的改动可能是用户手写的
  （如 `_faq_direct` 的 yield、`api_key` 的 `or "EMPTY"` 兜底），别当成自己的半成品顺手「修好」。
