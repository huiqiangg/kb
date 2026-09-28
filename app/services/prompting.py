"""各环节兜底提示词，以及按场景拆分的消息装配。

为什么不用 str.format / ChatPromptTemplate：
提示词正文内嵌了 JSON 输出示例（如 {"rewritten_query": "..."}），f-string 风格渲染会把
这些花括号当成占位符而抛错。这里只对显式声明的占位符做字符串替换（见 `fill`）。

提示词的形状（`Prompt`）：一条提示词拆成两段，固定与可变分开 ——
- `system`：角色与规则，**不含任何变量**，原样作为 system 消息；
- `user`：含 `{query}` / `{ragkm}` / `{answer}` 等占位符的正文，渲染后作为 user 消息。
业务方改文案只改 `db.sql` 的这两段，代码侧不动。

三个场景的提示词形状与是否需要上下文不同，因此各有一个装配函数，装配过程完整写在函数体里：
- `build_rewrite_messages`  —— 改写（`query_understanding`）：`{query}`，**带对话历史**
- `build_answer_messages`   —— 最终答案生成（`answer_summary`）：`{query}` + `{ragkm}`，单轮
- `build_polish_messages`   —— FAQ 直返润色（`answer_polish`）：`{query}` + `{answer}`，单轮

三者只共用 `fill()` 这一个底层原语：各自的 system 段原样下发、user 段填自己的占位符，
没有额外的通用装配层 —— 占位符契约写在各自签名里，读一个函数就知道它要什么。
"""

from __future__ import annotations

from app.models.prompt import Prompt


def fill(template: str, **values: str) -> str:
    """只替换显式占位符，模板里其它花括号原样保留。

    每个占位符的 `{{name}}`（双花括号）与 `{name}`（单花括号）两种写法都会被替换，
    顺序固定「先双后单」，避免 `{{query}}` 被单花括号规则先啃掉一层。
    模板中声明了、但这里没给值的占位符会**原样保留**——这是「漏传」的可见信号，
    不做静默剔除（静默剔除会让提示词和实际输入对不上，比报错更难查）。
    """
    result = template
    for key, value in values.items():
        result = result.replace("{{" + key + "}}", value)
        result = result.replace("{" + key + "}", value)
    return result


def build_rewrite_messages(
    prompt: Prompt, query: str, history: list[dict[str, str]]
) -> list[dict[str, str]]:
    """改写场景（`query_understanding`）：system 指令 + 对话历史 + user 正文。

    system 段是固定指令，**不做占位符替换**（它按定义不含变量）；空 system 不产生消息。
    user 段填 `{query}`，即当前问题在消息里唯一出现的地方。
    提示词要求「结合上下文」改写当前问题，所以 `history` 必须给到，插在 system 与 user 之间；
    `history` 里不含当前问题（调用方只传当前轮之前的轮次）。
    """
    messages: list[dict[str, str]] = []
    system = prompt.system.strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(history)
    messages.append({"role": "user", "content": fill(prompt.user, query=query).strip()})
    return messages


def build_answer_messages(prompt: Prompt, query: str, context: str) -> list[dict[str, str]]:
    """最终答案生成场景（`answer_summary`）：**单轮**，system 指令 + user 正文。

    user 段填 `{query}` 与 `{ragkm}` —— `context` 是跨库混合检索后拼好的参考上下文
    （`[资料1] …\\n\\n[资料2] …`）。
    """
    messages: list[dict[str, str]] = []
    system = prompt.system.strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {"role": "user", "content": fill(prompt.user, query=query, ragkm=context).strip()}
    )
    return messages


def build_polish_messages(prompt: Prompt, query: str, answer: str) -> list[dict[str, str]]:
    """FAQ 直返润色场景（`answer_polish`）：**单轮**，system 指令 + user 正文。

    user 段填 `{query}` 与 `{answer}` —— `answer` 是库中人工维护的标准答案。
    这条链路一次调用直出，不传对话历史：聊天历史对「把一条标准答案润色一遍」没有帮助，
    只会把上一轮的措辞带进来。
    """
    messages: list[dict[str, str]] = []
    system = prompt.system.strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {"role": "user", "content": fill(prompt.user, query=query, answer=answer).strip()}
    )
    return messages


# ---- 内置兜底提示词：DB 中不存在对应 prompt_key、或该行配置不完整时使用 ----

DEFAULT_REWRITE_PROMPT = Prompt(
    system=(
        "你是农商行业务查询改写器。结合上下文，把用户最新问题改写成一条独立、完整、"
        "适合银行知识库检索的标准业务术语问题，并提取关键业务标签。"
        "不要回答用户问题，只负责 Query 理解、改写与关键词提取。"
    ),
    user=(
        '只输出 JSON，不要输出任何解释，格式为'
        '{"rewritten_query": "适合检索的标准化 Query", "keywords": ["关键词1", "关键词2"]}。\n\n'
        "用户问题：\n{query}"
    ),
)

# 最终答案生成的兜底：DB 的 answer_summary 取不到时用它。
# 占位符与 db.sql 的 answer_summary 保持一致（{query} + {ragkm}）。
DEFAULT_SUMMARY_PROMPT = Prompt(
    system=(
        "你是一名银行智能客服助手。\n\n"
        "请根据用户问题和参考上下文，生成准确、简洁、专业的银行业务答案。\n\n"
        "【处理原则】\n"
        "1. 仅依据参考上下文回答，不得补充上下文之外的信息。\n"
        "2. 准确保留与问题相关的结论、条件、金额、利率、时间、流程等关键信息。\n"
        "3. 内容存在冲突或差异时，不自行判断，分别列出相关答案及其差异。\n"
        "4. 参考上下文无法回答用户问题时，输出：『未检索到相关知识』。\n"
        "5. 内容较多时进行归纳，避免重复；涉及步骤、条件或注意事项时使用分点表达。\n"
        "6. 直接回答用户问题，不解释推理过程，不提及内部检索过程。"
    ),
    user="用户问题：\n{query}\n\n参考上下文：\n{ragkm}\n\n请输出最终答案",
)

# FAQ 直返答案的润色兜底（人工维护的标准答案只需做表达层整理，没有检索知识）
DEFAULT_POLISH_PROMPT = Prompt(
    system=(
        "你是一名银行智能客服答案润色助手。\n\n"
        "请根据用户问题和原始答案，对答案进行润色。\n\n"
        "要求：\n"
        "1. 保持原始答案的事实和含义，不得编造知识库中不存在的信息。\n"
        "2. 语言准确、专业、简洁，符合银行客服场景。\n"
        "3. 删除重复、冗余和无意义的表达。\n"
        "4. 金额、利率、日期、政策名称、业务规则等关键信息不得擅自修改。\n"
        "5. 原始答案已经准确、清晰时，不要为了润色而改写。\n"
        "6. 直接输出润色后的最终答案，不要解释润色过程。"
    ),
    user="用户问题：\n{query}\n\n原始答案：\n{answer}\n\n请输出最终答案",
)
