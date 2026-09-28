"""提示词的领域模型。

库里一条提示词拆成两段：
- `system`：固定的角色与规则，**不含变量**，原样作为 system 消息下发；
- `user`：含 `{query}` / `{ragkm}` / `{answer}` 等占位符的正文，渲染后作为 user 消息。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    """一条提示词配置（`rag_prompt` 的一行）。"""

    system: str
    user: str
