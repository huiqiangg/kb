from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class KnowledgeRange(BaseModel):
    """一条召回范围：落在哪个知识库、可选地收窄到哪些文档。

    平台侧层级是「租户 -> 空间(project_id) -> 知识库」，且**知识库不支持跨空间检索**，
    所以 `project_id` / `tenant_id` 不是全局配置，而是逐条 range 的属性 —— 门户按 range
    传进来，用来把召回请求落到正确的空间里。`knowledge_type` 描述知识库类型
    （1 切片库，2 标准问答库），FAQ 直返只对**显式标注为 2** 的知识库有意义。

    字段名兼容：门户文档与示例里同时出现过 `tenantId`/`knowledge_type` 两种写法，
    入参一律两种都收（`populate_by_name=True` 让下划线写法也能用）。出网时统一还原成
    门户的驼峰写法（`serialization_alias`）—— 最终检索会把 ranges 原样透传给外部接口。
    """

    model_config = ConfigDict(populate_by_name=True)

    knowledge_base_id: str = Field(min_length=1)
    doc_range: list[str] = Field(default_factory=list)
    # 空间 id。门户字段名就叫 project_id，不另起别名
    project_id: str = ""
    # 租户 id
    tenant_id: str = Field(
        default="",
        validation_alias=AliasChoices("tenantId", "tenant_id"),
        serialization_alias="tenantId",
    )
    # 知识库类型：1 切片，2 标准问答（QA）。门户未传时为 None —— 「没标类型」不等于
    # 「是 QA 库」，故 None 不参与 FAQ 直返（只有显式 2 才算）
    knowledge_type: int | None = Field(
        default=None,
        validation_alias=AliasChoices("knowledgeType", "knowledge_type"),
        serialization_alias="knowledgeType",
    )

    @field_validator("doc_range", mode="before")
    @classmethod
    def accept_single_document_id(cls, value: object) -> list[str]:
        return [value] if isinstance(value, str) else value or []


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    ranges: list[KnowledgeRange] = Field(min_length=1)
