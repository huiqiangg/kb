from functools import lru_cache
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "农商行知识问答 Agent"
    log_level: str = "INFO"
    kb_platform_base_url: str = "http://127.0.0.1:8080"
    kb_project_id: str = "assets"
    # 租户 id，留空则请求不带 tenantId 参数
    kb_tenant_id: str = ""
    kb_authorization: str = ""
    # 召回接口地址（IP/路径尚未确定）：留空则按 kb_platform_base_url + 平台默认路径推导。
    # 定了地址后只填这几个，不必改代码；project_id / tenantId 仍会作为查询参数拼上。
    # FAQ 探测与最终答案检索**请求体与响应结构相同**，但知识库可能不在同一个网关地址上，
    # 所以两个地址各自可整条覆盖。
    kb_mix_retrieve_url: str = ""
    # 留空则复用 kb_mix_retrieve_url（即两通道同地址）
    kb_faq_retrieve_url: str = ""
    # ---- 模型服务：改写（非流式 JSON）与最终答案（流式文本） ----
    # 每个模型各一条完整地址（含 /chat/completions）、各一个密钥，两组配置完全对称、
    # 各自独立、互不覆盖 —— 不做「全局地址/全局密钥 + 局部覆盖」那种双层配置，
    # 少一层优先级要记，也不会出现全局值把某个模型的专用路径/密钥悄悄顶掉还不报错的情况。
    rewrite_model_url: str = "http://77.6.65.47:8099/api/model/chat/completions"
    rewrite_model_name: str = "Qwen2.5-coder-7B-Instruct"
    rewrite_model_key: str = ""
    answer_model_url: str = ""
    answer_model_name: str = "Qwen3-32B"
    answer_model_key: str = ""
    faq_similarity_threshold: float = 0.98
    retrieval_top_k: int = 12
    final_context_top_k: int = 6
    request_timeout_seconds: float = 45.0

    # ---- 数据库：术语映射与提示词配置 ----
    # 关闭后服务不连库，术语映射与 DB 提示词全部回落内置实现。
    db_enabled: bool = True
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = ""
    db_name: str = "kb"
    db_charset: str = "utf8mb4"
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_pool_recycle_seconds: int = 1800
    db_connect_timeout: float = 5.0
    # 术语映射与提示词的进程内缓存时长，过期后下次请求重新加载。
    db_cache_ttl_seconds: int = 60

    # ---- prompt key 映射：与 db.sql 中 rag_prompt 的记录一一对应 ----
    # Query 改写 + 关键词提取（只用于改写环节）
    prompt_key_query_understanding: str = "query_understanding"
    # 最终答案生成（检索资料以 {answer} 传入）
    prompt_key_answer_summary: str = "answer_summary"
    # FAQ 直返答案的润色
    prompt_key_answer_polish: str = "answer_polish"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def db_url(self) -> str:
        """SQLAlchemy 异步连接串，账号密码做 URL 编码以兼容特殊字符。"""
        return (
            f"mysql+aiomysql://{quote_plus(self.db_user)}:{quote_plus(self.db_password)}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset={self.db_charset}"
        )

    @property
    def mix_retrieve_url(self) -> str:
        """跨库召回地址：最终答案检索用（`kbs:mix-retrieve`）。"""
        if self.kb_mix_retrieve_url:
            return self.kb_mix_retrieve_url
        return self._platform_url("kbs:mix-retrieve")

    @property
    def faq_retrieve_url(self) -> str:
        """FAQ 探测地址。

        与 `mix_retrieve_url` 是同一个接口的两处部署（请求体与响应结构一致，只是知识库
        可能挂在不同网关地址上），因此可以各自整条覆盖；FAQ 地址没单独配时复用跨库地址，
        保持「两通道同端点」这种最常见的情况只需配一个变量。
        """
        return self.kb_faq_retrieve_url or self.mix_retrieve_url

    def _platform_url(self, endpoint: str) -> str:
        base = self.kb_platform_base_url.rstrip("/")
        return f"{base}/applet/api/v1/knowlhub/{endpoint}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
