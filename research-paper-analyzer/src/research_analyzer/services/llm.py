from pathlib import Path
import httpx
from openai import OpenAI
from research_analyzer.config import Settings


class LLMService:
    """Separate configurable chat-completions provider; embedding API is not a chat API."""

    def __init__(self, settings: Settings, client: OpenAI | None = None):
        self.model = settings.llm_model
        if client is not None:
            self.client = client
            return
        if not settings.llm_api_key or not settings.llm_model or not settings.llm_base_url:
            raise ValueError("Set LLM_API_KEY, LLM_MODEL and LLM_BASE_URL for analysis")
        if settings.llm_ca_cert and not Path(settings.llm_ca_cert).is_file():
            raise FileNotFoundError(settings.llm_ca_cert)
        http_client = httpx.Client(verify=settings.llm_ca_cert or True, timeout=90)
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            http_client=http_client,
            max_retries=3,
        )

    def complete(self, system: str, user: str) -> str:
        result = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return result.choices[0].message.content or ""
