from xcrs.config import get_settings
from xcrs.explain.base import Explainer, NoExplainer
from xcrs.explain.llm import LangChainExplainer


def get_explainer() -> Explainer:
    settings = get_settings()
    if not settings.llm_enabled:
        return NoExplainer()
    return LangChainExplainer(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout_s=settings.llm_timeout_s,
        disable_thinking=settings.llm_disable_thinking,
        api_key=settings.llm_api_key,
    )
