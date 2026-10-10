from xcrs.config import get_settings
from xcrs.explain.v2 import RoleExplainerV2


def get_explainer_v2() -> RoleExplainerV2:
    """Role explanations (ADR-0037), on the same LLM as skill matching."""
    settings = get_settings()
    return RoleExplainerV2(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout_s=settings.llm_timeout_s,
        disable_thinking=settings.llm_disable_thinking,
        api_key=settings.llm_api_key,
        max_tokens=settings.llm_max_tokens,
    )
