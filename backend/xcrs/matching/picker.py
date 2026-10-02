"""The LLM step of skill matching (ADR-0030): pick catalog skills for a phrase, as a LangChain chain
(prompt template | chat model with structured output), like the explainer (ADR-0019)."""

from collections.abc import Iterable

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from xcrs.matching.prompts import PICK_LIMIT, PICK_PROMPT_VERSION_2, PICK_SYSTEM_2, SkillPick

PICK_PROMPT = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{phrase}")])


class LLMSkillPicker:
    """Raises on failure (server down, timeout, invalid answer); the caller falls back to embeddings."""

    prompt_version = PICK_PROMPT_VERSION_2

    def __init__(
        self,
        base_url: str,
        model: str,
        skills: Iterable[tuple[str, str]],
        timeout_s: float = 60.0,
        disable_thinking: bool = True,
        api_key: str | None = None,
        system: str = PICK_SYSTEM_2,
        prompt_version: str = PICK_PROMPT_VERSION_2,
    ):
        self.model = model
        self.prompt_version = prompt_version
        self._system = system.format(catalog="\n".join(f"{sid}: {name}" for sid, name in skills))
        llm = ChatOpenAI(
            base_url=base_url,
            model=model,
            api_key=api_key or "not-needed",
            temperature=0,
            timeout=timeout_s,
            max_tokens=120,
            max_retries=0,
            reasoning_effort="none" if disable_thinking else None,
            use_responses_api=False,
        )
        self._chain = PICK_PROMPT | llm.with_structured_output(SkillPick, method="json_schema")

    def pick(self, phrase: str) -> list[str]:
        out = self._chain.invoke({"system": self._system, "phrase": phrase})
        return list(dict.fromkeys(out.skills))[:PICK_LIMIT]
