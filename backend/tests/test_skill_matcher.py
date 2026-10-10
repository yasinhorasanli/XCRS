"""Skill matching service (ADR-0030): order of the layers, confirmation, cache and fallback, with fakes."""

import openai
import pytest

from xcrs.catalog import model
from xcrs.domain.skill_matching import LexicalIndex
from xcrs.services.skill_matching import CONFIRM_FLOOR, FALLBACK_FLOOR, SkillMatcher


class FakeEmbedder:
    def embed_query(self, texts):
        return [texts[0]]  # the "vector" is the phrase; FakeStore looks similarities up by phrase


class FakeStore:
    catalog_checksum = "c1"

    def __init__(self, sims: dict[str, dict[str, float]]):
        self.sims, self.cache, self.saved = sims, {}, []

    def similarities(self, vector):
        return self.sims.get(vector, {})

    def cached(self, key, picker):
        return self.cache.get(key)

    def save(self, key, picker, skills, picked, llm_ms):
        self.cache[key] = skills
        self.saved.append((key, skills, picked))


class FakePicker:
    prompt_version, model = "pick-2", "fake"

    def __init__(self, answers: dict[str, list[str]], fail: bool = False):
        self.answers, self.fail, self.calls = answers, fail, 0

    def pick(self, phrase):
        self.calls += 1
        if self.fail:
            raise openai.APITimeoutError(request=None)
        return self.answers.get(phrase, [])


@pytest.fixture(scope="module")
def index():
    cat = model.load_catalog()
    return LexicalIndex.build((s.id, s.name, s.onet) for s in cat.skills.values())


def matcher(index, sims, answers=None, fail=False):
    store, picker = FakeStore(sims), FakePicker(answers or {}, fail)
    return SkillMatcher(index, store, FakeEmbedder(), picker), store, picker


def test_names_are_matched_without_the_llm(index):
    m, _, picker = matcher(index, {})
    [r] = m.match(["Kubernets"])
    assert (r.skills, r.method) == (["kubernetes"], "lookup") and picker.calls == 0


def test_llm_picks_are_kept_only_when_the_embedding_agrees(index):
    sims = {"carpentry": {"git": CONFIRM_FLOOR - 0.1, "linux": 0.1}, "Jira": {"agile-scrum": CONFIRM_FLOOR + 0.05}}
    m, store, _ = matcher(index, sims, {"carpentry": ["git", "linux"], "Jira": ["agile-scrum"]})
    carpentry, jira = m.match(["carpentry", "Jira"])
    assert (carpentry.skills, carpentry.method) == ([], "none")
    assert (jira.skills, jira.method) == (["agile-scrum"], "llm")
    assert ("carpentry", [], ["git", "linux"]) in store.saved  # what the LLM said is kept for auditing


def test_a_cached_answer_is_reused(index):
    m, _, picker = matcher(index, {"Jira": {"agile-scrum": 0.6}}, {"Jira": ["agile-scrum"]})
    m.match(["Jira"])
    [again] = m.match(["  jira "])
    assert (again.skills, again.method, picker.calls) == (["agile-scrum"], "cache", 1)


def test_named_parts_are_kept_next_to_the_llm_picks(index):
    phrase = "HTML CSS and JavaScript"
    sims = {phrase: {"html": 0.6, "css": 0.6, "javascript": 0.7}}
    m, _, _ = matcher(index, sims, {phrase: ["html", "css", "javascript"]})
    [r] = m.match([phrase])
    assert r.skills == ["javascript", "html", "css"] and r.method == "llm"


def test_without_the_llm_the_most_similar_skill_counts_if_similar_enough(index):
    sims = {"Jira": {"jenkins": FALLBACK_FLOOR + 0.01, "git": 0.2}, "cooking": {"git": FALLBACK_FLOOR - 0.1}}
    m, store, _ = matcher(index, sims, fail=True)
    jira, cooking = m.match(["Jira", "cooking"])
    assert (jira.skills, jira.method) == (["jenkins"], "embedding")
    assert (cooking.skills, cooking.method) == ([], "none")
    assert store.saved == []  # fallback answers aren't cached, so the LLM is asked again next time


def test_blank_input_matches_nothing(index):
    m, _, picker = matcher(index, {})
    assert m.match(["  ", "!!"])[0].method == "none" and picker.calls == 0
