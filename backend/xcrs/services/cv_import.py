"""CV import (ADR-0045): from the visible text of a CV to suggestions for the skill board, and the in-memory job
queue that runs one import at a time.

Order: clean (normalize; set aside sentences that read as instructions to an AI) → scan the text for catalog names
→ one LLM call for jobs and skills → keep a skill only if its evidence is really in the text → resolve to catalog
ids (the matcher, or the LLM's own pick confirmed by similarity) → years, levels and the experience band by rule.

Nothing is stored: jobs live in memory and are deleted once their result is read, or after `ttl_s`. Logs never
contain CV text.
"""

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol

from xcrs.cv.clean import normalize_text, remove_instructions
from xcrs.cv.extract import CvExtraction, Shape
from xcrs.domain import cv_profile
from xcrs.domain.skill_matching import LexicalIndex

log = logging.getLogger(__name__)

EXAMPLE_CHARS = 140
MAX_EXAMPLES = 3


class Extractor(Protocol):
    shape: Shape

    def extract(self, cv_text: str, found: list[str] | None = None) -> CvExtraction: ...


class PhraseMatcher(Protocol):
    def match(self, phrases: list[str]) -> list: ...  # PhraseResult (xcrs.services.skill_matching)


@dataclass
class Suggestion:
    skill: str
    name: str
    proficiency: int | None
    months: int
    evidence: str
    job: int | None  # the newest job it was used in (index into `jobs`); None = summary or skills list only
    source: Literal["llm", "scan"]


@dataclass
class Warnings:
    hidden: int = 0
    instructions: int = 0
    examples: list[str] = field(default_factory=list)


@dataclass
class ImportResult:
    suggestions: list[Suggestion]
    experience: str | None
    jobs: list[dict]
    warnings: Warnings
    dropped: int = 0  # LLM skills whose evidence was not in the text (not shown; for the benchmark and logs)
    llm_ms: int = 0


def _example(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= EXAMPLE_CHARS else text[: EXAMPLE_CHARS - 1] + "…"


class CvImporter:
    def __init__(
        self,
        index: LexicalIndex,
        names: dict[str, str],
        extractor: Extractor,
        matcher: PhraseMatcher | None = None,
        confirm: Callable[[str, str], bool] | None = None,
        today: date | None = None,
        check_evidence: bool = True,
    ):
        """`matcher` resolves phrases (shape "phrases"); `confirm(evidence, skill)` checks a picked id (shape
        "pick"; None keeps every pick)."""
        self.index, self.names, self.extractor, self.matcher, self.confirm = index, names, extractor, matcher, confirm
        self.today, self.check_evidence = today, check_evidence

    def run(self, text: str, hidden: list[str] | None = None) -> ImportResult:
        today_d = self.today or date.today()
        today = (today_d.year, today_d.month)
        clean, instructions = remove_instructions(normalize_text(text))
        hidden = [h for h in (hidden or []) if h.strip()]
        warnings = Warnings(
            hidden=len(hidden),
            instructions=len(instructions),
            examples=[_example(t) for t in [*instructions, *hidden][:MAX_EXAMPLES]],
        )
        scanned = cv_profile.scan(self.index, clean)

        started = time.perf_counter()
        if self.extractor.shape == "found":  # the model only says where the scanned skills were used
            extraction = self.extractor.extract(clean, found=sorted(scanned))
        else:
            extraction = self.extractor.extract(clean)
        llm_ms = int((time.perf_counter() - started) * 1000)

        text_words = cv_profile.words(clean)
        kept = []
        for skill in extraction.skills:
            if (
                not self.check_evidence
                or cv_profile.evidence_found(skill.evidence, text_words)
                or cv_profile.evidence_found(skill.name, text_words)
            ):
                kept.append(skill)
        dropped = len(extraction.skills) - len(kept)

        # Only positions whose start year is in the text (an invented one would set the experience band); skills
        # keep pointing at the same positions.
        real = [i for i, j in enumerate(extraction.jobs) if cv_profile.dated_in_text(j.start, clean)]
        renumber = {old: new for new, old in enumerate(real)}
        extraction.jobs = [extraction.jobs[i] for i in real]
        for skill in extraction.skills:
            skill.jobs = [renumber[i] for i in skill.jobs if i in renumber]
        extraction.found_jobs = {
            sid: [renumber[i] for i in idx if i in renumber] for sid, idx in extraction.found_jobs.items()
        }
        jobs = [
            cv_profile.Job(
                cv_profile.parse_month(j.start),
                None if cv_profile.is_present(j.end) else cv_profile.parse_month(j.end, end=True) or today,
                j.kind,
            )
            for j in extraction.jobs
        ]
        found: dict[str, dict] = {}  # skill id -> {"jobs": set, "evidence": str, "source": str}

        def add(skill_id: str, job_indexes, evidence: str, source: str) -> None:
            if skill_id not in self.names:
                return
            entry = found.setdefault(skill_id, {"jobs": set(), "evidence": evidence, "source": source})
            entry["jobs"].update(i for i in job_indexes if 0 <= i < len(jobs))

        if self.extractor.shape == "phrases" and self.matcher is not None:
            for skill, result in zip(kept, self.matcher.match([s.name for s in kept]), strict=True):
                for skill_id in result.skills:
                    add(skill_id, skill.jobs, skill.evidence, "llm")
        else:
            for skill in kept:
                if self.confirm is None or self.confirm(skill.evidence or skill.name, skill.name):
                    add(skill.name, skill.jobs, skill.evidence, "llm")
        for skill_id, line in scanned.items():
            if skill_id not in found:
                add(skill_id, extraction.found_jobs.get(skill_id, []), line, "scan")

        suggestions = []
        for skill_id, entry in found.items():
            months = cv_profile.months_used(jobs, entry["jobs"], today)
            last = cv_profile.last_used(jobs, entry["jobs"], today)
            suggestions.append(
                Suggestion(
                    skill=skill_id,
                    name=self.names[skill_id],
                    proficiency=cv_profile.suggest_level(months, last, today),
                    months=months,
                    evidence=_example(entry["evidence"]),
                    job=min(entry["jobs"]) if entry["jobs"] else None,
                    source=entry["source"],
                )
            )
        suggestions.sort(key=lambda s: (s.job is None, s.job or 0, -(s.proficiency or 0), s.name.casefold()))
        education = [cv_profile.Education(e.start_year, e.end_year) for e in extraction.education]
        return ImportResult(
            suggestions=suggestions,
            experience=cv_profile.experience_band(jobs, education, today),
            jobs=[j.model_dump() for j in extraction.jobs],
            warnings=warnings,
            dropped=dropped,
            llm_ms=llm_ms,
        )


# --- the job queue -------------------------------------------------------------------------------------------


class QueueFull(Exception):
    pass


class QuotaExceeded(Exception):
    pass


@dataclass
class ImportJob:
    id: str
    user_id: str
    text: str
    hidden: list[str]
    status: Literal["queued", "running", "done", "failed"] = "queued"
    result: ImportResult | None = None
    error: str | None = None
    created: float = field(default_factory=time.monotonic)


class CvImportJobs:
    """One import runs at a time (a CPU LLM gains nothing from two); at most `max_waiting` wait. In memory only:
    one API process (ADR-0018); a restart loses running imports and the learner tries again."""

    def __init__(
        self,
        run: Callable[[str, list[str]], ImportResult],
        max_waiting: int = 3,
        ttl_s: float = 900,
        per_day: int = 5,
        clock: Callable[[], float] = time.monotonic,
        start_worker: bool = True,
    ):
        self._run, self.max_waiting, self.ttl_s, self.per_day, self.clock = run, max_waiting, ttl_s, per_day, clock
        self._jobs: dict[str, ImportJob] = {}
        self._queue: deque[str] = deque()
        self._submitted: dict[str, list[float]] = {}  # user id -> submit times (quota)
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._worker: threading.Thread | None = None
        self._start_worker = start_worker

    def submit(self, user_id: str, text: str, hidden: list[str]) -> tuple[ImportJob, int]:
        now = self.clock()
        with self._lock:
            self._expire(now)
            recent = [t for t in self._submitted.get(user_id, []) if now - t < 86_400]
            if len(recent) >= self.per_day:
                raise QuotaExceeded
            if len(self._queue) >= self.max_waiting:
                raise QueueFull
            job = ImportJob(str(uuid.uuid4()), user_id, text, hidden, created=now)
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._submitted[user_id] = [*recent, now]
            position = len(self._queue)
            self._ensure_worker()
            self._wake.notify()
        return job, position

    def get(self, user_id: str, job_id: str) -> tuple[ImportJob, int] | None:
        """The job and its place in the queue (0 = running or finished). A finished job is handed out once."""
        with self._lock:
            self._expire(self.clock())
            job = self._jobs.get(job_id)
            if job is None or job.user_id != user_id:
                return None
            position = self._queue.index(job_id) + 1 if job_id in self._queue else 0
            if job.status in ("done", "failed"):
                del self._jobs[job_id]
            return job, position

    def run_next(self) -> bool:
        """Run one queued job in this thread (the worker's loop; tests call it directly)."""
        with self._lock:
            if not self._queue:
                return False
            job = self._jobs.get(self._queue.popleft())
            if job is None:
                return True
            job.status = "running"
        try:
            result, error = self._run(job.text, job.hidden), None
        except Exception as exc:  # any failure ends this import; the next one runs
            log.warning("CV import %s failed: %s", job.id, type(exc).__name__)
            result, error = None, "failed"
        with self._lock:
            job.text, job.hidden = "", []  # the CV is not kept once read
            job.result, job.error = result, error
            job.status = "done" if result is not None else "failed"
        return True

    def _expire(self, now: float) -> None:
        for job_id in [j.id for j in self._jobs.values() if now - j.created > self.ttl_s and j.status != "running"]:
            del self._jobs[job_id]
            if job_id in self._queue:
                self._queue.remove(job_id)

    def _ensure_worker(self) -> None:
        if self._start_worker and (self._worker is None or not self._worker.is_alive()):
            self._worker = threading.Thread(target=self._loop, name="cv-import", daemon=True)
            self._worker.start()

    def _loop(self) -> None:
        while True:
            with self._lock:
                while not self._queue:
                    self._wake.wait()
            self.run_next()
