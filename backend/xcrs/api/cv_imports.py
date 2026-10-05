"""CV import endpoints (ADR-0045): signed-in users upload a PDF or paste text; the import runs as an in-memory job
(one at a time) and its suggestions are handed out once. Nothing from the file is stored or logged."""

import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from xcrs.api.identity import required_user
from xcrs.api.limits import heavy
from xcrs.api.schemas_v2 import Experience
from xcrs.config import get_settings
from xcrs.cv.clean import normalize_text
from xcrs.cv.extract import CvExtractor
from xcrs.cv.pdf_text import CvInputError, pdf_text
from xcrs.db.models import User
from xcrs.db.session import new_session
from xcrs.services.cv_import import CvImporter, CvImportJobs, ImportResult, QueueFull, QuotaExceeded
from xcrs.services.skill_matching import CONFIRM_FLOOR, build_matcher, lexical_index, skill_names

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/cv-imports", tags=["cv-import"])

MIN_CHARS_PER_PAGE = 50  # less than this per page: a scanned PDF without a text layer
MIN_PASTED_CHARS = 40

MESSAGES = {
    "not_pdf": "That file isn't a PDF. Upload a PDF, or paste the text instead.",
    "encrypted": "That PDF is password-protected. Save it without a password, or paste the text instead.",
    "too_many_pages": "That PDF has more pages than we read (5). Paste the most relevant part instead.",
    "too_large": "That file is larger than 2 MB. Paste the text instead.",
    "timeout": "We couldn't read that PDF in time. Paste the text instead.",
    "unreadable": "We couldn't read that PDF. Paste the text instead.",
    "scanned": "This PDF looks scanned: it has no text we can read. Paste the text instead.",
    "too_long": "That's more text than we read (20,000 characters). Paste the most relevant part.",
    "empty": "There's too little text to find skills in.",
    "no_input": "Upload a PDF or paste the text of your CV.",
}


class CvSuggestionV2(BaseModel):
    skill: str
    name: str
    proficiency: int | None  # suggested 1–3 from years of use; None when the CV gives no dates for it
    years: float | None
    evidence: str  # a short quote from the CV (render as text)
    job: int | None  # index into `jobs` of the newest job it was used in; None = summary or skills list
    source: Literal["llm", "scan"]


class CvJobV2(BaseModel):
    title: str
    employer: str
    start: str | None
    end: str | None
    kind: str


class CvWarningsV2(BaseModel):
    hidden: int
    instructions: int
    examples: list[str]


class CvImportResultV2(BaseModel):
    suggestions: list[CvSuggestionV2]
    experience: Experience | None
    jobs: list[CvJobV2]
    warnings: CvWarningsV2


class CvImportStatusV2(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "failed"]
    position: int  # in the queue; 0 once running or finished
    result: CvImportResultV2 | None = None


def run_import(text: str, hidden: list[str]) -> ImportResult:
    """One import, in the worker thread, with its own database session."""
    settings = get_settings()
    session = new_session()
    try:
        names = skill_names(session)
        extractor = CvExtractor(
            settings.llm_base_url,
            settings.llm_model,
            names,
            shape="pick" if settings.cv_shape == "pick" else "phrases",
            timeout_s=settings.cv_llm_timeout_s,
            max_tokens=settings.cv_llm_max_tokens,
            disable_thinking=settings.llm_disable_thinking,
            api_key=settings.llm_api_key,
        )
        matcher = build_matcher(session)

        def confirm(evidence: str, skill: str) -> bool:
            vector = matcher.embedder.embed_query([evidence])[0]
            return matcher.store.similarities(vector).get(skill, 0.0) >= CONFIRM_FLOOR

        importer = CvImporter(lexical_index(session), dict(names), extractor, matcher=matcher, confirm=confirm)
        return importer.run(text, hidden)
    finally:
        session.close()


_jobs: CvImportJobs | None = None


def get_cv_jobs() -> CvImportJobs:
    global _jobs
    if _jobs is None:
        settings = get_settings()
        _jobs = CvImportJobs(run_import, max_waiting=settings.cv_max_waiting, per_day=settings.cv_per_day)
    return _jobs


def _refuse(code: str, status: int = 422) -> HTTPException:
    return HTTPException(status, {"code": code, "message": MESSAGES[code]})


def _available() -> None:
    settings = get_settings()
    if not settings.cv_import_enabled:
        raise HTTPException(404, "Not Found")
    if not settings.llm_enabled:
        raise HTTPException(503, "CV import needs the language model, which is off on this server.")


def read_input(file: UploadFile | None, text: str | None) -> tuple[str, list[str]]:
    """The visible text and the hidden pieces, or a CvInputError."""
    settings = get_settings()
    if file is not None:
        data = file.file.read(settings.cv_max_pdf_bytes + 1)
        if len(data) > settings.cv_max_pdf_bytes:
            raise CvInputError("too_large")
        pdf = pdf_text(data, max_pages=settings.cv_max_pages)
        visible = normalize_text(pdf.text)
        if len(visible) < MIN_CHARS_PER_PAGE * pdf.pages:
            raise CvInputError("scanned")
        hidden = pdf.hidden
    elif text and text.strip():
        visible, hidden = normalize_text(text), []
        if len(visible) < MIN_PASTED_CHARS:
            raise CvInputError("empty")
    else:
        raise CvInputError("no_input")
    if len(visible) > settings.cv_max_chars:
        raise CvInputError("too_long")
    return visible, hidden


@router.post("", status_code=202, response_model=CvImportStatusV2, dependencies=[Depends(_available), Depends(heavy)])
def create_cv_import(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None, max_length=100_000),
    user: User = Depends(required_user),
    jobs: CvImportJobs = Depends(get_cv_jobs),
) -> CvImportStatusV2:
    try:
        visible, hidden = read_input(file, text)
    except CvInputError as exc:
        raise _refuse(exc.code, 413 if exc.code == "too_large" else 422) from exc
    try:
        job, position = jobs.submit(str(user.id), visible, hidden)
    except QuotaExceeded as exc:
        raise HTTPException(
            429, {"code": "quota", "message": "You've used today's CV imports. Try again tomorrow."}
        ) from exc
    except QueueFull as exc:
        raise HTTPException(
            503,
            {"code": "busy", "message": "Other imports are running. Try again in a minute."},
            headers={"Retry-After": "60"},
        ) from exc
    log.info("CV import queued (%s, %d characters, position %d)", "pdf" if file else "text", len(visible), position)
    return CvImportStatusV2(id=job.id, status=job.status, position=position)


@router.get("/{job_id}", response_model=CvImportStatusV2, dependencies=[Depends(_available)])
def read_cv_import(
    job_id: str, user: User = Depends(required_user), jobs: CvImportJobs = Depends(get_cv_jobs)
) -> CvImportStatusV2:
    found = jobs.get(str(user.id), job_id)
    if found is None:
        raise HTTPException(404, "No such import (results are kept for 15 minutes and shown once).")
    job, position = found
    out = CvImportStatusV2(id=job.id, status=job.status, position=position)
    if job.result is not None:
        r = job.result
        out.result = CvImportResultV2(
            suggestions=[
                CvSuggestionV2(
                    skill=s.skill,
                    name=s.name,
                    proficiency=s.proficiency,
                    years=round(s.months / 12, 1) if s.months else None,
                    evidence=s.evidence,
                    job=s.job,
                    source=s.source,
                )
                for s in r.suggestions
            ],
            experience=r.experience,
            jobs=[CvJobV2(**j) for j in r.jobs],
            warnings=CvWarningsV2(
                hidden=r.warnings.hidden, instructions=r.warnings.instructions, examples=r.warnings.examples
            ),
        )
    return out
