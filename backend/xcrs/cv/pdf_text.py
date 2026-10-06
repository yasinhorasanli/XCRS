"""Text from an uploaded PDF (ADR-0045), with text a reader can't see set aside.

pypdf in plain mode keeps each column of LinkedIn's two-column export whole (measured on a real export; pdfminer's
layout analysis spliced the sidebar into a paragraph). Parsing runs in a child process with a time and memory
limit, because a malformed or deliberately heavy PDF can take minutes or gigabytes; the child gets the bytes on
stdin and answers JSON on stdout.

Hidden text is a known trick for screening software; it is returned separately and never reaches the LLM. Text
counts as hidden when its colour is (nearly) the colour painted beneath it (white paper unless a filled rectangle
lies under it: LinkedIn's own sidebar is white text on a dark panel), when it is rendered invisibly, when it is
under 2 pt, or when it lies outside the page.
"""

import io
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field

MIN_VISIBLE_PT = 2.0
MIN_CONTRAST = 0.1  # luminance difference between text and what lies beneath it
PAINT_OPS = {b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"}


class CvInputError(Exception):
    """The input can't be read. `code` is one of: not_pdf, encrypted, too_many_pages, too_large, timeout,
    unreadable, scanned, too_long, empty."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass
class PdfText:
    text: str
    pages: int
    hidden: list[str] = field(default_factory=list)


def _luminance(op: bytes, args: list) -> float | None:
    """0 (black) to 1 (white) for a fill-colour operator, None if it can't be told (patterns, odd spaces)."""
    try:
        v = [float(a) for a in args]
    except (TypeError, ValueError):
        return None
    if len(v) == 1:
        return v[0]
    if len(v) == 3:
        return 0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2]
    if len(v) == 4:  # CMYK
        return max(0.0, 1 - min(1.0, 0.3 * v[0] + 0.59 * v[1] + 0.11 * v[2] + v[3]))
    return None


def _point(m: list[float], x: float, y: float) -> tuple[float, float]:
    return x * m[0] + y * m[2] + m[4], x * m[1] + y * m[3] + m[5]


def _multiply(a: list[float], b: list[float]) -> list[float]:
    """PDF matrices [a b c d e f]: a × b."""
    return [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]


def read_pdf(data: bytes, max_pages: int) -> PdfText:
    """In the child process: the visible text of every page, and the hidden pieces."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    if not data.startswith(b"%PDF"):
        raise CvInputError("not_pdf")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise CvInputError("encrypted")
        pages = len(reader.pages)
    except PdfReadError as exc:
        raise CvInputError("unreadable", str(exc)) from exc
    if pages > max_pages:
        raise CvInputError("too_many_pages", str(pages))

    visible: list[str] = []
    hidden: list[str] = []
    for page in reader.pages:
        box = page.mediabox
        left, bottom, right, top = float(box.left), float(box.bottom), float(box.right), float(box.top)
        state = {"fill": 0.0, "invisible": False}  # PDF default fill: black
        stack: list[dict] = []
        pending: list[tuple[float, float, float, float]] = []  # rectangles in the current path
        painted: list[tuple[float, float, float, float, float]] = []  # x0, y0, x1, y1, luminance; paint order

        def before(op, args, cm, tm, state=state, stack=stack, pending=pending, painted=painted):
            if op == b"q":
                stack.append(dict(state))
            elif op == b"Q" and stack:
                state.update(stack.pop())
            elif op in (b"rg", b"g", b"k", b"sc", b"scn"):
                lum = _luminance(op, list(args))
                if lum is not None:
                    state["fill"] = lum
            elif op == b"Tr" and args:
                state["invisible"] = int(args[0]) in (3, 7)  # 3: invisible, 7: clipping only
            elif op == b"re" and len(args) == 4:
                x, y, w, h = (float(a) for a in args)
                (x0, y0), (x1, y1) = _point(list(cm), x, y), _point(list(cm), x + w, y + h)
                pending.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
            elif op in PAINT_OPS:
                painted.extend((*r, state["fill"]) for r in pending)
                pending.clear()
            elif op in (b"n", b"S", b"s"):
                pending.clear()

        def text(t, cm, tm, font, size, state=state, painted=painted, box=(left, bottom, right, top)):
            if not t.strip():
                visible.append(t)
                return
            m = _multiply(list(tm), list(cm))
            scale = math.sqrt(abs(m[0] * m[3] - m[1] * m[2]))
            x, y = m[4], m[5]
            outside = not (box[0] - 5 <= x <= box[2] + 5 and box[1] - 5 <= y <= box[3] + 5)
            beneath = next((r[4] for r in reversed(painted) if r[0] <= x <= r[2] and r[1] <= y <= r[3]), 1.0)
            same_colour = abs(state["fill"] - beneath) < MIN_CONTRAST
            if same_colour or state["invisible"] or (size or 0) * scale < MIN_VISIBLE_PT or outside:
                hidden.append(t.strip())
                visible.append("\n")
            else:
                visible.append(t)

        page.extract_text(visitor_operand_before=before, visitor_text=text)
        visible.append("\n")
    return PdfText("".join(visible), pages, [h for h in hidden if h])


def _limit_memory(mb: int) -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (mb * 1024 * 1024, mb * 1024 * 1024))
    except (ImportError, ValueError, OSError):  # macOS doesn't enforce RLIMIT_AS; Linux (the VMs) does
        pass


def pdf_text(data: bytes, max_pages: int = 5, timeout_s: float = 10.0, memory_mb: int = 512) -> PdfText:
    """Parse in a child process. Raises CvInputError."""
    try:
        done = subprocess.run(
            [sys.executable, "-m", "xcrs.cv.pdf_text", str(max_pages)],
            input=data,
            capture_output=True,
            timeout=timeout_s,
            preexec_fn=lambda: _limit_memory(memory_mb),
        )
    except subprocess.TimeoutExpired as exc:
        raise CvInputError("timeout") from exc
    try:
        out = json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        raise CvInputError("unreadable", f"parser exited with {done.returncode}") from exc
    if "error" in out:
        raise CvInputError(out["error"])
    return PdfText(out["text"], out["pages"], out["hidden"])


def _child() -> None:
    max_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    try:
        result = read_pdf(sys.stdin.buffer.read(), max_pages)
        out = {"text": result.text, "pages": result.pages, "hidden": result.hidden}
    except CvInputError as exc:
        out = {"error": exc.code}
    except Exception:  # any parser failure means the file can't be read
        out = {"error": "unreadable"}
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    _child()
