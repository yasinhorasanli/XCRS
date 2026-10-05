"""Cleaning CV text before it reaches the LLM (ADR-0045): normalization, and sentences that read as instructions
to an AI set aside.

The patterns are a warning layer, not the defence: someone can always rephrase. The defence is structural (the CV
is wrapped as data, the answer is schema-bound, every skill must quote visible text and resolve to a catalog id,
and the result reaches only the person who uploaded the file).
"""

import re
import unicodedata

# Sentences people plant in CVs for screening software, in English and Turkish.
INSTRUCTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(instructions?|prompts?|rules|guidelines|everything above)\b",
    r"\b(previous|prior|above|earlier)\s+(instructions?|prompts?)\b",
    r"\bsystem\s+prompt\b",
    r"\byou\s+are\s+(now\s+)?(an?\s+)?(ai|assistant|language\s+model|llm|chatbot|gpt)\b",
    r"\b(note|message|instructions?)\s+(to|for)\s+(the\s+|any\s+)?(ai|llm|model|assistant|gpt|chatgpt|ats|screener|"
    r"screening\s+(tool|software|system))\b",
    r"\b(as|for)\s+(an?\s+)?(ai|llm|language\s+model)\s*(screening|reading|reviewing|evaluating)\b",
    r"\b(rate|rank|score|mark|recommend)\s+(this|the)\s+(candidate|applicant|cv|resume|profile)\b",
    r"\b(list|add|include|mark|rate)\s+(all|every)\s+(the\s+)?skills?\b",
    r"\bönceki\s+(tüm\s+)?(talimat|komut|yönerge)",
    r"\b(talimatları|komutları|yönergeleri)\s+(yok\s+say|görmezden\s+gel|unut|dikkate\s+alma)",
    r"\byapay\s+zek[aâ]\s*(olarak|asistanı|modeli)?\s*[,:]?\s*(bu\s+aday|lütfen|şunu)",
    r"\bsistem\s+(istemi|komutu|promptu)\b",
]
_INSTRUCTION = re.compile("|".join(f"(?:{p})" for p in INSTRUCTION_PATTERNS), re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_FOOTER = re.compile(r"^\s*(page|sayfa)\s+\d+\s*(of|/)\s*\d+\s*$", re.IGNORECASE)
_SPACES = re.compile(r"[ \t\u00a0\u2000-\u200b]+")  # spaces, no-break and typographic spaces


def normalize_text(text: str) -> str:
    """NFKC (ligatures such as "ﬂ" → "fl"), no soft hyphens or page footers, single spaces, no empty runs."""
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    lines = [_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if _FOOTER.match(line):
            continue
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def looks_like_instruction(sentence: str) -> bool:
    return bool(_INSTRUCTION.search(sentence))


def remove_instructions(text: str) -> tuple[str, list[str]]:
    """The text without sentences that read as instructions to an AI, and those sentences."""
    removed: list[str] = []
    lines: list[str] = []
    for line in text.split("\n"):
        kept = []
        for sentence in _SENTENCE.split(line):
            if looks_like_instruction(sentence):
                removed.append(sentence.strip())
            else:
                kept.append(sentence)
        lines.append(" ".join(kept))
    return "\n".join(lines), removed
