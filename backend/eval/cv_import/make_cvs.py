"""Render the evaluation CVs in cases.yaml to HTML (html/<id>.html); print_pdfs.mjs then prints them to PDF.

    uv run python eval/cv_import/make_cvs.py
    cd eval/cv_import && npm i --no-save playwright-core && node print_pdfs.mjs   # needs Google Chrome

Layouts imitate LinkedIn's "Save to PDF" (a grey sidebar beside the main column, a company header over its
positions, durations in brackets, "Page 1 of 2" footers) and classic one-column CVs. The PDFs are committed, so
the benchmark doesn't need Chrome.
"""

import html
from datetime import date
from pathlib import Path

import yaml

HERE = Path(__file__).parent

MONTHS_EN = "January February March April May June July August September October November December"
MONTHS_TR = "Ocak Şubat Mart Nisan Mayıs Haziran Temmuz Ağustos Eylül Ekim Kasım Aralık"
MONTHS = {"en": MONTHS_EN.split(), "tr": MONTHS_TR.split()}
WORDS = {
    "en": {
        "present": "Present",
        "year": ("year", "years"),
        "month": ("month", "months"),
        "contact": "Contact",
        "top": "Top Skills",
        "languages": "Languages",
        "certs": "Certifications",
        "summary": "Summary",
        "experience": "Experience",
        "education": "Education",
        "skills": "Skills",
        "projects": "Projects",
    },
    "tr": {
        "present": "Halen",
        "year": ("yıl", "yıl"),
        "month": ("ay", "ay"),
        "contact": "İletişim Bilgileri",
        "top": "En Önemli Yetenekler",
        "languages": "Diller",
        "certs": "Sertifikalar",
        "summary": "Özet",
        "experience": "Deneyim",
        "education": "Eğitim",
        "skills": "Yetenekler",
        "projects": "Projeler",
    },
}

CSS_LINKEDIN = """
body{font-family:Helvetica,Arial,sans-serif;font-size:10.5pt;margin:0;color:#222}
.wrap{display:grid;grid-template-columns:185px 1fr;gap:26px;padding:30px 34px}
aside{background:#2f3b45;color:#fff;padding:14px;font-size:9.5pt;align-self:stretch}
aside h3{font-size:11pt;margin:14px 0 6px} aside p{margin:0 0 4px}
h1{font-size:24pt;margin:0} h2{font-size:14pt;margin:18px 0 6px}
.head{font-size:12pt;margin:4px 0}.muted{color:#666;margin:2px 0}
.company{font-weight:bold;font-size:11.5pt;margin:12px 0 0}.title{margin:6px 0 0;font-weight:bold}
.text{margin:4px 0 0;white-space:pre-line}
"""
CSS_CLASSIC = """
body{font-family:Georgia,serif;font-size:10.5pt;margin:42px;color:#222}
h1{font-size:22pt;margin:0} h2{font-size:12.5pt;border-bottom:1px solid #999;margin:18px 0 6px}
.muted{color:#555}.job{margin:8px 0 0}.text{margin:2px 0 0 0;white-space:pre-line}
"""
HIDDEN_CSS = ".hidden-white{color:#fff;font-size:9pt}.hidden-tiny{font-size:1pt;color:#333}"


def ym(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    y, m = str(value).split("-")
    return int(y), int(m)


def months_between(start: tuple[int, int], end: tuple[int, int]) -> int:
    return (end[0] - start[0]) * 12 + end[1] - start[1] + 1  # LinkedIn counts both ends


def duration(n: int, lang: str) -> str:
    w = WORDS[lang]
    years, months = divmod(n, 12)
    parts = []
    if years:
        parts.append(f"{years} {w['year'][years != 1]}")
    if months:
        parts.append(f"{months} {w['month'][months != 1]}")
    return " ".join(parts) or f"1 {w['month'][0]}"


def period(job: dict, lang: str, today: tuple[int, int], style: str, with_duration: bool) -> str:
    start, end = ym(job["start"]), ym(job["end"])
    present = WORDS[lang]["present"] if style != "numeric" else ("Günümüz" if lang == "tr" else "now")

    def fmt(d):
        if style == "numeric":
            return f"{d[1]:02d}/{d[0]}"
        name = MONTHS[lang][d[1] - 1]
        return f"{name if style == 'long' else name[:3]} {d[0]}"

    text = f"{fmt(start)} - {fmt(end) if end else present}"
    if with_duration:
        text += f" ({duration(months_between(start, end or today), lang)})"
    return text


def esc(text: str) -> str:
    return html.escape(text.strip())


def text_block(text: str) -> str:
    return f'<p class="text">{esc(text)}</p>'


def hidden_html(case: dict) -> str:
    items = case.get("hidden") or []
    classes = ["hidden-white", "hidden-tiny"]
    return "".join(f'<p class="{classes[i % 2]}">{esc(t)}</p>' for i, t in enumerate(items))


def linkedin(case: dict, today: tuple[int, int]) -> str:
    cv, lang = case["cv"], case["ui"]
    w = WORDS[lang]
    side = [f"<h3>{w['contact']}</h3><p>{esc(cv['contact'])}</p>"]
    slug = cv["name"].lower().replace(" ", "-")
    side.append(f"<p>www.linkedin.com/in/{esc(slug)}-x1 (LinkedIn)</p>")
    for key, items in (
        ("top", cv.get("top_skills")),
        ("languages", cv.get("languages")),
        ("certs", cv.get("certifications")),
    ):
        if items:
            side.append(f"<h3>{w[key]}</h3>" + "".join(f"<p>{esc(i)}</p>" for i in items))
    main = [
        f"<h1>{esc(cv['name'])}</h1>",
        f'<p class="head">{esc(cv["headline"])}</p>',
        f'<p class="muted">{esc(cv["location"])}</p>',
        f"<h2>{w['summary']}</h2>",
        text_block(cv["summary"]),
    ]
    if cv.get("jobs"):
        main.append(f"<h2>{w['experience']}</h2>")
        jobs = cv["jobs"]
        i = 0
        while i < len(jobs):  # LinkedIn groups consecutive positions at one employer under the company
            group = [jobs[i]]
            while i + len(group) < len(jobs) and jobs[i + len(group)]["employer"] == jobs[i]["employer"]:
                group.append(jobs[i + len(group)])
            main.append(f'<p class="company">{esc(group[0]["employer"])}</p>')
            if len(group) > 1:
                total = months_between(ym(group[-1]["start"]), ym(group[0]["end"]) or today)
                main.append(f'<p class="muted">{duration(total, lang)}</p>')
            for job in group:
                main.append(f'<p class="title">{esc(job["title"])}</p>')
                main.append(f'<p class="muted">{period(job, lang, today, "long", True)}</p>')
                if job.get("location"):
                    main.append(f'<p class="muted">{esc(job["location"])}</p>')
                main.append(text_block(job["text"]))
            i += len(group)
    if cv.get("education"):
        main.append(f"<h2>{w['education']}</h2>")
        for ed in cv["education"]:
            main.append(f'<p class="company">{esc(ed["school"])}</p><p>{esc(ed["degree"])} · ({esc(ed["years"])})</p>')
    main.append(hidden_html(case))
    return (
        f"<style>{CSS_LINKEDIN}{HIDDEN_CSS}</style><div class='wrap'><aside>{''.join(side)}</aside>"
        f"<main>{''.join(main)}</main></div>"
    )


def classic(case: dict, today: tuple[int, int]) -> str:
    cv, lang = case["cv"], case["ui"]
    w = WORDS[lang]
    out = [
        f"<h1>{esc(cv['name'])}</h1>",
        f'<p class="muted">{esc(cv["headline"])} · {esc(cv["location"])} · {esc(cv["contact"])}</p>',
        f"<h2>{w['summary']}</h2>",
        text_block(cv["summary"]),
    ]
    if cv.get("jobs"):
        out.append(f"<h2>{w['experience']}</h2>")
        for job in cv["jobs"]:
            style = job.get("date_style", "short")
            out.append(
                f'<p class="job"><b>{esc(job["title"])}</b> — {esc(job["employer"])} · '
                f"{period(job, lang, today, style, False)}</p>"
            )
            out.append(text_block(job["text"]))
    if cv.get("projects"):
        out += [f"<h2>{w['projects']}</h2>", text_block(cv["projects"])]
    if cv.get("education"):
        out.append(f"<h2>{w['education']}</h2>")
        out += [f"<p>{esc(ed['school'])}, {esc(ed['degree'])} ({esc(ed['years'])})</p>" for ed in cv["education"]]
    if cv.get("skills_line"):
        out += [f"<h2>{w['skills']}</h2>", f"<p>{esc(cv['skills_line'])}</p>"]
    out.append(hidden_html(case))
    return f"<style>{CSS_CLASSIC}{HIDDEN_CSS}</style>{''.join(out)}"


def main() -> None:
    data = yaml.safe_load((HERE / "cases.yaml").read_text())
    today = data["today"] if isinstance(data["today"], date) else date.fromisoformat(data["today"])
    out = HERE / "html"
    out.mkdir(exist_ok=True)
    for case in data["cases"]:
        body = (
            linkedin(case, (today.year, today.month))
            if case["layout"] == "linkedin"
            else classic(case, (today.year, today.month))
        )
        (out / f"{case['id']}.html").write_text(f'<!doctype html><meta charset="utf-8">{body}')
    print(f"{len(data['cases'])} CVs → {out}")


if __name__ == "__main__":
    main()
