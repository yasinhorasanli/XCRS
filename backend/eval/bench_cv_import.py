"""Benchmark the CV import (ADR-0045) on eval/cv_import/cases.yaml: skill precision and recall, experience-band
accuracy, injection warnings, and LLM latency per model, device and answer shape.

    uv run python -u eval/bench_cv_import.py                                  # 9B on the GPU, both shapes
    uv run python -u eval/bench_cv_import.py --cpu-model qwen3.5:4b --threads 12   # + 4B on CPU only
    uv run python -u eval/bench_cv_import.py --latency-pdf ~/Downloads/Profile.pdf  # + timing on a private PDF
    uv run python eval/bench_cv_import.py --summarize eval/results/cv-import-....json

LLM answers are cached in eval/results/cv-import-llm-cache.json (per model, device, shape, prompt version and
text), so re-runs only redo the cheap parts. A --latency-pdf is never cached and only its timing is saved.

Precision counts a found skill as right when it is in `expect` or `accept`; recall counts `expect` only.
Variants per model and shape: the full pipeline, without the catalog scan, without the evidence check, and (for
"pick") without the similarity confirmation; plus the scan alone (no LLM).
"""

import argparse
import hashlib
import json
import platform
import time
from datetime import date, datetime
from pathlib import Path

import httpx
import yaml
from common import EVAL_DIR, RESULTS_DIR
from sqlalchemy import select

from xcrs.cv.clean import normalize_text, remove_instructions
from xcrs.cv.extract import CvExtraction, answer_model, parse_answer, prompt_version, system_prompt, user_message
from xcrs.cv.pdf_text import CvInputError, pdf_text
from xcrs.db.models import EmbeddingModel
from xcrs.db.session import new_session
from xcrs.domain.cv_profile import scan
from xcrs.domain.skill_matching import LexicalIndex
from xcrs.embeddings import embedder_for
from xcrs.matching.prompts import QUERY_INSTRUCTION
from xcrs.services.cv_import import CvImporter
from xcrs.services.skill_matching import CONFIRM_FLOOR, DatabaseMatchStore, build_matcher, lexical_index, skill_names

CASES = EVAL_DIR / "cv_import" / "cases.yaml"
PDFS = EVAL_DIR / "cv_import" / "pdf"
CACHE = RESULTS_DIR / "cv-import-llm-cache.json"
OLLAMA = "http://localhost:11434/api/chat"


class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def key(self, *parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))


def chat(client: httpx.Client, model: str, system: str, user: str, schema: dict, threads: int | None) -> dict:
    options = {"temperature": 0.1, "num_ctx": 16384, "num_predict": 3000}
    if threads:
        options |= {"num_gpu": 0, "num_thread": threads}
    started = time.perf_counter()
    r = client.post(
        OLLAMA,
        json={
            "model": model,
            "stream": False,
            "think": False,
            "format": schema,
            "options": options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        },
    ).json()
    return {
        "content": r["message"]["content"],
        "ms": int((time.perf_counter() - started) * 1000),
        "prompt_tokens": r.get("prompt_eval_count"),
        "output_tokens": r.get("eval_count"),
    }


class CachedExtractor:
    def __init__(self, shape: str, extraction: CvExtraction):
        self.shape, self._extraction = shape, extraction

    def extract(self, cv_text: str, found: list[str] | None = None) -> CvExtraction:
        return self._extraction


def score(found: set[str], case: dict) -> dict:
    expect, accept = set(case["expect"]), set(case.get("accept", []))
    right = found & (expect | accept)
    return {
        "found": len(found),
        "right": len(right),
        "hits": len(found & expect),
        "expected": len(expect),
        "wrong": sorted(found - expect - accept),
        "missed": sorted(expect - found),
    }


def totals(rows: list[dict]) -> dict:
    found = sum(r["found"] for r in rows)
    right = sum(r["right"] for r in rows)
    hits = sum(r["hits"] for r in rows)
    expected = sum(r["expected"] for r in rows)
    p = right / found if found else 0.0
    rec = hits / expected if expected else 0.0
    return {
        "precision": round(p, 3),
        "recall": round(rec, 3),
        "f1": round(2 * p * rec / (p + rec), 3) if p + rec else 0,
    }


def run(args) -> dict:
    data = yaml.safe_load(CASES.read_text())
    today = data["today"] if isinstance(data["today"], date) else date.fromisoformat(data["today"])
    cases = [c for c in data["cases"] if not args.only or c["id"] in args.only]
    session = new_session()
    index = lexical_index(session)
    names = dict(skill_names(session))
    model_row = session.scalars(select(EmbeddingModel).where(EmbeddingModel.status == "active")).one()
    store = DatabaseMatchStore(session, model_row)
    embedder = embedder_for(model_row, query_prefix=QUERY_INSTRUCTION)
    matcher = build_matcher(session)
    cache = Cache(CACHE)
    client = httpx.Client(timeout=1800)

    def confirm(evidence: str, skill: str) -> bool:
        return store.similarities(embedder.embed_query([evidence])[0]).get(skill, 0.0) >= CONFIRM_FLOOR

    # Text and input errors first (no LLM).
    texts, inputs = {}, []
    for case in cases:
        entry = {"id": case["id"]}
        try:
            pdf = pdf_text((PDFS / f"{case['id']}.pdf").read_bytes())
            text = normalize_text(pdf.text)
            if len(text) < 50 * pdf.pages:
                raise CvInputError("scanned")
            texts[case["id"]] = (text, pdf.hidden)
            clean, instructions = remove_instructions(text)
            entry |= {"chars": len(text), "hidden": len(pdf.hidden), "instructions": len(instructions)}
            want = case.get("warnings") or {}
            entry["warnings_ok"] = len(pdf.hidden) >= want.get("hidden", 0) and len(instructions) >= want.get(
                "instructions", 0
            )
            entry["false_alarm"] = not want and bool(pdf.hidden or instructions)
        except CvInputError as exc:
            entry["error"] = exc.code
        entry["error_ok"] = entry.get("error") == case.get("error")
        inputs.append(entry)
    print("inputs:", json.dumps(inputs, ensure_ascii=False))

    results = {"inputs": inputs, "runs": [], "latency_pdf": None}
    scored_cases = [c for c in cases if c["id"] in texts]

    # The scan alone.
    rows = []
    for case in scored_cases:
        clean, _ = remove_instructions(texts[case["id"]][0])
        rows.append({"id": case["id"], **score(set(scan(index, clean)), case)})
    results["runs"].append(
        {"model": "-", "device": "-", "shape": "scan only", "variant": "scan only", **totals(rows), "rows": rows}
    )
    print(f"scan only: {totals(rows)}")

    plans = [(m, "gpu", None) for m in args.models]
    if args.cpu_model:
        plans.append((args.cpu_model, "cpu", args.threads))
    for model, device, threads in plans:
        for shape in args.shapes:
            system = system_prompt(shape, names.items())
            answers, latency = {}, []
            for case in scored_cases:
                text, hidden = texts[case["id"]]
                clean, _ = remove_instructions(text)
                found = sorted(scan(index, clean)) if shape == "found" else None
                schema = answer_model(shape, list(names), found).model_json_schema()
                key = cache.key(model, device, shape, prompt_version(shape), clean, *(found or []))
                if key not in cache.data:
                    print(f"  {model} {device} {shape} {case['id']} ...", flush=True)
                    cache.data[key] = chat(client, model, system, user_message(clean, found), schema, threads)
                    cache.save()
                answer = cache.data[key]
                latency.append(
                    {
                        "id": case["id"],
                        "chars": len(clean),
                        **{k: answer[k] for k in ("ms", "prompt_tokens", "output_tokens")},
                    }
                )
                try:
                    answers[case["id"]] = parse_answer(shape, answer["content"])
                except ValueError:
                    answers[case["id"]] = CvExtraction()
            variants = {
                "full": {},
                # "found" relies on the scan (the model only places scanned skills): removing it measures nothing.
                **({} if shape == "found" else {"no scan": {"index": LexicalIndex()}}),
                "no evidence check": {"check_evidence": False},
            }
            if shape in ("pick", "compact", "found"):
                variants["no similarity check"] = {"confirm": None}
            for variant, overrides in variants.items():
                rows = []
                for case in scored_cases:
                    text, hidden = texts[case["id"]]
                    kw = {"index": index, "matcher": matcher, "confirm": confirm, "today": today} | overrides
                    importer = CvImporter(names=names, extractor=CachedExtractor(shape, answers[case["id"]]), **kw)
                    result = importer.run(text, hidden)
                    found = {s.skill for s in result.suggestions}
                    row = {
                        "id": case["id"],
                        **score(found, case),
                        "band": result.experience,
                        "band_ok": result.experience == case.get("band"),
                        "dropped": result.dropped,
                    }
                    if variant == "full":
                        row["levels"] = {s.skill: s.proficiency for s in result.suggestions}
                    rows.append(row)
                run_ = {
                    "model": model,
                    "device": device,
                    "shape": shape,
                    "variant": variant,
                    **totals(rows),
                    "band_exact": round(sum(r["band_ok"] for r in rows) / len(rows), 3),
                    "rows": rows,
                }
                if variant == "full":
                    run_["latency"] = latency
                results["runs"].append(run_)
                print(f"{model} {device} {shape} {variant}: {totals(rows)} band {run_['band_exact']}")
            session.commit()

    if args.latency_pdf:  # a private file: timing only, never cached or stored
        pdf = pdf_text(Path(args.latency_pdf).expanduser().read_bytes())
        clean, _ = remove_instructions(normalize_text(pdf.text))
        timings = []
        for model, device, threads in plans:
            for shape in args.shapes:
                found = sorted(scan(index, clean)) if shape == "found" else None
                schema = answer_model(shape, list(names), found).model_json_schema()
                a = chat(
                    client, model, system_prompt(shape, names.items()), user_message(clean, found), schema, threads
                )
                timings.append(
                    {
                        "model": model,
                        "device": device,
                        "shape": shape,
                        "ms": a["ms"],
                        "prompt_tokens": a["prompt_tokens"],
                        "output_tokens": a["output_tokens"],
                    }
                )
                print("latency pdf:", timings[-1])
        results["latency_pdf"] = {"pages": pdf.pages, "chars": len(clean), "timings": timings}
    session.close()
    return results


def summarize(results: dict) -> str:
    lines = [
        "# CV import benchmark",
        "",
        f"Run: {results.get('when', '')} on {results.get('host', '')}; "
        f"prompt `{results.get('prompt_version', '')}`; cases: `eval/cv_import/cases.yaml`.",
        "",
    ]
    if results.get("note"):
        lines += [f"Note: {results['note']}", ""]
    lines += [
        "## Inputs",
        "",
        "| Case | Chars | Hidden | Instruction lines | Error | Checks |",
        "|---|---|---|---|---|---|",
    ]
    for i in results["inputs"]:
        ok = "ok" if i["error_ok"] and i.get("warnings_ok", True) and not i.get("false_alarm") else "**check**"
        lines.append(
            f"| {i['id']} | {i.get('chars', '')} | {i.get('hidden', '')} | {i.get('instructions', '')} | "
            f"{i.get('error', '')} | {ok} |"
        )
    lines += [
        "",
        "## Skills and experience band",
        "",
        "| Model | Device | Shape | Variant | Precision | Recall | F1 | Band exact |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results["runs"]:
        lines.append(
            f"| {r['model']} | {r['device']} | {r['shape']} | {r['variant']} | {r['precision']:.2f} | "
            f"{r['recall']:.2f} | {r['f1']:.2f} | {r.get('band_exact', '–')} |"
        )
    lines += [
        "",
        "## LLM latency (one call per CV)",
        "",
        "| Model | Device | Shape | Median s | Max s | Median output tokens | Max output tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results["runs"]:
        if r.get("latency"):
            ms = sorted(x["ms"] for x in r["latency"])
            out = sorted(x["output_tokens"] or 0 for x in r["latency"])
            lines.append(
                f"| {r['model']} | {r['device']} | {r['shape']} | {ms[len(ms) // 2] / 1000:.1f} | "
                f"{ms[-1] / 1000:.1f} | {out[len(out) // 2]} | {out[-1]} |"
            )
    if results.get("latency_pdf"):
        lp = results["latency_pdf"]
        lines += ["", f"A real LinkedIn export ({lp['pages']} pages, {lp['chars']} characters; not included):", ""]
        lines += [
            f"- {t['model']} {t['device']} {t['shape']}: {t['ms'] / 1000:.1f} s, {t['prompt_tokens']} prompt "
            f"tokens, {t['output_tokens']} output tokens"
            for t in lp["timings"]
        ]
    lines += ["", "## Per case (full pipeline)", ""]
    for r in results["runs"]:
        if r["variant"] != "full":
            continue
        lines += [
            f"### {r['model']} {r['device']} {r['shape']}",
            "",
            "| Case | Found | Right | Recall | Band | Wrong | Missed |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in r["rows"]:
            rec = f"{row['hits']}/{row['expected']}"
            band = f"{row['band']}" + ("" if row["band_ok"] else " ✗")
            lines.append(
                f"| {row['id']} | {row['found']} | {row['right']} | {rec} | {band} | "
                f"{', '.join(row['wrong'])} | {', '.join(row['missed'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["qwen3.5:9b"])
    parser.add_argument("--shapes", nargs="*", default=["phrases", "pick", "compact", "found"])
    parser.add_argument("--cpu-model", default=None)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--latency-pdf", default=None)
    parser.add_argument("--summarize", default=None)
    args = parser.parse_args()
    if args.summarize:
        path = Path(args.summarize)
        path.with_suffix(".md").write_text(summarize(json.loads(path.read_text())))
        return
    results = run(args)
    results |= {
        "when": f"{datetime.now():%Y-%m-%d %H:%M}",
        "host": platform.node().split(".")[0],
        "prompt_version": "cv-extract-1 (phrases, pick), cv-extract-3 (compact), cv-extract-4 (found)",
    }
    stem = RESULTS_DIR / f"cv-import-{datetime.now():%Y%m%d-%H%M}-{platform.node().split('.')[0]}"
    stem.with_suffix(".json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    stem.with_suffix(".md").write_text(summarize(results))
    print(f"saved {stem}.json / .md")


if __name__ == "__main__":
    main()
