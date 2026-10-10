"""Explainer benchmark: speed and grounding per model and device (ADR-0020).

    uv run python eval/bench_explainer.py                          # 9B vs 4B, GPU and CPU
    uv run python eval/bench_explainer.py --devices cpu --threads 16   # on a VM

Uses Ollama's native API: it reports token timings and can force CPU inference (num_gpu=0), which the
OpenAI-compatible endpoint can't. The prompt and the output schema are the objects the app uses
(xcrs.explain.v2, ADR-0037), so the results carry over. Explanation inputs (facts) come from engine v2 run
on the learner profiles in eval/learner_profiles.yaml, built exactly as the app builds them; nothing is saved.

Results go to eval/results/bench-explainer-<date>-<host>.json (every run) and a .md summary next to it.
Both are rewritten after each model/device phase, so an interrupted run keeps the finished phases.
    uv run python eval/bench_explainer.py --summarize eval/results/<file>.json   # (re)write the .md only

Prefill speed is not reported: Ollama reuses the cached system prompt between calls, so its prompt
timings don't measure a cold prefill (the raw prompt_eval_s per run is kept in the JSON).
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from bench_role_scoring import load_profiles
from common import RESULTS_DIR, grounding_flags
from pydantic import ValidationError
from sqlalchemy import select

from xcrs.db.models import CareerRole
from xcrs.db.session import new_session
from xcrs.domain.role_scoring import score_roles
from xcrs.explain.v2 import PROMPT_VERSION_V2 as PROMPT_VERSION
from xcrs.explain.v2 import RoleExplanationV2Out, build_facts, system_prompt
from xcrs.repository import catalog_store
from xcrs.services.recommend_v2 import LEVEL_NAMES, ROLES_SHOWN, RecommendationServiceV2

NS = 1e9
EMBED_MODEL = "qwen3-embedding:0.6b"
EMBED_PHRASES = ["Java", "SQL", "Spring Boot", "HTML", "PHP", "Docker", "Kubernetes", "React", "Python", "Linux"]


def build_cases(max_roles: int) -> list[tuple[str, str, dict, int]]:
    """(profile id, role, facts, rank) for the first `max_roles` roles of every profile."""
    cases = []
    with new_session() as session:
        snapshot = catalog_store.load_snapshot(session)
        summaries = dict(session.execute(select(CareerRole.slug, CareerRole.summary)).all())
    for profile in load_profiles():
        mentions = profile["mentions"]
        category = {m.skill: m.category for m in mentions}
        matched = [
            {
                "text": snapshot.skill_names[m.skill],
                "category": m.category.value,
                "skills": [{"id": m.skill, "name": snapshot.skill_names[m.skill]}],
            }
            for m in mentions
        ]
        for rank, score in enumerate(score_roles(snapshot, mentions)[: min(max_roles, ROLES_SHOWN)]):
            role = RecommendationServiceV2._role(snapshot, score, category, set(category))
            facts = build_facts(role, matched, summaries.get(role["id"]), LEVEL_NAMES)
            cases.append((profile["id"], role["id"], facts, rank))
    return cases


MAX_TOKENS = 500  # same cap as the app (XCRS_LLM_MAX_TOKENS); hitting it means a runaway generation


def options(device: str, threads: int) -> dict:
    opts = {"temperature": 0.1, "num_predict": MAX_TOKENS}
    if device == "cpu":
        opts |= {"num_gpu": 0, "num_thread": threads}
    return opts


def explain(client: httpx.Client, model: str, device: str, threads: int, facts: dict) -> dict:
    response = client.post(
        "/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt(facts)},
                {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
            ],
            "format": RoleExplanationV2Out.model_json_schema(),
            "think": False,
            "stream": False,
            "keep_alive": "15m",
            "options": options(device, threads),
        },
    )
    response.raise_for_status()
    return response.json()


def embed_ms(client: httpx.Client, device: str, threads: int) -> float:
    started = time.perf_counter()
    client.post(
        "/api/embed", json={"model": EMBED_MODEL, "input": EMBED_PHRASES, "options": options(device, threads)}
    ).raise_for_status()
    return (time.perf_counter() - started) * 1000


def contention(client: httpx.Client, model: str, device: str, threads: int, context: dict) -> dict:
    """Request-path embedding latency, idle vs while an explanation is generating on the same machine."""
    embed_ms(client, device, threads)  # load
    idle = [embed_ms(client, device, threads) for _ in range(5)]
    done = threading.Event()

    def generate():
        try:
            explain(client, model, device, threads, context)
        finally:
            done.set()

    threading.Thread(target=generate, daemon=True).start()
    time.sleep(1.0)  # let generation start
    busy = []
    while not done.is_set() and len(busy) < 5:
        busy.append(embed_ms(client, device, threads))
    done.wait()
    return {"idle_ms": statistics.median(idle), "during_llm_ms": statistics.median(busy) if busy else None}


def summarize(runs: list[dict]) -> dict:
    ok = [r for r in runs if "error" not in r]
    totals = [r["total_s"] for r in ok]
    return {
        "cases": len(runs),
        "errors": len(runs) - len(ok),
        "median_s": round(statistics.median(totals), 1) if totals else None,
        "p90_s": round(sorted(totals)[int(0.9 * (len(totals) - 1))], 1) if totals else None,
        "prompt_tokens": round(statistics.median(r["prompt_tokens"] for r in ok)) if ok else None,
        "output_tokens": round(statistics.median(r["output_tokens"] for r in ok)) if ok else None,
        "decode_tok_s": round(statistics.median(r["decode_tok_s"] for r in ok), 1) if ok else None,
        "clean_pct": round(100 * sum(not r["flags"] for r in ok) / len(ok)) if ok else None,
        "flags": sorted({f.split(":")[0] for r in ok for f in r["flags"]}),
    }


def markdown_summary(report: dict) -> str:
    lines = [
        "# Explainer benchmark",
        "",
        f"- Date: {report['date']}; host: {report['host']}; CPU threads: {report['threads']}"
        + (f"; prompt: {report['prompt_version']}" if "prompt_version" in report else ""),
        "- Inputs: engine v2 facts for `eval/learner_profiles.yaml`; the app's v2 prompt and schema.",
        "- Clean = no automatic grounding flag (heuristics that mark explanations to read, not proof).",
        "- Prefill speed is not reported: Ollama's prompt cache makes its prompt timings unreliable.",
        "",
        "| Model / device | Cases | Errors | Median s/role | p90 s | Prompt tok | Output tok | Decode tok/s | Clean % "
        "| Flags | Model load s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, value in report["runs"].items():
        s = value["summary"]
        lines.append(
            f"| {key} | {s['cases']} | {s['errors']} | {s['median_s']} | {s['p90_s']} | {s['prompt_tokens']} "
            f"| {s['output_tokens']} | {s['decode_tok_s']} | {s['clean_pct']} | {', '.join(s['flags']) or '-'} "
            f"| {s.get('load_s', '-')} |"
        )
    lines += [
        "",
        "## Request-path embedding while an explanation generates on the same device",
        "",
        "| Model / device | Idle ms | During generation ms |",
        "|---|---|---|",
    ]
    for key, value in report["runs"].items():
        c = value["summary"].get("contention")
        if c:
            during = f"{c['during_llm_ms']:.0f}" if c["during_llm_ms"] else "n/a"
            lines.append(f"| {key} | {c['idle_ms']:.0f} | {during} |")
    lines += ["", "## Flagged explanations", ""]
    for key, value in report["runs"].items():
        flagged = [r for r in value["runs"] if r.get("flags")]
        if not flagged:
            continue
        lines += [f"### {key}", ""]
        for r in flagged:
            output = r.get("output")
            text = output.get("explanation", "") if isinstance(output, dict) else str(output)[:200]
            lines.append(f"- **{r['profile']} / {r['role']}**: `{', '.join(r['flags'])}`. {text}")
        lines.append("")
    return "\n".join(lines)


def save(report: dict, path: Path) -> None:
    """Rewrite the JSON (every run) and its Markdown summary."""
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    path.with_suffix(".md").write_text(markdown_summary(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--models", nargs="+", default=["qwen3.5:9b", "qwen3.5:4b"])
    parser.add_argument("--devices", nargs="+", default=["gpu", "cpu"], choices=["gpu", "cpu"])
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--max-roles", type=int, default=3, help="roles per profile on GPU")
    parser.add_argument("--cpu-max-roles", type=int, default=1, help="roles per profile on CPU (slow)")
    parser.add_argument("--limit", type=int, help="at most this many cases per model/device (slow CPUs)")
    parser.add_argument("--no-contention", action="store_true", help="skip the embedding-contention check")
    parser.add_argument("--summarize", type=Path, help="only (re)write the .md summary of an existing results file")
    args = parser.parse_args()

    if args.summarize:
        args.summarize.with_suffix(".md").write_text(markdown_summary(json.loads(args.summarize.read_text())))
        print(f"Wrote {args.summarize.with_suffix('.md')}")
        return

    sys.stdout.reconfigure(line_buffering=True)  # progress shows up in a redirected log as it happens
    cases = build_cases(max(args.max_roles, args.cpu_max_roles))
    print(f"{len(cases)} explanation inputs from {len(load_profiles())} profiles; prompt {PROMPT_VERSION}")
    client = httpx.Client(base_url=args.ollama, timeout=900)
    report = {
        "date": datetime.now(UTC).isoformat(),
        "host": os.uname().nodename,
        "threads": args.threads,
        "prompt_version": PROMPT_VERSION,
        "runs": {},
    }
    path = RESULTS_DIR / f"bench-explainer-{datetime.now(UTC):%Y%m%d-%H%M}-{os.uname().nodename.split('.')[0]}.json"

    for model in args.models:
        for device in args.devices:
            limit = args.max_roles if device == "gpu" else args.cpu_max_roles
            selected = [(p, role, ctx) for p, role, ctx, rank in cases if rank < limit][: args.limit]
            key = f"{model} / {device}"
            warm = explain(client, model, device, args.threads, selected[0][2])  # load; not counted
            runs = []
            for profile_id, role, context in selected:
                run = {"profile": profile_id, "role": role}
                try:
                    r = explain(client, model, device, args.threads, context)
                    run |= {
                        "total_s": r["total_duration"] / NS,
                        "prompt_tokens": r.get("prompt_eval_count", 0),
                        "output_tokens": r["eval_count"],
                        "prompt_eval_s": r.get("prompt_eval_duration", 0) / NS,  # cache-affected, see docstring
                        "decode_tok_s": r["eval_count"] / (r["eval_duration"] / NS),
                    }
                    if r["eval_count"] >= MAX_TOKENS:
                        run |= {"flags": ["runaway"], "output": r["message"]["content"][:2000]}
                    else:
                        try:
                            out = RoleExplanationV2Out.model_validate_json(r["message"]["content"])
                            run |= {"flags": grounding_flags(context, out), "output": out.model_dump()}
                        except ValidationError:
                            run |= {"flags": ["invalid_json"], "output": r["message"]["content"]}
                except httpx.HTTPError as exc:
                    run["error"] = str(exc)
                runs.append(run)
                outcome = run.get("flags", run.get("error"))
                print(f"  {key:22} {profile_id:22} {role:22} {run.get('total_s', 0):6.1f}s {outcome}")
            summary = summarize(runs)
            summary["load_s"] = round(warm.get("load_duration", 0) / NS, 1)
            if not args.no_contention:
                summary["contention"] = contention(client, model, device, args.threads, selected[0][2])
            report["runs"][key] = {"summary": summary, "runs": runs}
            save(report, path)  # after every phase: an interrupted run keeps what finished
            print(f"Saved {key} to {path.relative_to(RESULTS_DIR.parent.parent)}")

    print("\n" + markdown_summary(report))


if __name__ == "__main__":
    main()
