"""Explainer benchmark: speed and grounding per model and device (ADR-0020).

    uv run python eval/bench_explainer.py                          # 9B vs 4B, GPU and CPU
    uv run python eval/bench_explainer.py --devices cpu --threads 16   # on a VM

Uses Ollama's native API: it reports token timings and can force CPU inference (num_gpu=0), which the
OpenAI-compatible endpoint can't. The prompt and the output schema are the objects the app uses
(xcrs.explain), so the results carry over. Explanation inputs come from the real algorithm run on the
profiles in eval/profiles.json. Results: printed as Markdown and saved to eval/results/.
"""

import argparse
import json
import os
import statistics
import threading
import time
from datetime import UTC, datetime

import httpx
from common import RESULTS_DIR, grounding_flags, load_profiles, to_user_input
from pydantic import ValidationError

from xcrs.db.session import new_session
from xcrs.explain.base import RoleContext
from xcrs.explain.llm import RoleExplanationOut
from xcrs.explain.prompts import PROMPT_VERSION, build_payload, build_system_prompt
from xcrs.services.recommend import RecommendationService

NS = 1e9
EMBED_MODEL = "qwen3-embedding:0.6b"
EMBED_PHRASES = ["Java", "SQL", "Spring Boot", "HTML", "PHP", "Docker", "Kubernetes", "React", "Python", "Linux"]


def build_cases(max_roles: int) -> list[tuple[str, str, RoleContext, int]]:
    """(profile id, role, explanation input, rank) for the first `max_roles` roles of every profile."""
    cases = []
    with new_session() as session:
        service = RecommendationService(session, explanations=None)
        for profile in load_profiles():
            computed = service.compute(to_user_input(profile))
            for rank, role in enumerate(computed.roles[:max_roles]):
                cases.append((profile["id"], role.role, computed.contexts[role.role_id], rank))
    return cases


def options(device: str, threads: int) -> dict:
    opts = {"temperature": 0.1}
    if device == "cpu":
        opts |= {"num_gpu": 0, "num_thread": threads}
    return opts


def explain(client: httpx.Client, model: str, device: str, threads: int, context: RoleContext) -> dict:
    payload = build_payload(context)
    response = client.post(
        "/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": build_system_prompt(payload)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "format": RoleExplanationOut.model_json_schema(),
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


def contention(client: httpx.Client, model: str, device: str, threads: int, context: RoleContext) -> dict:
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
        "prefill_tok_s": round(statistics.median(r["prefill_tok_s"] for r in ok), 1) if ok else None,
        "decode_tok_s": round(statistics.median(r["decode_tok_s"] for r in ok), 1) if ok else None,
        "clean_pct": round(100 * sum(not r["flags"] for r in ok) / len(ok)) if ok else None,
        "flags": sorted({f.split(":")[0] for r in ok for f in r["flags"]}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--models", nargs="+", default=["qwen3.5:9b", "qwen3.5:4b"])
    parser.add_argument("--devices", nargs="+", default=["gpu", "cpu"], choices=["gpu", "cpu"])
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--max-roles", type=int, default=3, help="roles per profile on GPU")
    parser.add_argument("--cpu-max-roles", type=int, default=1, help="roles per profile on CPU (slow)")
    args = parser.parse_args()

    cases = build_cases(max(args.max_roles, args.cpu_max_roles))
    print(f"{len(cases)} explanation inputs from {len(load_profiles())} profiles; prompt {PROMPT_VERSION}")
    client = httpx.Client(base_url=args.ollama, timeout=900)
    report = {"date": datetime.now(UTC).isoformat(), "host": os.uname().nodename, "threads": args.threads, "runs": {}}

    for model in args.models:
        for device in args.devices:
            limit = args.max_roles if device == "gpu" else args.cpu_max_roles
            selected = [(p, role, ctx) for p, role, ctx, rank in cases if rank < limit]
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
                        "prefill_tok_s": r.get("prompt_eval_count", 0)
                        / max(r.get("prompt_eval_duration", 1) / NS, 1e-9),
                        "decode_tok_s": r["eval_count"] / (r["eval_duration"] / NS),
                    }
                    try:
                        out = RoleExplanationOut.model_validate_json(r["message"]["content"])
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
            summary["contention"] = contention(client, model, device, args.threads, selected[0][2])
            report["runs"][key] = {"summary": summary, "runs": runs}

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"bench-explainer-{datetime.now(UTC):%Y%m%d-%H%M}-{os.uname().nodename.split('.')[0]}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(
        "\n| Model / device | Cases | Median s/role | p90 s | Prompt tok | Output tok | Prefill tok/s "
        "| Decode tok/s | Clean % | Flags | Embed idle → during LLM (ms) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, value in report["runs"].items():
        s, c = value["summary"], value["summary"]["contention"]
        during = f"{c['during_llm_ms']:.0f}" if c["during_llm_ms"] else "n/a"
        print(
            f"| {key} | {s['cases']} | {s['median_s']} | {s['p90_s']} | {s['prompt_tokens']} | {s['output_tokens']} "
            f"| {s['prefill_tok_s']} | {s['decode_tok_s']} | {s['clean_pct']} | {', '.join(s['flags']) or '-'} "
            f"| {c['idle_ms']:.0f} → {during} |"
        )
    print(f"\nSaved {path.relative_to(RESULTS_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
