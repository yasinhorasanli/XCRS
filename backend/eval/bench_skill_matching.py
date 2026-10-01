"""How should free text reach catalog skills? (ADR-0029 → ADR-0030)

Runs every matching pipeline on eval/skill_matching.yaml, chooses each pipeline's thresholds (similarity
floor and window below the best) by cross-validation, and reports accuracy, precision, recall, rejection
of non-skills and latency.

    uv run python -u eval/bench_skill_matching.py                       # GPU (the Mac): all pipelines
    uv run python -u eval/bench_skill_matching.py --cpu-sample 25 --threads 12   # + CPU latency of the LLM steps
    uv run python eval/bench_skill_matching.py --summarize eval/results/skill-matching-....json

LLM answers are cached in eval/results/skill-matching-llm-cache.json (by prompt version, model, phrase), so
re-runs only call the LLM for new phrases or prompts. Uses Ollama's native API (timings, CPU-only mode).
"""

import argparse
import json
import platform
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import yaml
from common import EVAL_DIR, RESULTS_DIR

from xcrs.catalog import model
from xcrs.domain.skill_matching import LexicalIndex
from xcrs.matching.prompts import (
    DEFINE_PROMPT_VERSION,
    DEFINE_SYSTEM,
    PICK_LIMIT,
    PICK_PROMPT_VERSION,
    PICK_PROMPT_VERSION_2,
    PICK_SYSTEM,
    PICK_SYSTEM_2,
    QUERY_INSTRUCTION,
    PhraseDefinition,
    SkillPick,
)

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "qwen3-embedding:0.6b"
CACHE = RESULTS_DIR / "skill-matching-llm-cache.json"
FLOORS = np.round(np.arange(0.20, 0.86, 0.01), 2)
WINDOWS = np.round(np.arange(0.0, 0.21, 0.01), 2)
LIMIT = 5


# --- Data ------------------------------------------------------------------------------------------


def load_cases() -> list[dict]:
    cases = yaml.safe_load((EVAL_DIR / "skill_matching.yaml").read_text())["cases"]
    for c in cases:
        c["expect"], c["any"], c["accept"] = set(c.get("expect", [])), set(c.get("any", [])), set(c.get("accept", []))
        c["allowed"] = c["expect"] | c["any"] | c["accept"]
    return cases


def skill_document(skill) -> str:
    return f"{skill.name}: {skill.description}"


# --- Model calls -------------------------------------------------------------------------------------


def embed(client: httpx.Client, texts: list[str]) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), 64):
        r = client.post("/api/embed", json={"model": EMBED_MODEL, "input": texts[i : i + 64], "keep_alive": "15m"})
        r.raise_for_status()
        vectors.extend(r.json()["embeddings"])
    m = np.array(vectors, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def chat(client: httpx.Client, llm: str, system: str, user: str, schema: dict, cpu_threads: int | None) -> tuple:
    options = {"temperature": 0, "num_predict": 200}
    if cpu_threads:
        options |= {"num_gpu": 0, "num_thread": cpu_threads}
    started = time.perf_counter()
    r = client.post(
        "/api/chat",
        json={
            "model": llm,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "think": False,
            "stream": False,
            "keep_alive": "15m",
            "options": options,
        },
    )
    r.raise_for_status()
    return r.json()["message"]["content"], (time.perf_counter() - started) * 1000


class LLMCache:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        self.data[key] = value

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False, sort_keys=True))


def define(client, cache: LLMCache, llm: str, phrase: str) -> dict:
    key = f"{DEFINE_PROMPT_VERSION}|{llm}|{phrase}"
    if (hit := cache.get(key)) is None:
        content, ms = chat(client, llm, DEFINE_SYSTEM, phrase, PhraseDefinition.model_json_schema(), None)
        try:
            out = PhraseDefinition.model_validate_json(content)
            hit = {"is_software_skill": out.is_software_skill, "definition": out.definition.strip(), "ms": ms}
        except ValueError:
            hit = {"is_software_skill": True, "definition": "", "ms": ms, "invalid": content}
        cache.put(key, hit)
    return hit


PICK_PROMPTS = {PICK_PROMPT_VERSION: PICK_SYSTEM, PICK_PROMPT_VERSION_2: PICK_SYSTEM_2}


def pick(client, cache: LLMCache, llm: str, phrase: str, catalog_text: str, skill_ids: set[str], version: str):
    key = f"{version}|{llm}|{phrase}"
    if (hit := cache.get(key)) is None:
        system = PICK_PROMPTS[version].format(catalog=catalog_text)
        content, ms = chat(client, llm, system, phrase, SkillPick.model_json_schema(), None)
        try:
            picked = SkillPick.model_validate_json(content).skills
        except ValueError:
            picked = []
        hit = {"skills": [s for s in picked if s in skill_ids], "unknown": [s for s in picked if s not in skill_ids]}
        hit["ms"] = ms
        cache.put(key, hit)
    return hit


# --- Pipelines -----------------------------------------------------------------------------------------
# Each pipeline gives, per case, a fixed set of skills plus (optionally) similarities to every skill that
# the floor/window rule turns into further matches.


class Pipeline:
    def __init__(self, name: str, fixed: list[set[str]], sims: list[np.ndarray | None]):
        self.name, self.fixed, self.sims = name, fixed, sims
        # The best LIMIT skills per case, best first: all that the floor/window rule can ever pick.
        self.top = [None if s is None else [(int(j), float(s[j])) for j in np.argsort(-s)[:LIMIT]] for s in sims]

    def predict(self, i: int, floor: float, window: float, ids: list[str]) -> set[str]:
        out = set(self.fixed[i])
        top = self.top[i]
        if top is not None:
            best = top[0][1]
            out |= {ids[j] for j, sim in top if sim >= floor and sim >= best - window}
        return out

    @property
    def tuned(self) -> bool:
        return any(s is not None for s in self.sims)


class ConfirmedPipeline(Pipeline):
    """LLM picks kept only where the embedding agrees a little: similarity to the phrase >= floor.
    Lookup results are kept as they are. The window is unused."""

    def __init__(self, name: str, fixed: list[set[str]], picks: list[set[str]], sims: list[np.ndarray], ids):
        self.name, self.fixed, self.picks = name, fixed, picks
        position = {s: j for j, s in enumerate(ids)}
        self.pick_sims = [{s: float(v[position[s]]) for s in p} for p, v in zip(picks, sims, strict=True)]
        self.sims = sims

    def predict(self, i: int, floor: float, window: float, ids: list[str]) -> set[str]:
        return set(self.fixed[i]) | {s for s, sim in self.pick_sims[i].items() if sim >= floor}


def score(cases: list[dict], preds: list[set[str]]) -> dict:
    right = tp = predicted = needed = found = 0
    unrelated_ok = unrelated = 0
    for c, p in zip(cases, preds, strict=True):
        ok_recall = c["expect"] <= p and (not c["any"] or bool(c["any"] & p))
        ok_precision = p <= c["allowed"]
        right += ok_recall and ok_precision
        tp += len(p & c["allowed"])
        predicted += len(p)
        needed += len(c["expect"]) + (1 if c["any"] else 0)
        found += len(c["expect"] & p) + (1 if c["any"] and c["any"] & p else 0)
        if c["kind"] == "unrelated":
            unrelated += 1
            unrelated_ok += ok_precision
    precision = tp / predicted if predicted else 1.0
    recall = found / needed if needed else 1.0
    return {
        "accuracy": right / len(cases),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "rejects_unrelated": unrelated_ok / unrelated if unrelated else None,
    }


def tune(p: Pipeline, cases: list[dict], idx: list[int], ids: list[str]) -> tuple[float, float]:
    if not p.tuned:
        return 0.0, 0.0
    best, best_key = (0.0, 0.0), (-1.0, -1.0)
    sub = [cases[i] for i in idx]
    windows = [0.0] if isinstance(p, ConfirmedPipeline) else WINDOWS
    for floor in FLOORS:
        for window in windows:
            m = score(sub, [p.predict(i, floor, window, ids) for i in idx])
            key = (m["accuracy"], m["f1"])
            if key > best_key:
                best, best_key = (float(floor), float(window)), key
    return best


def cross_validate(p: Pipeline, cases: list[dict], ids: list[str], repeats: int = 5) -> dict:
    """Thresholds chosen on half the cases (stratified by kind), measured on the other half."""
    held_out = []
    for seed in range(repeats):
        rng = random.Random(seed)
        folds: list[list[int]] = [[], []]
        for kind in sorted({c["kind"] for c in cases}):
            members = [i for i, c in enumerate(cases) if c["kind"] == kind]
            rng.shuffle(members)
            for n, i in enumerate(members):
                folds[n % 2].append(i)
        for train, test in ((folds[0], folds[1]), (folds[1], folds[0])):
            floor, window = tune(p, cases, train, ids)
            held_out.append(score([cases[i] for i in test], [p.predict(i, floor, window, ids) for i in test]))
    return {k: statistics.mean(m[k] for m in held_out) for k in ("accuracy", "precision", "recall", "f1")} | {
        "accuracy_sd": statistics.pstdev(m["accuracy"] for m in held_out)
    }


# --- Run -------------------------------------------------------------------------------------------------


def run(args) -> dict:
    cat = model.load_catalog()
    cases = load_cases()[: args.limit or None]
    ids = list(cat.skills)
    index = LexicalIndex.build((s.id, s.name, s.onet) for s in cat.skills.values())
    catalog_text = "\n".join(f"{s.id}: {s.name}" for s in cat.skills.values())
    client = httpx.Client(base_url=OLLAMA, timeout=600)
    cache = LLMCache(CACHE)
    phrases = [c["phrase"] for c in cases]

    print(f"{len(cases)} cases, {len(ids)} skills; embedding skills and phrases ...")
    skills_m = embed(client, [skill_document(cat.skills[s]) for s in ids])

    def sims(texts: list[str]) -> list[np.ndarray]:
        return list(embed(client, texts) @ skills_m.T)

    raw = sims(phrases)
    instr = sims([QUERY_INSTRUCTION + p for p in phrases])

    print(f"definitions with {args.llm} ...")
    definitions = []
    for n, p in enumerate(phrases, 1):
        definitions.append(define(client, cache, args.llm, p))
        if n % 50 == 0:
            cache.save()
            print(f"  {n}/{len(phrases)}")
    cache.save()
    texts = [d["definition"] or p for d, p in zip(definitions, phrases, strict=True)]
    definition = sims(texts)
    definition_instr = sims([QUERY_INSTRUCTION + t for t in texts])

    pick_runs = {}
    for version, llm in [(PICK_PROMPT_VERSION, args.llm), *((PICK_PROMPT_VERSION_2, m) for m in args.pick_llms)]:
        print(f"picks from the list: {version} with {llm} ...")
        picked = []
        for n, p in enumerate(phrases, 1):
            picked.append(pick(client, cache, llm, p, catalog_text, set(ids), version))
            if n % 50 == 0:
                cache.save()
                print(f"  {n}/{len(phrases)}")
        cache.save()
        # pick-1 is kept as answered (the measured baseline); later prompts are capped as in production.
        limit = None if version == PICK_PROMPT_VERSION else PICK_LIMIT
        pick_runs[f"{version}, {llm}"] = [{**k, "skills": k["skills"][:limit]} for k in picked]

    lookups = [index.lookup(p) for p in phrases]
    software = [d["is_software_skill"] for d in definitions]
    none: list[set[str]] = [set() for _ in phrases]

    def after_lookup(vectors: list[np.ndarray], gate: bool) -> Pipeline:
        fixed, rest = [], []
        for (found, resolved), v, sw in zip(lookups, vectors, software, strict=True):
            fixed.append(found)
            rest.append(None if resolved or (gate and not sw) else v)
        return Pipeline("", fixed, rest)

    def named(name: str, p: Pipeline) -> Pipeline:
        p.name = name
        return p

    gated = [v if sw else None for v, sw in zip(definition, software, strict=True)]
    gated_instr = [v if sw else None for v, sw in zip(definition_instr, software, strict=True)]
    pipelines = [
        named("embed phrase", Pipeline("", none, raw)),
        named("embed instruction + phrase", Pipeline("", none, instr)),
        named("lookup → embed instruction + phrase", after_lookup(instr, gate=False)),
        named("embed LLM definition (gate)", Pipeline("", none, gated)),
        named("embed instruction + definition (gate)", Pipeline("", none, gated_instr)),
        named("lookup → embed definition (no gate)", after_lookup(definition, gate=False)),
        named("lookup → embed definition (gate)", after_lookup(definition, gate=True)),
        named("lookup → embed instruction + definition (gate)", after_lookup(definition_instr, gate=True)),
        *(
            named(f"{prefix}LLM picks ({run})", Pipeline("", fixed, [None] * len(phrases)))
            for run, picked in pick_runs.items()
            for prefix, fixed in (
                ("", [set(k["skills"]) for k in picked]),
                ("lookup → ", [f if r else f | set(k["skills"]) for (f, r), k in zip(lookups, picked, strict=True)]),
            )
        ),
        *(
            ConfirmedPipeline(
                f"lookup → LLM picks ({run}), confirmed by similarity",
                [f for f, _ in lookups],
                [set() if r else set(k["skills"]) for (_, r), k in zip(lookups, picked, strict=True)],
                instr,
                ids,
            )
            for run, picked in pick_runs.items()
            if not run.startswith(PICK_PROMPT_VERSION + ",")
        ),
    ]

    results = []
    kinds = sorted({c["kind"] for c in cases})
    for p in pipelines:
        cv = cross_validate(p, cases, ids)
        floor, window = tune(p, cases, list(range(len(cases))), ids)
        preds = [p.predict(i, floor, window, ids) for i in range(len(cases))]
        full = score(cases, preds)
        per_kind = {
            k: score(
                [c for c in cases if c["kind"] == k], [x for c, x in zip(cases, preds, strict=True) if c["kind"] == k]
            )["accuracy"]
            for k in kinds
        }
        errors = [
            {
                "phrase": c["phrase"],
                "kind": c["kind"],
                "got": sorted(x),
                "missing": sorted(c["expect"] - x),
                "wrong": sorted(x - c["allowed"]),
            }
            for c, x in zip(cases, preds, strict=True)
            if not (c["expect"] <= x and (not c["any"] or c["any"] & x) and x <= c["allowed"])
        ]
        results.append(
            {
                "pipeline": p.name,
                "floor": floor,
                "window": window,
                "held_out": cv,
                "all_cases": full,
                "per_kind": per_kind,
                "errors": errors,
            }
        )
        acc = f"{cv['accuracy']:.1%} ± {cv['accuracy_sd']:.1%}"
        print(f"  {p.name:48} held-out acc {acc}  (floor {floor}, window {window})")

    def ms(values: list[float]) -> dict:
        values = sorted(values)
        return {"median": statistics.median(values), "p90": values[int(0.9 * (len(values) - 1))], "n": len(values)}

    started = time.perf_counter()
    for p in phrases:
        index.lookup(p)
    lookup_ms = (time.perf_counter() - started) * 1000 / len(phrases)
    embed_one = []
    for p in phrases[:25]:
        t = time.perf_counter()
        embed(client, [QUERY_INSTRUCTION + p])
        embed_one.append((time.perf_counter() - t) * 1000)
    latency = {
        "lookup_ms_per_phrase": lookup_ms,
        "embed_one_phrase_ms": ms(embed_one),
        f"define_{args.llm}_gpu_ms": ms([d["ms"] for d in definitions]),
        **{f"pick ({run}) gpu_ms": ms([k["ms"] for k in picked]) for run, picked in pick_runs.items()},
    }
    if args.cpu_sample:
        sample = random.Random(1).sample(phrases, args.cpu_sample)
        print(f"CPU latency with {args.cpu_llm}, {args.threads} threads, {len(sample)} phrases ...")
        schema_d, schema_p = PhraseDefinition.model_json_schema(), SkillPick.model_json_schema()
        cpu_d = [chat(client, args.cpu_llm, DEFINE_SYSTEM, p, schema_d, args.threads)[1] for p in sample]
        latency[f"define_{args.cpu_llm}_cpu_ms"] = ms(cpu_d)
        system = PICK_SYSTEM_2.format(catalog=catalog_text)
        cpu_p = [chat(client, args.cpu_llm, system, p, schema_p, args.threads)[1] for p in sample[:8]]
        latency[f"pick_{args.cpu_llm}_cpu_ms"] = ms(cpu_p)

    return {
        "date": datetime.now().isoformat(timespec="minutes"),
        "host": platform.node(),
        "cases": len(cases),
        "skills": len(ids),
        "embed_model": EMBED_MODEL,
        "llm": args.llm,
        "prompts": [DEFINE_PROMPT_VERSION, *PICK_PROMPTS],
        "definitions_say_not_software": sorted(p for p, sw in zip(phrases, software, strict=True) if not sw),
        "pipelines": results,
        "latency": latency,
    }


def markdown(report: dict) -> str:
    lines = [
        f"# Skill matching: {report['date']} on {report['host']}",
        "",
        f"{report['cases']} cases, {report['skills']} skills, `{report['embed_model']}`, LLM `{report['llm']}`. "
        "Held-out = thresholds chosen on half the cases (stratified by kind), measured on the other half; "
        "5 x 2-fold. Accuracy = a case is fully right (every expected skill, nothing wrong).",
        "",
        "| Pipeline | Held-out accuracy | Precision | Recall | Rejects non-skills | Floor | Window |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(report["pipelines"], key=lambda r: -r["held_out"]["accuracy"]):
        h, a = r["held_out"], r["all_cases"]
        rejects = "-" if a["rejects_unrelated"] is None else f"{a['rejects_unrelated']:.0%}"
        lines.append(
            f"| {r['pipeline']} | **{h['accuracy']:.1%}** ± {h['accuracy_sd']:.1%} | {h['precision']:.1%} | "
            f"{h['recall']:.1%} | {rejects} | {r['floor']} | {r['window']} |"
        )
    kinds = list(report["pipelines"][0]["per_kind"])
    lines += ["", "Accuracy by kind (thresholds tuned on all cases):", ""]
    lines += ["| Pipeline | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    for r in sorted(report["pipelines"], key=lambda r: -r["held_out"]["accuracy"]):
        lines.append(f"| {r['pipeline']} | " + " | ".join(f"{r['per_kind'][k]:.0%}" for k in kinds) + " |")
    lines += ["", "Latency:", ""]
    for k, v in report["latency"].items():
        lines.append(
            f"- {k}: " + (f"{v:.2f}" if isinstance(v, float) else ", ".join(f"{a} {b:.0f}" for a, b in v.items()))
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", default="qwen3.5:9b")
    parser.add_argument("--cpu-llm", default="qwen3.5:4b")
    parser.add_argument("--pick-llms", nargs="*", default=["qwen3.5:9b", "qwen3.5:4b"], help="models for pick-2")
    parser.add_argument("--cpu-sample", type=int, default=0, help="phrases for CPU latency (0 = skip)")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--summarize", type=Path, help="rebuild the .md of a saved run")
    parser.add_argument("--limit", type=int, default=0, help="only the first N cases (a smoke test; not saved)")
    args = parser.parse_args()
    if args.summarize:
        args.summarize.with_suffix(".md").write_text(markdown(json.loads(args.summarize.read_text())))
        return
    report = run(args)
    if args.limit:
        print(markdown(report))
        return
    stem = RESULTS_DIR / f"skill-matching-{datetime.now():%Y%m%d-%H%M}-{platform.node().split('.')[0]}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    stem.with_suffix(".md").write_text(markdown(report))
    print(markdown(report))
    print(f"saved {stem}.json / .md")


if __name__ == "__main__":
    main()
