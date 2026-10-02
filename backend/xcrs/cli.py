"""XCRS admin commands.

uv run xcrs import-research-data
uv run xcrs register-model qwen3-embedding:0.6b --id 1 --status active
uv run xcrs embed-catalog qwen3-embedding:0.6b
uv run xcrs search "Docker"
uv run xcrs search-courses "Docker"
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from xcrs.catalog import model, validate
from xcrs.config import REPO_ROOT
from xcrs.db.models import EmbeddingModel, RoadmapNode, Role
from xcrs.db.session import new_session
from xcrs.embeddings import embedder_for
from xcrs.ingest import embed_catalog, embed_skills, research_data
from xcrs.repository import catalog_store, vectors
from xcrs.retrieval import CourseRetriever
from xcrs.services import skill_matching

# Known model settings, so registration doesn't depend on remembering prefixes.
# Qwen3-Embedding expects an instruction on queries and nothing on documents (model card).
MODEL_PRESETS = {
    "qwen3-embedding:0.6b": {
        "runtime": "ollama",
        "quantization": "Q8_0",
        "dimensions": 1024,
        "query_prefix": (
            "Instruct: Given a skill, technology, or course that a learner mentions, "
            "retrieve related concepts from software career roadmaps\nQuery:"
        ),
        "document_prefix": None,
    },
}


def cmd_import_research_data(args) -> None:
    with new_session() as session:
        print(json.dumps(research_data.run(session), indent=2))


def cmd_register_model(args) -> None:
    preset = MODEL_PRESETS.get(args.name)
    if preset is None:
        raise SystemExit(f"no preset for {args.name!r}; add one to MODEL_PRESETS")
    values = {"id": args.id, "name": args.name, "status": args.status, **preset}
    with new_session() as session:
        stmt = insert(EmbeddingModel).values(values)
        session.execute(
            stmt.on_conflict_do_update(index_elements=["id"], set_={k: stmt.excluded[k] for k in values if k != "id"})
        )
        session.commit()
    print(f"registered {args.name} as model {args.id} ({args.status})")


def cmd_embed_catalog(args) -> None:
    with new_session() as session:
        print(json.dumps(embed_catalog.run(session, args.model), indent=2))


def cmd_search(args) -> None:
    """Smoke test: which concepts does a phrase match above the model's 2.5-sigma threshold?"""
    with new_session() as session:
        model = embed_catalog.get_model(session, args.model)
        threshold = model.sim_mean + args.sigma * model.sim_std
        vector = embedder_for(model).embed_query([args.phrase])[0]
        matches = vectors.concepts_above_threshold(session, model, [vector], threshold)
        print(
            f"threshold {threshold:.3f} (mean {model.sim_mean:.3f} + {args.sigma}σ {model.sim_std:.3f}); "
            f"{len(matches)} matches"
        )
        names = dict(
            session.execute(
                select(RoadmapNode.id, Role.name + " › " + RoadmapNode.name).join(Role, Role.id == RoadmapNode.role_id)
            ).all()
        )
        for m in matches[: args.limit]:
            print(f"  {m.similarity:.3f}  {names[m.concept_id]}")


def cmd_search_courses(args) -> None:
    """Smoke test for the LangChain retriever: courses nearest to a phrase (k-NN via the HNSW index)."""
    with new_session() as session:
        model = embed_catalog.get_model(session, args.model)
        retriever = CourseRetriever(session=session, model=model, embedder=embedder_for(model), k=args.limit)
        for doc in retriever.invoke(args.phrase):
            print(f"  {doc.metadata['similarity']:.3f}  {doc.metadata['title']}")


def cmd_catalog_validate(args) -> None:
    """Check catalog/*.yaml (ADR-0028). Exit code 1 on errors, so CI blocks the pull request."""
    report = validate.validate(model.load_catalog(args.dir))
    for warning in report.warnings:
        print(f"warning: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print(json.dumps(report.stats, indent=2))
    print(f"{len(report.errors)} errors, {len(report.warnings)} warnings")
    if not report.ok:
        raise SystemExit(1)


def cmd_catalog_path(args) -> None:
    """A role's roadmap up to a level, stage by stage, for reviewing."""
    cat = model.load_catalog(args.dir)
    target = model.RoleLevel.parse(args.role_level)
    for level in cat.roadmaps[target.role].levels:
        print(f"== {target.role}@{level.level}: {level.summary}")
        for stage in level.stages:
            items = ", ".join(
                f"{'|'.join(cat.skills[o].name for o in i.options)} ({model.PROFICIENCY[i.level]})" for i in stage.items
            )
            print(f"   {stage.name}: {items}")
        if level.level == target.level:
            break


def cmd_catalog_bridge(args) -> None:
    """Skills to learn for a move between roles (ADR-0027 transitions)."""
    cat = model.load_catalog(args.dir)
    gap = validate.bridge(cat, model.RoleLevel.parse(args.source), model.RoleLevel.parse(args.target))
    for skill, (have, need) in sorted(gap.items(), key=lambda g: (-g[1][1] + g[1][0], g[0])):
        name = " or ".join(cat.skills[o].name for o in skill.split("|"))
        print(f"  {name:45} {model.PROFICIENCY.get(have, '-'):>8} -> {model.PROFICIENCY[need]}")
    print(f"{len(gap)} skills to learn or deepen")


def _catalog_source(directory: Path) -> tuple[str | None, bool, str]:
    """Git commit, whether catalog/ has uncommitted changes, and a checksum of the YAML files."""

    def git(*cmd: str) -> str | None:
        try:
            return subprocess.run(["git", *cmd], cwd=directory, capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain", "--", "."))
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.yaml")):
        digest.update(str(path.relative_to(directory)).encode() + b"\0" + path.read_bytes())
    return commit.strip() if commit else None, dirty, digest.hexdigest()


def cmd_catalog_import(args) -> None:
    """Load catalog/*.yaml into the `catalog` schema (ADR-0028), in one transaction; only a valid catalog."""
    cat = model.load_catalog(args.dir)
    report = validate.validate(cat)
    if not report.ok:
        for error in report.errors:
            print(f"ERROR: {error}")
        raise SystemExit("not imported: run `xcrs catalog validate` and fix the errors first")
    commit, dirty, checksum = _catalog_source(Path(args.dir))
    with new_session() as session:
        if not args.force and catalog_store.last_import_checksum(session) == checksum:
            print(f"catalog unchanged since the last import (checksum {checksum[:12]}); --force to re-import")
            return
        changes = catalog_store.import_catalog(
            session, cat, git_commit=commit, git_dirty=dirty, checksum=checksum, stats=report.stats
        )
        session.commit()
    print(f"imported catalog from {commit or 'unknown commit'}{' (uncommitted changes)' if dirty else ''}")
    for name in ("skills", "roles", "families", "levels"):
        c = changes[name]
        print(f"  {name:9} +{len(c['added'])} ~{len(c['updated'])} -{len(c['removed'])}", end="")
        listed = [f"{sign}{x}" for sign, key in (("-", "removed"), ("~", "updated")) for x in c[key]][:8]
        print(f"   {' '.join(listed)}" if listed and len(c["added"]) < 50 else "")
    print("  parts    " + ", ".join(f"{k} {v}" for k, v in changes["parts"].items()))
    print("  resources " + ", ".join(f"{k} {v}" for k, v in changes["resources"].items()))


def cmd_catalog_embed(args) -> None:
    """Embed catalog skills with the active model ("name: description"; only new or changed ones)."""
    with new_session() as session:
        model = session.scalars(select(EmbeddingModel).where(EmbeddingModel.status == "active")).one()
        count = embed_skills.embed_skills(session, model)
        name = model.name
        session.commit()
    print(f"{count} skills embedded with {name}")


def cmd_catalog_match(args) -> None:
    """Match phrases to catalog skills as the API does (ADR-0030); shows the method used."""
    with new_session() as session:
        matcher = skill_matching.build_matcher(session, use_llm=not args.no_llm)
        for r in matcher.match(args.phrases):
            print(f"  {r.phrase!r:40} {r.method:9} {', '.join(r.skills) or '-'}")


def cmd_catalog_moves(args) -> None:
    """Every other role ranked by how much of it someone already covers (ADR-0027); * = common path."""
    cat = model.load_catalog(args.dir)
    source = model.RoleLevel.parse(args.role_level)
    print(f"From {source}: share of each role's first level already covered; start = likely starting level")
    for m in validate.moves(cat, source)[: args.top]:
        start = f"{m.starting_level} ({m.starting_coverage:.0%})" if m.starting_level else "-"
        common = f"  * common path to {', '.join(m.common)}" if m.common else ""
        print(f"  {m.coverage:4.0%}  {cat.roles[m.role].name:30} start: {start:16}{common}")


def cmd_resources_ingest(args) -> None:
    """Fetch a source into ingest.raw_records and catalog.learning_resources (ADR-0032, ADR-0033)."""
    from xcrs.config import get_settings
    from xcrs.ingest.resources import freecodecamp, youtube

    with new_session() as session, httpx.Client(timeout=60, follow_redirects=True) as client:
        if args.source == "freecodecamp":
            stats = freecodecamp.ingest(session, client)
        else:
            try:
                stats = youtube.ingest(session, client, get_settings().youtube_api_key, youtube.configured_playlists())
            except youtube.NoApiKey as exc:
                raise SystemExit(str(exc)) from exc
        session.commit()
    print(json.dumps(stats, indent=2))


def cmd_resources_tag(args) -> None:
    """Tag untagged (non-curated) resources with skills: the LLM picks, embeddings confirm (ADR-0030)."""
    from xcrs.config import get_settings
    from xcrs.ingest.resources.tagging import tag_untagged
    from xcrs.matching.picker import LLMSkillPicker
    from xcrs.matching.prompts import QUERY_INSTRUCTION, RESOURCE_PROMPT_VERSION, RESOURCE_SYSTEM

    settings = get_settings()
    with new_session() as session:
        model = session.scalars(select(EmbeddingModel).where(EmbeddingModel.status == "active")).one()
        embedder = embedder_for(model, query_prefix=QUERY_INSTRUCTION)
        picker = LLMSkillPicker(
            settings.llm_base_url,
            settings.llm_model,
            skill_matching.skill_names(session),
            timeout_s=settings.match_llm_timeout_s,
            disable_thinking=settings.llm_disable_thinking,
            api_key=settings.llm_api_key,
            system=RESOURCE_SYSTEM,
            prompt_version=RESOURCE_PROMPT_VERSION,
        )
        stats = tag_untagged(
            session,
            picker,
            lambda t: vectors.skill_similarities(session, model, embedder.embed_query([t])[0]),
            args.limit,
        )
        session.commit()
    print(json.dumps(stats, indent=2))


def cmd_resources_check_links(args) -> None:
    """Fetch every active resource's URL and record its status (never in CI)."""
    from xcrs.ingest.resources.links import check_links

    with new_session() as session:
        result = check_links(session)
        session.commit()
    print(f"{result['checked']} checked, {len(result['broken'])} broken")
    for url, status in result["broken"]:
        print(f"  {status or 'unreachable'}  {url}")


def cmd_resources_expire(args) -> None:
    """Delete YouTube data not refreshed within 30 days (API terms, ADR-0033)."""
    from xcrs.ingest.resources import youtube

    with new_session() as session:
        count = youtube.expire(session)
        session.commit()
    print(f"{count} expired YouTube resources deleted")


def cmd_catalog_coverage(args) -> None:
    """O*NET coverage report of the roadmaps (ADR-0025): writes docs/catalog-coverage.md."""
    from xcrs.catalog import taxonomy

    cat = model.load_catalog(args.dir)
    onet_dir = Path(args.onet_dir)
    if not (onet_dir / taxonomy.SOFTWARE_SKILLS).exists():
        raise SystemExit(f"O*NET text files not found in {onet_dir}; download O*NET 31.0 (data/taxonomy/raw/)")
    reports = taxonomy.coverage(cat, onet_dir)
    Path(args.out).write_text(taxonomy.markdown(cat, reports))
    for r in sorted(reports, key=lambda r: r.share)[:10]:
        print(f"  {r.share:4.0%}  {r.role}")
    print(f"wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="xcrs")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("import-research-data", help="import the research dataset (data/research-2024)")
    p.set_defaults(func=cmd_import_research_data)

    p = sub.add_parser("register-model", help="add or update an embedding model in the registry")
    p.add_argument("name")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--status", choices=["candidate", "active", "retired"], default="candidate")
    p.set_defaults(func=cmd_register_model)

    p = sub.add_parser("embed-catalog", help="embed new/changed items, compute stats and top-k matches")
    p.add_argument("model")
    p.set_defaults(func=cmd_embed_catalog)

    p = sub.add_parser("search", help="smoke test: concepts matching a phrase")
    p.add_argument("phrase")
    p.add_argument("--model", default="qwen3-embedding:0.6b")
    p.add_argument("--sigma", type=float, default=2.5)
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("search-courses", help="smoke test: courses nearest to a phrase (LangChain retriever)")
    p.add_argument("phrase")
    p.add_argument("--model", default="qwen3-embedding:0.6b")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_search_courses)

    catalog = sub.add_parser("catalog", help="the skills/roles/roadmaps catalog in catalog/ (ADR-0028)")
    catalog_sub = catalog.add_subparsers(required=True)
    p = catalog_sub.add_parser("validate", help="check the catalog; exit 1 on errors")
    p.set_defaults(func=cmd_catalog_validate)
    p = catalog_sub.add_parser("path", help="show a role's roadmap up to a level, e.g. backend-engineer@senior")
    p.add_argument("role_level")
    p.set_defaults(func=cmd_catalog_path)
    p = catalog_sub.add_parser("bridge", help="skills to learn for a move, e.g. backend-engineer@mid data-engineer@mid")
    p.add_argument("source")
    p.add_argument("target")
    p.set_defaults(func=cmd_catalog_bridge)
    p = catalog_sub.add_parser("import", help="load the catalog YAML into the database (validated first)")
    p.add_argument("--force", action="store_true", help="import even if nothing changed since the last import")
    p.set_defaults(func=cmd_catalog_import)
    resources = sub.add_parser("resources", help="learning resources: ingest, tag, check links (ADR-0033)")
    resources_sub = resources.add_subparsers(required=True)
    p = resources_sub.add_parser("ingest", help="fetch a source: freecodecamp, or youtube (needs an API key)")
    p.add_argument("source", choices=["freecodecamp", "youtube"])
    p.set_defaults(func=cmd_resources_ingest)
    p = resources_sub.add_parser("tag", help="tag untagged resources with skills (LLM + embeddings)")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_resources_tag)
    p = resources_sub.add_parser("check-links", help="record each resource's HTTP status")
    p.set_defaults(func=cmd_resources_check_links)
    p = resources_sub.add_parser("expire", help="delete YouTube data older than 30 days")
    p.set_defaults(func=cmd_resources_expire)
    p = catalog_sub.add_parser("coverage", help="O*NET coverage report of the roadmaps (docs/catalog-coverage.md)")
    p.add_argument("--onet-dir", default=str(REPO_ROOT / "data" / "taxonomy" / "raw" / "db_31_0_text"))
    p.add_argument("--out", default=str(REPO_ROOT / "docs" / "catalog-coverage.md"))
    p.set_defaults(func=cmd_catalog_coverage)
    p = catalog_sub.add_parser("embed", help="embed catalog skills with the active model (after import)")
    p.set_defaults(func=cmd_catalog_embed)
    p = catalog_sub.add_parser("match", help="match typed phrases to skills, e.g. k8s Jira 'neural networks'")
    p.add_argument("phrases", nargs="+")
    p.add_argument("--no-llm", action="store_true", help="lookup and embeddings only (the fallback)")
    p.set_defaults(func=cmd_catalog_match)
    p = catalog_sub.add_parser("moves", help="rank every other role by distance, e.g. backend-engineer@mid")
    p.add_argument("role_level")
    p.add_argument("--top", type=int, default=25)
    p.set_defaults(func=cmd_catalog_moves)
    for p in catalog_sub.choices.values():
        p.add_argument("--dir", type=Path, default=model.CATALOG_DIR, help="catalog folder (default: repo catalog/)")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
