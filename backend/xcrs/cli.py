"""XCRS admin commands.

uv run xcrs register-model qwen3-embedding:0.6b --id 1 --status active
uv run xcrs catalog import && uv run xcrs catalog embed
uv run xcrs catalog match k8s "neural networks"
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import httpx
import yaml
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from xcrs.catalog import model, validate
from xcrs.config import REPO_ROOT
from xcrs.db.models import EmbeddingModel
from xcrs.db.session import new_session
from xcrs.embeddings import embedder_for
from xcrs.ingest import embed_skills
from xcrs.repository import catalog_store, vectors
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


def cmd_resources_youtube_discover(args) -> None:
    """Find candidate playlists for the skills most roles rely on (ADR-0033); within the daily search quota."""
    from collections import Counter

    from xcrs.config import get_settings
    from xcrs.ingest.resources import youtube_discovery

    key = get_settings().youtube_api_key
    if not key:
        raise SystemExit("set XCRS_YOUTUBE_API_KEY in .env first (README: getting a YouTube API key)")
    cat = model.load_catalog()
    usage: Counter = Counter()
    for rid, role in cat.roles.items():
        usage.update({o for opts in validate.requirements(cat, model.RoleLevel(rid, role.levels[-1])) for o in opts})
    # Technical skills first (YouTube search is weak for practices such as mentoring), then by how many roles need them.
    skills = sorted(
        ((s.id, s.name) for s in cat.skills.values()),
        key=lambda s: (cat.skills[s[0]].kind == "practice", -usage[s[0]], s[1]),
    )
    config = yaml.safe_load((model.CATALOG_DIR / "sources" / "youtube.yaml").read_text()) or {}
    trusted = {c.lower() for c in config.get("trusted_channels") or []}
    blocked = {c.lower() for c in config.get("blocked_channels") or []}
    path = model.CATALOG_DIR / "sources" / "youtube-candidates.yaml"
    review = REPO_ROOT / "untracked" / "youtube-candidates.md"
    data = youtube_discovery.load_candidates(path)
    with httpx.Client(timeout=30) as client:
        found, used = youtube_discovery.discover(
            client, key, skills, set(data["searched"]), args.max_searches, trusted=trusted, blocked=blocked
        )
    review.parent.mkdir(exist_ok=True)
    youtube_discovery.save(path, review, data, found, {s.id: s.name for s in cat.skills.values()})
    left = len([s for s in skills if s[0] not in data["searched"]])
    print(f"{used} searches, {sum(map(len, found.values()))} candidates for {len(found)} skills; {left} skills left")
    print(f"review: {review}  (approve by moving ids to catalog/sources/youtube.yaml)")


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

    p = sub.add_parser("register-model", help="add or update an embedding model in the registry")
    p.add_argument("name")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--status", choices=["candidate", "active", "retired"], default="candidate")
    p.set_defaults(func=cmd_register_model)

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
    p = resources_sub.add_parser("youtube-discover", help="find candidate playlists per skill (needs the API key)")
    p.add_argument("--max-searches", type=int, default=90, help="search calls this run (100 quota units each)")
    p.set_defaults(func=cmd_resources_youtube_discover)
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
