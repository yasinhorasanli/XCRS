"""XCRS admin commands.

    uv run xcrs import-prototype
    uv run xcrs register-model qwen3-embedding:0.6b --id 1 --status active
    uv run xcrs embed-catalog qwen3-embedding:0.6b
    uv run xcrs search "Docker"
"""

import argparse
import json

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from xcrs.db.models import EmbeddingModel, RoadmapNode, Role
from xcrs.db.session import new_session
from xcrs.embeddings import embedder_for
from xcrs.ingest import embed_catalog, import_prototype
from xcrs.repository import vectors

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


def cmd_import_prototype(args) -> None:
    with new_session() as session:
        print(json.dumps(import_prototype.run(session), indent=2))


def cmd_register_model(args) -> None:
    preset = MODEL_PRESETS.get(args.name)
    if preset is None:
        raise SystemExit(f"no preset for {args.name!r}; add one to MODEL_PRESETS")
    values = {"id": args.id, "name": args.name, "status": args.status, **preset}
    with new_session() as session:
        stmt = insert(EmbeddingModel).values(values)
        session.execute(stmt.on_conflict_do_update(index_elements=["id"], set_={k: stmt.excluded[k] for k in values if k != "id"}))
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
        print(f"threshold {threshold:.3f} (mean {model.sim_mean:.3f} + {args.sigma}σ {model.sim_std:.3f}); {len(matches)} matches")
        names = dict(
            session.execute(
                select(RoadmapNode.id, Role.name + " › " + RoadmapNode.name).join(Role, Role.id == RoadmapNode.role_id)
            ).all()
        )
        for m in matches[: args.limit]:
            print(f"  {m.similarity:.3f}  {names[m.concept_id]}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="xcrs")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("import-prototype", help="import the prototype's CSV data")
    p.set_defaults(func=cmd_import_prototype)

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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
