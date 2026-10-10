# Architecture Decision Records

Significant design decisions for XCRS, one per file. The format is explained in [ADR-0001](0001-record-architecture-decisions.md); new records start from the [template](0000-template.md).

| # | Decision | Status | Date |
|---|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted | 2026-09-27 |
| [0002](0002-self-hosted-first-cloud-last.md) | Self-hosted first, cloud last | Accepted | 2026-09-27 |
| [0003](0003-postgresql-pgvector-primary-store.md) | PostgreSQL + pgvector as the primary data store | Accepted | 2026-09-27 |
| [0004](0004-mongodb-for-ingestion-layer.md) | MongoDB for the ingestion layer | Superseded by 0032 | 2026-09-27 |
| [0005](0005-local-embedding-models.md) | Self-hosted embedding models instead of hosted APIs | Accepted | 2026-09-27 |
| [0006](0006-own-embedding-interface.md) | Own embedding interface with an OpenAI-compatible adapter (LangChain deferred) | Accepted | 2026-09-27 |
| [0007](0007-ollama-qwen3-embedding.md) | Ollama as the model runtime, `qwen3-embedding:0.6b` as the embedding model | Accepted | 2026-09-28 |
| [0008](0008-embedding-tables-per-entity.md) | Store embeddings in per-entity tables keyed by model | Accepted | 2026-09-28 |
| [0009](0009-per-model-indexes-and-precomputed-matches.md) | Per-model vector indexes and precomputed concept → course matches | Accepted | 2026-09-28 |
| [0010](0010-exact-threshold-search-and-candidate-penalty.md) | Exact threshold search for user phrases; disliked penalty only on candidate courses | Accepted | 2026-09-28 |
| [0011](0011-alembic-schema-migrations.md) | Alembic for database schema migrations | Accepted | 2026-09-28 |
| [0012](0012-sqlalchemy-orm-with-raw-sql-repository.md) | SQLAlchemy 2.0 (ORM + raw SQL) behind a repository, on psycopg 3 | Accepted | 2026-09-28 |
| [0013](0013-user-activity-hybrid-then-normalized.md) | User activity storage: hybrid now, fully normalized once the input format settles | Accepted | 2026-09-28 |
| [0014](0014-local-first-then-split-by-role.md) | Run locally first; target deployment splits the two VMs by role | Accepted | 2026-09-28 |
| [0015](0015-sync-endpoints-async-ready.md) | Synchronous endpoints now, structured to switch to async later | Accepted | 2026-09-29 |
| [0016](0016-versioned-structured-recommendation-api.md) | Versioned recommendation API with structured input | Accepted | 2026-09-29 |
| [0017](0017-layered-backend-with-pure-domain.md) | Layered backend with a pure domain core | Accepted | 2026-09-29 |
| [0018](0018-decoupled-per-role-explanations.md) | Explanations generated after the response, one LLM call per role | Accepted | 2026-09-29 |
| [0019](0019-explanation-layer-on-langchain.md) | Explanation layer on LangChain, with a grounded contract; retriever over our own SQL | Accepted | 2026-09-30 |
| [0020](0020-explanation-model-per-hardware.md) | `qwen3.5:4b` for explanations on the CPU VM, `qwen3.5:9b` on GPUs; embeddings kept off the LLM's machine | Accepted | 2026-09-30 |
| [0021](0021-ci-on-github-actions.md) | Continuous integration on GitHub Actions; container images for backend and frontend | Accepted | 2026-09-30 |
| [0022](0022-threshold-fallback-for-unmatched-phrases.md) | Keep the 2.5σ threshold, with a fallback for phrases that match nothing | Accepted | 2026-09-30 |
| [0023](0023-skill-board-input-and-linked-results.md) | Skill-board input with suggestions; results as linked roles and courses; thumbs feedback | Accepted | 2026-09-30 |
| [0024](0024-frontend-nuxt-4-and-nuxt-ui-4.md) | Frontend on Nuxt 4 with Nuxt UI 4 and Tailwind CSS 4 | Accepted | 2026-10-01 |
| [0025](0025-skills-catalog-from-onet-esco-with-llm-learning-paths.md) | A shared skills catalog from O*NET and ESCO, with LLM-drafted learning paths reviewed by a human | Accepted | 2026-10-01 |
| [0026](0026-learning-resources-courses-videos-docs.md) | Learning resources: courses, YouTube and documentation in one model; free and paid; English first | Accepted | 2026-10-01 |
| [0027](0027-career-roles-levels-and-transitions.md) | Career roles as specializations on one level ladder, connected by transitions | Accepted (delegated, reviewed) | 2026-10-01 |
| [0028](0028-catalog-as-code-skills-graph-and-prerequisites.md) | Catalog as code: one skills graph, prerequisites as AND-of-OR with proficiency, reviewed as pull requests | Accepted (delegated, reviewed) | 2026-10-01 |
| [0029](0029-engine-v2-skills-input-proficiency-and-coverage-scoring.md) | Engine v2: catalog skills with free-text fallback, optional proficiency, coverage × interest scoring, matching thresholds set by measurement | Accepted | 2026-10-01 |
| [0030](0030-skill-matching-lookup-llm-pick-confirmed-by-similarity.md) | Skill matching: name lookup, then the LLM picks from the catalog, confirmed by embedding similarity | Accepted | 2026-10-02 |
| [0031](0031-engine-v2-role-scoring-calibrated-blend.md) | Engine v2 role scoring: a calibrated blend of interest and coverage, levels from each level's additions | Accepted (calibration delegated, reviewed) | 2026-10-02 |
| [0032](0032-raw-ingested-data-in-postgres-jsonb.md) | Raw ingested data in PostgreSQL (JSONB, an `ingest` schema), not MongoDB | Accepted | 2026-10-02 |
| [0033](0033-first-learning-resource-sources.md) | First learning-resource sources: curated list as code, freeCodeCamp's open curriculum, YouTube adapter off until a key | Accepted (sources delegated, reviewed) | 2026-10-02 |
| [0034](0034-deployment-ghcr-images-compose-per-vm-caddy.md) | Deployment: images published to GHCR, one Compose file per VM, Caddy in front, a pull-based deploy script | Accepted (delegated, reviewed) | 2026-10-02 |
| [0035](0035-abuse-protection-rate-limits-and-llm-caps.md) | Abuse protection: per-client rate limits on expensive endpoints, and a cap on new LLM matches per request | Accepted (delegated, reviewed) | 2026-10-02 |
| [0036](0036-backups-nightly-verified-copied-to-the-other-vm.md) | Backups: nightly verified dumps on VM-A, copied to VM-B, weekly restore test | Accepted (delegated, reviewed) | 2026-10-02 |
| [0037](0037-engine-v2-becomes-the-main-site-with-grounded-explanations.md) | Engine v2 becomes the main site, with grounded LLM explanations per role; the classic engine moves to /classic | Accepted | 2026-10-02 |
| [0038](0038-one-video-slot-in-each-roles-resources.md) | Each role's resources keep one slot for an on-topic video; the learner's languages rank first | Accepted | 2026-10-02 |
| [0039](0039-retire-the-classic-engine.md) | Retire the classic engine: code, API, tables and research data removed after an archive | Accepted | 2026-10-02 |
| [0040](0040-public-access-through-tailscale-funnel.md) | Public access through Tailscale Funnel now, a Cloudflare Tunnel once there is a domain; VMs managed over Tailscale | Accepted | 2026-10-02 |
| [0041](0041-level-from-evidence-and-experience-next-level-gaps.md) | Levels from per-level evidence and optional experience; gaps and resources for the next level only; even staff levels | Accepted | 2026-10-03 |
| [0042](0042-aws-demo-copy-on-one-ec2-instance.md) | AWS runs a disposable demo copy on one EC2 t4g.small with Docker Compose; both models come from VM-B over Tailscale | Accepted | 2026-10-03 |
| [0043](0043-accounts-with-social-sign-in-in-the-nuxt-server.md) | Accounts with GitHub, Google and LinkedIn sign-in in the Nuxt server (sealed-cookie sessions, signed identity header to the API); anonymous use stays; export and delete | Accepted | 2026-10-03 |
| [0044](0044-job-titles-per-role-and-one-for-the-learner.md) | Job titles per role (market titles from O*NET) and one built from the learner's strongest skills; each links to a job search | Accepted | 2026-10-05 |
| [0045](0045-skills-from-a-cv-or-linkedin-pdf.md) | Skills from a CV or LinkedIn "Save to PDF": a lookup scan plus one local-LLM extraction, reviewed by the learner; signed-in only; nothing from the file kept | Accepted | 2026-10-05 |
| [0046](0046-youtube-long-videos-sections-and-channel-discovery.md) | YouTube long videos approved like playlists; sections (chapters, a playlist's videos) tagged with skills so a gap opens at its part; duration and quality per resource; discovery from trusted channels' uploads | Accepted | 2026-10-07 |
| [0047](0047-cloudfront-in-front-of-the-aws-demo-copy.md) | CloudFront in front of the AWS demo copy: a VPC origin (no public path to the instance), pay-as-you-go free tier, only Nuxt build files cached | Accepted | 2026-10-10 |
| [0048](0048-budget-action-stops-the-demo-instance-at-20-dollars.md) | A budget action stops the AWS demo instance when the month's actual spend (credits excluded) reaches $20 | Accepted | 2026-10-10 |

## Upcoming decisions

- Progress tracking: "I took this course" → follow-up questions → updated board ratings; resource thumbs as a quality signal
- Re-run the explainer benchmark (now on v2 explanations) on the new LLM VM before launch: 4B or 9B (ADR-0020)
- (Later) Roadmap visualization
- LLM tracing: Langfuse (cloud, then self-hosted on the LLM VM) or LangSmith; see ADR-0019
- Chat/agent feature (LangGraph)
- The domain name (bought later; then a Cloudflare Tunnel, ADR-0040)
- First real deployment to the VMs (checklist in `deploy/README.md`); push-based CD once there are users (ADR-0034)
- AWS phase steps (ADR-0042): account safety, Terraform bootstrap, network + EC2, arm64 images, CloudFront, S3 backups, OIDC, cost write-up
