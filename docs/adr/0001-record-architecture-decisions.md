# ADR-0001: Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-09-27
- **Decider:** Muhammed Yasin Horasanli

## Context

XCRS is moving from a research prototype to a production-ready, self-hosted system. The modernization involves many interdependent choices: data store, model hosting, backend structure, frontend, CI/CD and cloud. Without a written record:

- the reasons behind a choice get lost, and the same debate comes back later;
- nobody can tell whether a decision is still valid when its conditions change;
- the project can't be explained clearly to others, whether contributors, reviewers or interviewers.

## Options considered

### Option A — No formal record (commit messages and memory only)
- ✅ Zero overhead
- ❌ Reasoning and rejected alternatives aren't captured anywhere
- ❌ No way to spot when a decision's assumptions have expired

### Option B — One large design document
- ✅ A single place to read
- ❌ It gets rewritten over time, so the history of *why* is lost
- ❌ Hard to see which parts are decided and which are still open

### Option C — Architecture Decision Records (one short file per decision)
- ✅ Each decision stays immutable and dated; superseding keeps the history
- ✅ Forces the options, trade-offs and a revisit trigger to be written down
- ✅ Widely recognized format
- ❌ Small ongoing effort per decision

## Decision

Use **Architecture Decision Records** in `docs/adr/`, numbered sequentially and based on [`0000-template.md`](0000-template.md). A decision is only marked *Accepted* once it has actually been made. Changing a decision means writing a new ADR that supersedes the old one, never editing history.

## Trade-offs accepted

- A few minutes of writing per significant decision.
- Smaller, easily reversible choices (library versions, naming) are deliberately **not** recorded, to keep the log meaningful.

## Revisit when

The ADR log stops being read or updated, or decisions start happening outside it.
