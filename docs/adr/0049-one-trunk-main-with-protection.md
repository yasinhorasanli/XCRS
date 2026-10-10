# ADR-0049: One trunk: `modernization` merges into `main`, which becomes the only long-lived branch and is protected

- **Status:** Accepted
- **Date:** 2026-10-10
- **Decider:** Muhammed Yasin Horasanli

## Context

- Since September 2026 all work happened on `modernization`; `main` still held the 2024 research prototype. `main` is the default branch, so the repository's front page, Dependabot (66 open alerts, almost all in the prototype's dependencies) and anyone following a link from a CV or a post saw the old system.
- `main` had no commits that `modernization` lacked, so the merge can't conflict.
- The research version is archived on Zenodo (DOI 10.5281/zenodo.14291086, tag `Zenodo-v1.0`); `main` had two later commits (a README edit and dependency bumps). The Zenodo GitHub integration was switched off by the decider, so a GitHub release no longer creates a Zenodo version.
- One developer, CI on every pull request (ADR-0021), and deployments pinned to commit SHAs (ADR-0034); two parallel Claude Code workstreams open pull requests.
- `main` had no protection.

## Options considered

### Keeping the research version findable
- **A: a tag `research-prototype` at `main`'s last prototype commit (chosen)**, beside `Zenodo-v1.0`. ✅ Immutable; ❌ none worth noting.
- **B: also a `research` branch.** ❌ Same snapshot, but a branch can move by accident.

### Moving `main` forward
- **A: a pull request `modernization` → `main` with a merge commit (chosen).** ✅ CI runs on the final state; one visible milestone; the description summarizes the modernization.
- **B: a fast-forward push.** ✅ Linear history; ❌ no record of the milestone.

### Branches afterwards
- **A: only `main` (chosen).** Every change is a pull request into `main`; `modernization` is frozen, then deleted. ✅ The usual flow for one developer with CI per pull request; deploys are pinned by SHA, so a separate "stable" branch adds no safety.
- **B: `modernization` for development, `main` as stable.** ❌ Every change merged twice and CI on both, ceremony suited to release trains.

### Protection
- **A: a repository ruleset on `main` (chosen):** changes only through pull requests whose CI jobs (`backend`, `frontend`, `images`) pass; no force-push; no deletion; the repository admin can bypass in an emergency. ✅ Free for public repositories; catches pushes to `main` by mistake.
- **B: none.**

## Decision

1. Tag `research-prototype` at `ee74d82`; the README links it and the DOI instead of "the `main` branch".
2. Merge `modernization` into `main` through a pull request with a merge commit.
3. `main` is the only long-lived branch. CI and the image release run on `main` (images tagged `main` and the commit SHA). The VM-A `.env` example and the AWS demo's `release` default point at `main`. Both workstreams branch from and target `main`.
4. A ruleset protects `main` as above.

## Trade-offs accepted

- The prototype's history stays in `main`'s history, under the merge. Readers find the prototype through the tag and the DOI, not a branch.
- The AWS demo's `release` default changes, so the next `terraform apply` in `infra/demo` replaces the instance (and updates the VPC origin).
- VM-A's checkout tracks `modernization` until it is switched to `main` at the next deploy; deleting `modernization` waits until then.
- Admin bypass means the rule can be overridden; it is meant for emergencies only.

## Revisit when

- A second person contributes: require reviews in the ruleset.
- Releases with versions (e.g. `v2.0` at launch) are wanted: add a release workflow or tags.
