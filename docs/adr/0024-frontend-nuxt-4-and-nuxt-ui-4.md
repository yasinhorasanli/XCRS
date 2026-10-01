# ADR-0024: Frontend on Nuxt 4 with Nuxt UI 4 and Tailwind CSS 4

- **Status:** Accepted
- **Date:** 2026-10-01
- **Decider:** Muhammed Yasin Horasanli

## Context

- The frontend ran on Nuxt 3, Nuxt UI 2 and Tailwind CSS 3. The UX was rebuilt in [ADR-0023](0023-skill-board-input-and-linked-results.md) on purpose without choosing a framework: the components are small and use no drag-and-drop or graph libraries.
- The first push with CI surfaced **30 open Dependabot alerts**, all in the frontend lockfile: 1 critical (`@nuxt/devtools`), 18 high (including `nuxt` itself, `tar`, `postcss`, `svgo`), the rest medium or low. Several sit in Nuxt 3 / Nuxt UI 2 dependency ranges that newer major versions leave behind.
- Nuxt UI 2 supports only Nuxt 3, so a Nuxt 4 move includes Nuxt UI 4, which requires Tailwind CSS 4.
- Goals: production-ready and CV value ([ADR-0001](0001-record-architecture-decisions.md) context).

## Options considered

### Option A — Rewrite in Next.js (React)
- ✅ The largest frontend ecosystem and job market
- ❌ A full rewrite of working, tested pages; the Nuxt server proxy, composables and components all change

### Option B — Upgrade to Nuxt 4 + Nuxt UI 4 + Tailwind 4
- ✅ Keeps the code, its structure and the tests of the flow; current, supported major versions
- ✅ Clears dependency ranges behind most alerts
- ❌ Breaking changes to absorb: the `app/` directory layout, Nuxt UI 4 component props (colors, sizes, toasts via `UApp`), Tailwind 4's CSS-first configuration and renamed utilities

### Option C — Stay on Nuxt 3 and patch with overrides
- ✅ Smallest change
- ❌ Pins old majors with overrides; the debt grows

## Decision

**Option B**, chosen by the decider. The UI moves to Nuxt 4's `app/` layout, Nuxt UI 4 (`UApp` for toasts, semantic colors `primary`/`neutral`/`error`/`warning`, Heroicons kept via `@iconify-json/heroicons`) and Tailwind 4 (configured in CSS: `@import "tailwindcss"; @import "@nuxt/ui";`, no `@nuxtjs/tailwindcss` module). After the upgrade, the remaining alerts are fixed by upgrading or, where needed, by pnpm overrides in `pnpm-workspace.yaml`.

## Trade-offs accepted

- A migration of every component's Nuxt UI props and Tailwind class names, verified by the type check, a production build and a browser run of the full flow.
- Vue rather than React on the CV.

## Revisit when

- A feature needs a library that exists only for React.
- Nuxt 5 is released.
