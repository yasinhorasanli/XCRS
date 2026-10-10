# ADR-0043: Accounts with GitHub, Google and LinkedIn sign-in, handled in the Nuxt server; anonymous use stays

- **Status:** Accepted
- **Date:** 2026-10-03
- **Decider:** Muhammed Yasin Horasanli

## Context

The next three features build on each other: accounts and sign-in, then a CV/LinkedIn import that pre-fills the board, then progress tracking ("I took this course"). All three need to know who the learner is. Until now XCRS was anonymous: a result is a row reached by an unguessable link, and the board lives only in the browser tab (it is gone after a reload).

Constraints:

- Zero budget, self-hosted first (ADR-0002). VM-A has 8 GB for Caddy, web, API, PostgreSQL and the embedding model (ADR-0034).
- The browser talks only to the Nuxt server, which proxies `/api/v2/**` to FastAPI. The API is never exposed (ADR-0034).
- Anonymous use must keep working: signing in is for keeping results, not for using the site.
- KVKK/GDPR: store as little as possible, say what is stored, let people export and delete it, no tracking.
- Providers wanted by the decider: LinkedIn, Google, GitHub. Their sign-in returns name, email (with a verified flag) and a stable id. LinkedIn's sign-in does not return positions or skills; that needs a LinkedIn partner program (the CV import will use an uploaded PDF instead).

## Options considered

### Where sign-in lives
- **A. nuxt-auth-utils in the Nuxt server** (chosen). ✅ One small module (0.5.30) with GitHub, Google and LinkedIn handlers; the session is a sealed (encrypted and signed) cookie, so no session table; passkeys can be added later with the same module. Fits the existing proxy: the Nuxt server becomes the backend-for-frontend that tells FastAPI who the user is. ❌ Sign-in code is TypeScript; FastAPI must trust the proxy, so the proxy strips client-sent identity headers and signs its own.
- **B. FastAPI with Authlib.** ✅ Sign-in next to the data, in Python. ❌ More hand-written security code (state, cookies), and the redirects and cookies still pass through the Nuxt proxy.
- **C. A self-hosted identity server (Keycloak, Authentik, Zitadel).** ✅ Standard OIDC, a strong CV keyword. ❌ 0.5–1.5 GB of RAM plus its own database on VM-A, its own upgrades and backups; too much for three social logins.
- **D. Hosted (Clerk, Auth0).** ✅ Quickest. ❌ Users' data at a US vendor (a cross-border transfer to explain), lock-in, against self-hosted first.

### Sessions
- **Sealed cookie, 30 days, plus `users.sessions_valid_after`** (chosen). ✅ Stateless; "sign out everywhere" and account deletion still cut off cookies already issued, because the API compares the cookie's sign-in time with that column. ❌ A plain sign-out deletes the cookie in that browser only; a copied cookie stays valid until "sign out everywhere" or 30 days.
- **A sessions table.** ✅ Each session can be revoked. ❌ One more table and a write per sign-in, for no need we have today.

### One person, several providers
- **Link automatically on the same verified email** (chosen). ✅ No duplicate accounts; nothing to explain. ❌ Trusts the three providers to verify emails honestly; an unverified email never links (it isn't even stored).
- **Separate accounts, manual linking in settings.** ✅ No trust in other providers' checks. ❌ Duplicate accounts, and a settings flow to build.

### Deleting an account
- **Delete everything linked** (chosen): the user, their identities, their board, their results and the feedback and explanations on those results. ✅ The simplest promise to keep and to explain. ❌ Evaluation data from those results is lost.
- **Keep results with the owner removed.** ✅ Keeps evaluation data. ❌ A longer privacy notice, and a "deleted" that isn't quite.

## Decision

1. **Sign-in** with GitHub, Google and LinkedIn through nuxt-auth-utils in the Nuxt server (`/auth/github`, `/auth/google`, `/auth/linkedin`). A provider appears only when its client id and secret are set. Redirect URLs come from `XCRS_PUBLIC_URL` (behind Caddy and Funnel the Nuxt server can't see the public scheme).
2. **The session** is nuxt-auth-utils' sealed cookie (`nuxt-session`, HttpOnly, Secure, SameSite=Lax), 30 days. It holds the user id, display name and sign-in time; no provider tokens are kept.
3. **The proxy signs the identity.** It removes any `X-XCRS-*` header the browser sent and, for a signed-in user, adds `X-XCRS-User: <user id>.<signed in at>.<now>.<HMAC-SHA256>` with a secret shared by web and API (`XCRS_INTERNAL_SECRET`). The API accepts it only with a valid signature, at most 5 minutes old, from a user who still exists and whose sessions were not revoked after that sign-in; otherwise it answers 401 with `X-XCRS-Session: revoked`, and the proxy clears the cookie. The sign-in call itself (`POST /internal/sign-in`) is outside `/api/v2`, so the proxy never forwards it, and it needs the same secret.
4. **What an account stores** (migration 0012): `users` (id, display name, verified email, created, last sign-in, `sessions_valid_after`), `user_identities` (provider + the provider's user id, unique), `boards` (one current board per user for now: the chips and experience), and `recommendations_v2.user_id` (null for anonymous results). No profile pictures, no provider tokens.
5. **Keeping results:**
   - a signed-in user's new results are theirs, and the board they came from becomes their saved board;
   - "Save to my account" on an anonymous result signs in and attaches it, if nobody owns it yet and it is at most a day old;
   - the board page reloads the saved board when it opens empty.
   - Results stay viewable by anyone with the link, as before.
   - *Note 2026-10-05:* the saved board now follows every change to the board, on any page (skills added from a result included), about a second later and when the page is left; before, only a run or a change on the board page saved it.
6. **Privacy:** a plain-language notice at `/privacy`; JSON export (`GET /api/v2/me/export`); "sign out everywhere"; delete my account (everything linked). No analytics; the only cookies are the session and the short-lived ones the sign-in itself needs.
7. **Without the secrets** (`XCRS_INTERNAL_SECRET`, `NUXT_SESSION_PASSWORD`, a provider's id and secret) the site runs as before, anonymous only.

## Trade-offs accepted

- **The API trusts the Nuxt server.** Anyone with `XCRS_INTERNAL_SECRET` can act as any user, so it lives only in the VM's `.env` and the signature is short-lived. Mitigation: the API is not reachable from outside (ADR-0034).
- **Claiming by link.** Someone who has a fresh anonymous result's link can save it to their account before its creator does, and deleting their account then deletes that result. The one-day window keeps this small; a claim token in the browser would close it if it ever matters.
- **Account linking trusts provider verification** of emails (see Options).
- **Google and the `ts.net` address.** Google may ask to verify ownership of the redirect domain for the consent screen; if it won't accept `xcrs-a.tail3afc6e.ts.net`, Google sign-in waits for the bought domain (ADR-0040). GitHub and LinkedIn accept any HTTPS redirect.
- **CSRF** relies on SameSite=Lax cookies (cross-site POST, PUT and DELETE don't carry the session) and JSON request bodies; there is no separate CSRF token.

## Revisit when

- An account needs more than these three providers, roles (admin), or organizations: consider a self-hosted identity server (option C).
- The API runs as a separate public service (mobile app, other clients): give it its own token validation (OIDC access tokens) instead of trusting a proxy header.
- Someone reports a result claimed by another person: add a claim token.
