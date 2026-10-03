// The browser talks only to this Nuxt server; it forwards /api/v2/** to the backend API unchanged
// (ADR-0029–0031 contract), so the backend never has to be exposed publicly or handle CORS.
// XCRS_API_URL is read here, at runtime, so one built image works in any environment.
//
// Accounts (ADR-0043): X-XCRS-* headers and cookies from the browser are dropped; for a signed-in user the
// proxy adds the signed X-XCRS-User header instead. When the API answers `X-XCRS-Session: revoked` (account
// deleted, signed out everywhere), the session cookie is cleared.
import { SESSION_COOKIE, apiUrl, internalSecret, signedUser } from '../../utils/accounts'

export default defineEventHandler(async (event) => {
  const secret = internalSecret()
  // Read before the cookie header is dropped, and only when there is a session cookie: getUserSession would
  // otherwise start an empty session and set a cookie for every anonymous visitor.
  const session = secret && getCookie(event, SESSION_COOKIE) ? await getUserSession(event) : undefined
  const incoming = event.node.req.headers
  for (const name of Object.keys(incoming)) {
    if (name.startsWith('x-xcrs-') || name === 'cookie') delete incoming[name]
  }
  const headers: Record<string, string> = {}
  if (secret && session?.user?.id && session.signedInAtMs) {
    headers['x-xcrs-user'] = signedUser(secret, session.user.id, session.signedInAtMs)
  }
  // fetch() resolves dot segments, %2e%2e included: never let a path step outside /api/v2 (e.g. to /internal).
  const target = new URL(`${apiUrl(event)}${event.path}`)
  if (!target.pathname.startsWith('/api/v2/')) throw createError({ statusCode: 404, statusMessage: 'Not Found' })
  return proxyRequest(event, target.href, {
    headers,
    async onResponse(event, response) {
      if (response.headers.get('x-xcrs-session') === 'revoked') {
        removeResponseHeader(event, 'x-xcrs-session')
        if (session?.user) await clearUserSession(event)
      }
    },
  })
})
