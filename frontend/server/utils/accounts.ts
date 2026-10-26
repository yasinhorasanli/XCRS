// Accounts (ADR-0043). Sign-in happens here, in the Nuxt server, with nuxt-auth-utils; the session is its sealed
// cookie. The backend API learns who is signed in from a header this server signs (signedUser), and keeps the
// accounts themselves (POST /internal/sign-in, which the /api/v2 proxy never forwards).
import { createHmac } from 'node:crypto'
import type { EventHandler, H3Event } from 'h3'
import { safeNext } from '#shared/utils/next'

export type Provider = 'github' | 'google' | 'linkedin'
export const PROVIDERS: Provider[] = ['github', 'google', 'linkedin']

export interface SignedInIdentity {
  provider: Provider
  subject: string // the provider's stable user id
  name: string | null
  email: string | null
  emailVerified: boolean
}

const NEXT_COOKIE = 'xcrs-next'
const STATE_COOKIE = 'nuxt-auth-state' // nuxt-auth-utils' name
export const SESSION_COOKIE = 'nuxt-session' // nuxt-auth-utils' default name

// Read at runtime (like XCRS_API_URL), so one built image works in any environment.
export const apiUrl = (event: H3Event) => (process.env.XCRS_API_URL || useRuntimeConfig(event).apiUrl).replace(/\/$/, '')
export const internalSecret = () => process.env.XCRS_INTERNAL_SECRET || ''

/** Providers that can be used: accounts on (secret and session password set) and the provider's app registered. */
export function enabledProviders(event: H3Event): Provider[] {
  if (!internalSecret() || !process.env.NUXT_SESSION_PASSWORD) return []
  const oauth = useRuntimeConfig(event).oauth as Record<string, { clientId?: string; clientSecret?: string }>
  return PROVIDERS.filter((p) => oauth[p]?.clientId && oauth[p]?.clientSecret)
}

/** The provider's redirect URL. Behind Caddy and Tailscale Funnel this server sees plain HTTP on an internal host,
 * so the public address comes from XCRS_PUBLIC_URL; without it (local dev) nuxt-auth-utils uses the request URL. */
export const redirectUrl = (provider: Provider) =>
  process.env.XCRS_PUBLIC_URL ? `${process.env.XCRS_PUBLIC_URL.replace(/\/$/, '')}/auth/${provider}` : undefined

/** `X-XCRS-User: <user id>.<signed in at>.<issued at>.<HMAC-SHA256>`, checked by xcrs/api/identity.py. */
export function signedUser(secret: string, userId: string, signedInAtMs: number, issuedAtMs = Date.now()) {
  const payload = `${userId}.${signedInAtMs}.${issuedAtMs}`
  return `${payload}.${createHmac('sha256', secret).update(payload).digest('hex')}`
}

/** A provider's sign-in route: the redirect to the provider, then its callback. A callback must carry the state this
 * browser was given (its cookie): nuxt-auth-utils 0.5.30 let one through when both were missing, so a link with
 * someone else's code signed the visitor in to that account (security audit 2026-10-26; fixed upstream in 0.5.32,
 * checked here too). What fails after that without reaching onError (the provider or the API unreachable, a used
 * or expired code, which Google answers with 400) goes back to the sign-in page too, not to an error page. So does a
 * callback with `error` (the user pressed Cancel): Google's and LinkedIn's handlers ignore it and start over. */
export function signInRoute(provider: Provider, oauth: EventHandler) {
  return defineEventHandler(async (event) => {
    const { code, state, error } = getQuery(event)
    if (!code && error) return signInFailed(event, provider, new Error(`the provider answered ${String(error).slice(0, 100)}`))
    if (code &&!(typeof state === 'string' && state && getCookie(event, STATE_COOKIE) === state)) {
      deleteCookie(event, STATE_COOKIE, { path: '/' })
      return signInFailed(event, provider, new Error('the state is missing or does not match'))
    }
    rememberNext(event)
    try {
      return await oauth(event)
    } catch (error) {
      return signInFailed(event, provider, error)
    }
  })
}

/** Before the redirect to the provider: remember where to come back to (the OAuth round trip drops the query). */
function rememberNext(event: H3Event) {
  const query = getQuery(event)
  if (query.code || query.error) return
  const next = safeNext(query.next)
  if (next) setCookie(event, NEXT_COOKIE, next, { httpOnly: true, sameSite: 'lax', secure: !import.meta.dev, maxAge: 600, path: '/' })
  else deleteCookie(event, NEXT_COOKIE, { path: '/' })
}

function takeNext(event: H3Event) {
  const next = safeNext(getCookie(event, NEXT_COOKIE))
  deleteCookie(event, NEXT_COOKIE, { path: '/' })
  return next
}

/** After the provider said who this is: the backend finds or creates the account (linking by verified email),
 * then the session cookie is replaced and the browser goes back to where it was. */
export async function completeSignIn(event: H3Event, identity: SignedInIdentity, next?: string) {
  const secret = internalSecret()
  if (!secret) throw createError({ statusCode: 404, statusMessage: 'Accounts are not enabled' })
  const account = await $fetch<{ user_id: string; display_name: string | null; signed_in_at_ms: number }>(
    `${apiUrl(event)}/internal/sign-in`,
    {
      method: 'POST',
      headers: { 'X-XCRS-Internal-Secret': secret },
      body: {
        provider: identity.provider,
        subject: identity.subject,
        name: identity.name,
        email: identity.email,
        email_verified: identity.emailVerified,
      },
    },
  )
  await replaceUserSession(event, {
    user: { id: account.user_id, name: account.display_name },
    signedInAtMs: account.signed_in_at_ms,
    loggedInAt: Date.now(),
  })
  return sendRedirect(event, safeNext(next) ?? takeNext(event) ?? '/account')
}

export function signInFailed(event: H3Event, provider: Provider, error: unknown) {
  console.error(`sign-in with ${provider} failed:`, error instanceof Error ? error.message : error)
  deleteCookie(event, NEXT_COOKIE, { path: '/' })
  return sendRedirect(event, `/sign-in?error=${provider}`)
}
