import type { Provider } from '~/types/apiV2'

/**
 * Accounts (ADR-0043): the session (nuxt-auth-utils' useUserSession) plus which sign-in providers this server
 * offers. With none (accounts off), the site stays anonymous and shows no sign-in.
 */
export function useAccount() {
  const session = useUserSession()
  const api = useXcrsApiV2()
  const providers = useState<Provider[]>('account-providers', () => [])
  const devSignIn = useState<boolean>('account-dev-sign-in', () => false)
  const enabled = computed(() => providers.value.length > 0 || devSignIn.value)

  async function loadProviders() {
    try {
      const r = await api.providers()
      providers.value = r.providers
      devSignIn.value = r.dev
    } catch {
      providers.value = []
    }
  }

  /** The sign-in page, coming back to `next` (a path on this site) afterwards. */
  const signInPath = (next?: string) =>
    next && next !== '/' && !next.startsWith('/sign-in') ? `/sign-in?next=${encodeURIComponent(next)}` : '/sign-in'

  async function signOut() {
    await session.clear()
    await navigateTo('/')
  }

  return { ...session, providers, devSignIn, enabled, loadProviders, signInPath, signOut }
}

/** An API error's status and message (FastAPI's `detail`). */
export function apiError(e: unknown): { status?: number; message?: string } {
  const err = e as { statusCode?: number; data?: { detail?: unknown } }
  return { status: err?.statusCode, message: typeof err?.data?.detail === 'string' ? err.data.detail : undefined }
}
