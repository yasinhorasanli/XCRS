/** A same-site path to come back to after signing in (never another site), for the sign-in page and the server.
 * Read the way a browser will read it: browsers drop tabs and newlines from a URL, so "/<TAB>/evil.example" would
 * become "//evil.example", another site (security audit 2026-10-26). */
export function safeNext(value: unknown): string | undefined {
  if (typeof value !== 'string' || !value.startsWith('/') || /[\u0000-\u001f\u007f\\]/.test(value)) return undefined
  try {
    const url = new URL(value, 'https://x.invalid')
    return url.origin === 'https://x.invalid' ? `${url.pathname}${url.search}${url.hash}`.slice(0, 500) : undefined
  } catch {
    return undefined
  }
}
