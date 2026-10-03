// Development only: sign in as a test user without a provider, to try the account pages locally
// (/auth/dev?name=Test). `import.meta.dev` is false in production builds, where this answers 404.
import { completeSignIn } from '../../utils/accounts'

export default defineEventHandler((event) => {
  if (!import.meta.dev) throw createError({ statusCode: 404, statusMessage: 'Not Found' })
  const query = getQuery(event)
  const name = String(query.name || 'Dev user').slice(0, 40)
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-')
  return completeSignIn(event, {
    provider: 'github',
    subject: `dev-${slug}`,
    name,
    email: `${slug}@dev.invalid`,
    emailVerified: true,
  }, String(query.next ?? ''))
})
