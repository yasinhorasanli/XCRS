// Sign in with GitHub (ADR-0043). GitHub's public profile email has no verified flag, so the verified primary
// email is read from /user/emails (scope user:email).
import { completeSignIn, rememberNext, redirectUrl, signInFailed } from '../../utils/accounts'

interface GitHubEmail {
  email: string
  primary: boolean
  verified: boolean
}

const oauth = defineOAuthGitHubEventHandler({
  config: { scope: ['read:user', 'user:email'], redirectURL: redirectUrl('github') },
  async onSuccess(event, { user, tokens }) {
    const emails = await $fetch<GitHubEmail[]>('https://api.github.com/user/emails', {
      headers: { 'Authorization': `token ${tokens.access_token}`, 'User-Agent': 'XCRS' },
    }).catch(() => [])
    const primary = emails.find((e) => e.primary && e.verified)
    return completeSignIn(event, {
      provider: 'github',
      subject: String(user.id),
      name: user.name || user.login || null,
      email: primary?.email ?? null,
      emailVerified: Boolean(primary),
    })
  },
  onError: (event, error) => signInFailed(event, 'github', error),
})

export default defineEventHandler((event) => {
  rememberNext(event)
  return oauth(event)
})
