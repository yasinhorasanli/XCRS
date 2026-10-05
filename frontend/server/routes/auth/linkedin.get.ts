// Sign in with LinkedIn (ADR-0043): the "Sign In with LinkedIn using OpenID Connect" product returns only sub,
// name, email (with email_verified), picture and locale; positions and skills need a partner program.
import { completeSignIn, rememberNext, redirectUrl, signInFailed } from '../../utils/accounts'

const oauth = defineOAuthLinkedInEventHandler({
  config: { scope: ['openid', 'profile', 'email'], redirectURL: redirectUrl('linkedin') },
  onSuccess: (event, { user }) =>
    completeSignIn(event, {
      provider: 'linkedin',
      subject: String(user.sub),
      name: user.name ?? null,
      email: user.email ?? null,
      emailVerified: user.email_verified === true,
    }),
  onError: (event, error) => signInFailed(event, 'linkedin', error),
})

export default defineEventHandler((event) => {
  rememberNext(event)
  return oauth(event)
})
