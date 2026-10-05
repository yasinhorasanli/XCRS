// Sign in with Google (ADR-0043): OpenID Connect userinfo (sub, name, email, email_verified).
import { completeSignIn, rememberNext, redirectUrl, signInFailed } from '../../utils/accounts'

const oauth = defineOAuthGoogleEventHandler({
  config: { scope: ['openid', 'email', 'profile'], redirectURL: redirectUrl('google') },
  onSuccess: (event, { user }) =>
    completeSignIn(event, {
      provider: 'google',
      subject: String(user.sub),
      name: user.name ?? null,
      email: user.email ?? null,
      emailVerified: user.email_verified === true,
    }),
  onError: (event, error) => signInFailed(event, 'google', error),
})

export default defineEventHandler((event) => {
  rememberNext(event)
  return oauth(event)
})
