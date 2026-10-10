// Which sign-in buttons to show (ADR-0043): none when accounts are off. `dev`: the local test sign-in
// (server/routes/auth/dev.get.ts), only in `nuxt dev` with accounts configured.
import { enabledProviders, internalSecret } from '../../utils/accounts'

export default defineEventHandler((event) => ({
  providers: enabledProviders(event),
  dev: import.meta.dev && Boolean(internalSecret() && process.env.NUXT_SESSION_PASSWORD),
}))
