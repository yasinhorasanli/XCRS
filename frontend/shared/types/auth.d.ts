// The session cookie's content (nuxt-auth-utils, ADR-0043): no provider tokens, no email.
declare module '#auth-utils' {
  interface User {
    id: string
    name: string | null
  }
  interface UserSession {
    signedInAtMs?: number // the account's sign-in time; "sign out everywhere" ends sessions from before it
  }
}

export {}
