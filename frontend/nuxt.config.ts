// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2026-10-01',
  modules: ['@nuxt/ui', 'nuxt-auth-utils'],
  devtools: { enabled: false },
  css: ['~/assets/css/main.css'],
  // Light theme only (the design has no dark variant yet); fonts load through @nuxt/fonts (Inter, main.css).
  ui: { colorMode: false },
  app: {
    head: {
      title: 'XCRS · Explainable course recommendations',
      htmlAttrs: { lang: 'en' },
      meta: [
        {
          name: 'description',
          content: 'Tell XCRS what you know and what you are curious about; get career roles and courses, with the reasons.',
        },
      ],
    },
  },
  runtimeConfig: {
    // Default target of the /api/v2/** proxy; XCRS_API_URL overrides it at runtime (server/api/v2/[...path].ts).
    apiUrl: 'http://localhost:8000',
    // Accounts (ADR-0043, nuxt-auth-utils): the sealed session cookie lasts 30 days; its password comes from
    // NUXT_SESSION_PASSWORD, the providers' apps from NUXT_OAUTH_{GITHUB,GOOGLE,LINKEDIN}_CLIENT_{ID,SECRET}.
    session: {
      maxAge: 60 * 60 * 24 * 30,
      cookie: { sameSite: 'lax', httpOnly: true, secure: true },
    },
  },
})
