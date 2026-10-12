// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2026-10-01',
  modules: ['@nuxt/ui', 'nuxt-auth-utils'],
  devtools: { enabled: false },
  css: ['~/assets/css/main.css'],
  // Follow the device by default; remember explicit choices in localStorage, never a cookie.
  // Fonts load through @nuxt/fonts (Inter, main.css); 800 for the home headline.
  fonts: { defaults: { weights: [400, 500, 600, 700, 800] } },
  colorMode: { preference: 'system', fallback: 'light', storage: 'localStorage' },
  app: {
    head: {
      title: 'XCRS · Explainable course recommendations',
      htmlAttrs: { lang: 'en' },
      link: [
        { rel: 'icon', href: '/favicon.ico', sizes: '48x48' },
        { rel: 'icon', type: 'image/png', href: '/icon-192.png', sizes: '192x192' },
        { rel: 'apple-touch-icon', href: '/apple-touch-icon.png' },
      ],
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
