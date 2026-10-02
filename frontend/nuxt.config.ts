// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2026-10-01',
  modules: ['@nuxt/ui'],
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
  },
})
