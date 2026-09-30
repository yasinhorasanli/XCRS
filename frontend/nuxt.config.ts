// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  modules: ['@nuxt/ui', '@nuxtjs/tailwindcss'],
  devtools: { enabled: false },
  colorMode: { preference: 'light', fallback: 'light' },
  css: ['~/assets/css/main.css'],
  app: {
    head: {
      title: 'XCRS · Explainable course recommendations',
      htmlAttrs: { lang: 'en' },
      meta: [
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        {
          name: 'description',
          content: 'Tell XCRS what you know and what you are curious about; get career roles and courses, with the reasons.',
        },
      ],
      link: [
        { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
        { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' },
        { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap' },
      ],
    },
  },
  runtimeConfig: {
    // Default target of the /api/v1/** proxy; XCRS_API_URL overrides it at runtime (server/api/v1/[...path].ts).
    apiUrl: 'http://localhost:8000',
  },
})
