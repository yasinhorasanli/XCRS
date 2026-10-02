// The browser talks only to this Nuxt server; it forwards /api/v2/** to the backend API unchanged
// (ADR-0029–0031 contract), so the backend never has to be exposed publicly or handle CORS.
// XCRS_API_URL is read here, at runtime, so one built image works in any environment.
export default defineEventHandler((event) => {
  const apiUrl = process.env.XCRS_API_URL || useRuntimeConfig(event).apiUrl
  return proxyRequest(event, `${apiUrl.replace(/\/$/, '')}${event.path}`)
})
