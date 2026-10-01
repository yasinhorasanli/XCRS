// Engine v2 (ADR-0029–0031): forwarded to the backend unchanged, like /api/v1/** (see server/api/v1).
export default defineEventHandler((event) => {
  const apiUrl = process.env.XCRS_API_URL || useRuntimeConfig(event).apiUrl
  return proxyRequest(event, `${apiUrl.replace(/\/$/, '')}${event.path}`)
})
