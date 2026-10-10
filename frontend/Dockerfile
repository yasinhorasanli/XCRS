# Frontend image: Nuxt server build (SSR + the server route that calls the API).
#   docker build -t xcrs-web frontend
# XCRS_API_URL points the server route at the backend (docker-compose.yml sets http://api:8000).

FROM node:24-slim AS build
RUN npm install -g pnpm@11.8.0
WORKDIR /app
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY patches ./patches
RUN --mount=type=cache,target=/root/.local/share/pnpm/store pnpm install --frozen-lockfile
COPY . .
RUN pnpm run build


FROM node:24-slim
WORKDIR /app
COPY --from=build /app/.output ./.output
ENV NODE_ENV=production HOST=0.0.0.0 PORT=3000
USER node
EXPOSE 3000
CMD ["node", ".output/server/index.mjs"]
