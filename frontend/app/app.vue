<script setup lang="ts">
const route = useRoute()
const { loadProviders } = useAccount()
await callOnce('account-providers', loadProviders)
// Signed in, the board is saved after every change, wherever it happens (board page, results page).
if (import.meta.client) useBoardAutosave()

// Link previews (LinkedIn, X, chat apps) read the server-rendered page and need absolute URLs. Behind Caddy this
// server sees an internal address, so the public one comes from XCRS_PUBLIC_URL (as for the OAuth redirects);
// without it (local dev) the request's own. Every page shares the one card: a result's roles stay out of previews.
const siteUrl = useState('site-url', () =>
  import.meta.server ? (process.env.XCRS_PUBLIC_URL || useRequestURL().origin).replace(/\/$/, '') : '',
)
useSeoMeta({
  ogType: 'website',
  ogSiteName: 'XCRS',
  ogTitle: 'XCRS · Which software career fits you?',
  ogDescription: 'Tell XCRS what you know and what you are curious about; get career roles and courses, with the reasons.',
  ogUrl: () => `${siteUrl.value}${route.path}`,
  ogImage: () => `${siteUrl.value}/social-preview.jpg`,
  ogImageWidth: 1280,
  ogImageHeight: 640,
  ogImageAlt: 'XCRS: find the software career that fits you, and what to learn next',
  twitterCard: 'summary_large_image',
})
// One address per page for search engines: `/privacy/` and `/?ref=…` are the same page as `/privacy` and `/`.
// Built from XCRS_PUBLIC_URL like the link previews, never from the request's Host header.
useHead({
  link: [{ rel: 'canonical', href: () => `${siteUrl.value}${route.path.replace(/(.)\/+$/, '$1')}` }],
})
</script>

<template>
  <UApp :toaster="{ position: 'top-center' }">
    <div class="flex min-h-screen flex-col">
      <AppHeader class="print:hidden" />
      <!-- Home only: the wordmark and headline scroll away while the bar above stays. -->
      <HomeHero v-if="route.path === '/'" class="print:hidden" />
      <main class="flex-1">
        <NuxtPage />
      </main>
      <!-- The O*NET attribution (CC BY 4.0) is given in full on the privacy page; the link here carries it. -->
      <footer class="page-width grid gap-1.5 py-6 text-center text-xs text-dimmed print:hidden">
        <!-- Where XCRS comes from (2026-10-12): the thesis, the ISD 2025 paper and the archived first version. -->
        <p class="text-balance">
          XCRS began as a master's thesis at Boğaziçi University and was published at ISD 2025
          (<a href="https://doi.org/10.62036/ISD.2025.133" class="underline hover:text-toned" target="_blank" rel="noopener">paper</a>).
          Today's app is a rebuild shaped by its user study and expert evaluation
          (<a href="https://doi.org/10.5281/zenodo.14291086" class="underline hover:text-toned" target="_blank" rel="noopener">replication package</a>).
        </p>
        <p>
          Data: <NuxtLink to="/privacy#data-sources" class="underline hover:text-toned">O*NET 31.0 (CC BY 4.0)</NuxtLink>
          ·
          <NuxtLink to="/privacy" class="underline hover:text-toned">Privacy</NuxtLink>
        </p>
      </footer>
      <!-- On paper (the results PDF) links don't work: the attribution in full. -->
      <p class="hidden px-4 pt-6 text-center text-[10px] text-dimmed print:block">
        Includes information from the O*NET 31.0 Database by USDOL/ETA, used under CC BY 4.0, modified by XCRS; USDOL/ETA
        has not approved or endorsed it.
      </p>
    </div>
  </UApp>
</template>
