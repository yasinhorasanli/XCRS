<script setup lang="ts">
import { PROVIDER_META } from '~/composables/useXcrsApiV2'
import { safeNext } from '#shared/utils/next'

const route = useRoute()
const { providers, devSignIn, enabled, loggedIn } = useAccount()

const next = computed(() => safeNext(route.query.next) ?? '/account')
const href = (provider: string, extra = '') => `/auth/${provider}?next=${encodeURIComponent(next.value)}${extra}`
const failed = computed(() => {
  const p = route.query.error
  return typeof p === 'string' ? (PROVIDER_META[p as keyof typeof PROVIDER_META]?.label ?? 'the provider') : null
})

if (loggedIn.value) await navigateTo(next.value)

useHead({ title: 'Sign in · XCRS' })
</script>

<template>
  <div class="page-width pb-16 pt-8">
    <!-- Same width as every page (the header stays put); the form is a narrow column in the middle. -->
    <div class="mx-auto max-w-md">
      <h1 class="text-2xl font-bold tracking-tight">Sign in</h1>
      <p class="mt-2 text-toned hyphens-auto sm:text-justify">
        Save your board and results, and pick up where you left off. XCRS works without an account too.
      </p>

      <UAlert
        v-if="failed"
        class="mt-5"
        color="error"
        icon="i-heroicons-exclamation-triangle"
        :title="`Signing in with ${failed} didn't work`"
        description="Please try again, or use another account."
      />

      <UAlert
        v-if="!enabled"
        class="mt-6"
        color="neutral"
        variant="subtle"
        title="Sign-in isn't available here yet"
        description="This server has no sign-in configured; everything else works without an account."
      />

      <div v-else class="mt-6 grid gap-3">
        <UButton
          v-for="p in providers"
          :key="p"
          :to="href(p)"
          external
          size="lg"
          block
          color="neutral"
          variant="outline"
          :icon="PROVIDER_META[p].icon"
          :label="`Continue with ${PROVIDER_META[p].label}`"
        />
        <UButton
          v-if="devSignIn"
          :to="href('dev', '&name=Dev%20user')"
          external
          size="lg"
          block
          color="warning"
          variant="soft"
          icon="i-heroicons-wrench-screwdriver"
          label="Dev sign-in (local only)"
        />
      </div>

      <div class="mt-8 rounded-2xl bg-default p-4 text-sm text-toned shadow-xs ring-1 ring-default">
        <p class="font-medium text-default">What we keep</p>
        <ul class="mt-2 list-disc space-y-1 pl-5">
          <li>Your name, and your email if the provider has verified it</li>
          <li>Which sign-in accounts you used (the same verified email joins them into one)</li>
          <li>Your board and your results</li>
        </ul>
        <p class="mt-2 hyphens-auto sm:text-justify">
          No passwords, profile pictures or tracking. You can download or delete everything on your account page.
          <NuxtLink to="/privacy" class="text-indigo-600 dark:text-indigo-300 underline">Privacy notice</NuxtLink>
        </p>
      </div>
    </div>
  </div>
</template>
