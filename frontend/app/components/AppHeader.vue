<script setup lang="ts">
const route = useRoute()
const { enabled, loggedIn, user, signInPath, signOut } = useAccount()

const menu = computed(() => [
  [{ label: 'My account', icon: 'i-heroicons-user-circle', to: '/account' }],
  [{ label: 'Sign out', icon: 'i-heroicons-arrow-left-start-on-rectangle', onSelect: () => signOut() }],
])
</script>

<template>
  <header class="sticky top-0 z-30 border-b border-default bg-default/85 backdrop-blur">
    <div class="page-width flex h-14 items-center justify-between sm:h-16">
      <NuxtLink to="/" class="flex shrink-0 items-center" aria-label="XCRS home">
        <img src="/xcrs-logo.webp" alt="" width="766" height="240" class="h-7 w-auto sm:h-10">
      </NuxtLink>
      <div class="flex items-center gap-1">
      <UButton
        to="https://github.com/yasinhorasanli/XCRS"
        target="_blank"
        color="neutral"
        variant="ghost"
        icon="i-heroicons-code-bracket"
        label="Source"
        size="sm"
        class="hidden sm:inline-flex"
      />
      <!-- Icon only on phones, so the header fits next to a signed-in name at 320 px. -->
      <UColorModeSelect size="sm" class="h-8 w-16 shrink-0 sm:w-30" :ui="{ value: 'sr-only sm:not-sr-only', content: 'min-w-32' }" aria-label="Colour mode" :content="{ align: 'end' }" />
      <template v-if="enabled">
        <UDropdownMenu v-if="loggedIn" :items="menu" :content="{ align: 'end' }">
          <UButton color="neutral" variant="ghost" icon="i-heroicons-user-circle" :label="user?.name || 'Account'" :ui="{ label: 'max-w-20 truncate sm:max-w-48' }" size="sm" trailing-icon="i-heroicons-chevron-down-20-solid" />
        </UDropdownMenu>
        <UButton v-else :to="signInPath(route.fullPath)" color="neutral" variant="soft" icon="i-heroicons-arrow-right-end-on-rectangle" label="Sign in" size="sm" />
      </template>
      </div>
    </div>
  </header>
</template>
