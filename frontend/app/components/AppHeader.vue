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
    <div class="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
      <NuxtLink to="/" class="flex items-center gap-2.5" aria-label="XCRS home">
        <span class="grid h-8 w-8 place-items-center rounded-lg bg-indigo-600 dark:bg-indigo-700 text-sm font-bold text-white shadow-xs">X</span>
        <span class="font-semibold tracking-tight">XCRS</span>
        <span class="hidden text-sm text-dimmed sm:inline">Explainable course recommendations</span>
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
      <UColorModeSelect size="sm" class="h-8 w-30 shrink-0" aria-label="Colour mode" :content="{ align: 'end' }" />
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
