<template>
  <header class="sticky top-0 z-30 border-b border-slate-200 bg-white/85 backdrop-blur">
    <div class="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
      <NuxtLink to="/" class="flex items-center gap-2.5" aria-label="XCRS home">
        <span class="grid h-8 w-8 place-items-center rounded-lg bg-indigo-600 text-sm font-bold text-white shadow-xs">X</span>
        <span class="font-semibold tracking-tight">XCRS</span>
        <span class="hidden text-sm text-slate-400 sm:inline">Explainable course recommendations</span>
      </NuxtLink>
      <UButton
        to="https://github.com/yasinhorasanli/XCRS"
        target="_blank"
        color="neutral"
        variant="ghost"
        icon="i-heroicons-code-bracket"
        label="Source"
        size="sm"
      />
    </div>
  </header>
</template>
