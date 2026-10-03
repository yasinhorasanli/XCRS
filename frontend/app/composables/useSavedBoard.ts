/**
 * The signed-in user's saved board (ADR-0043): reloaded when the board page opens empty, and saved shortly after
 * each change. Anonymous boards stay in the tab only, as before.
 */
export function useSavedBoard() {
  const { chips, experience, add, asInput } = useBoardV2()
  const { loggedIn } = useUserSession()
  const api = useXcrsApiV2()
  const ready = ref(false) // don't save before the saved board was read, or an empty tab would overwrite it
  let timer: ReturnType<typeof setTimeout> | undefined

  async function restore() {
    if (loggedIn.value && !chips.value.length) {
      try {
        const { board } = await api.board()
        // Typed chips are matched in one request (cached phrases answer at once), not one request per chip,
        // which a long board would run into the rate limit with (ADR-0035).
        const phrases = board?.chips.flatMap((c) => (c.skill || !c.text ? [] : [c.text])) ?? []
        const matches = phrases.length ? (await api.match(phrases.slice(0, 30)).catch(() => null))?.matches : undefined
        if (board && !chips.value.length) {
          for (const c of board.chips) {
            const label = c.name ?? c.text ?? c.skill ?? ''
            const matched = c.skill ? undefined : matches?.find((m) => m.phrase === c.text)
            add(label, c.category, c.skill ? { id: c.skill, name: label } : undefined, c.proficiency ?? undefined, matched)
          }
          experience.value = board.experience ?? null
        }
      } catch {
        // the board just starts empty
      }
    }
    ready.value = true
  }

  watch(
    [chips, experience],
    () => {
      if (!ready.value || !loggedIn.value) return
      clearTimeout(timer)
      timer = setTimeout(() => api.saveBoard(asInput(), experience.value).catch(() => {}), 1500)
    },
    { deep: true },
  )
  onBeforeUnmount(() => clearTimeout(timer))

  return { restore }
}
