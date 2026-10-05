/**
 * The signed-in user's saved board (ADR-0043). The board page reloads it when it opens empty (`useSavedBoard`);
 * every change to the board, on any page (the board, or skills added from a result), is saved about a second
 * later (`useBoardAutosave`, set up once in app.vue). Anonymous boards stay in the tab only, as before.
 */

const SAVE_DELAY_MS = 1000
let timer: ReturnType<typeof setTimeout> | undefined
let saveNow: (() => Promise<void>) | undefined
let saving: Promise<void> | undefined

export function useSavedBoard() {
  const { chips, experience, add } = useBoardV2()
  const { loggedIn } = useUserSession()
  const api = useXcrsApiV2()

  async function restore() {
    if (!loggedIn.value || chips.value.length) return
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

  return { restore }
}

/** Saves the board shortly after each change while someone is signed in. Call once, from app.vue. */
export function useBoardAutosave() {
  const { chips, experience, asInput } = useBoardV2()
  const { loggedIn } = useUserSession()
  const api = useXcrsApiV2()

  saveNow = async () => {
    clearTimeout(timer)
    timer = undefined
    saving = api
      .saveBoard(asInput(), experience.value)
      .then(() => undefined)
      .catch(() => undefined)
    await saving
  }

  watch(
    [chips, experience],
    () => {
      if (!loggedIn.value) return
      clearTimeout(timer)
      timer = setTimeout(() => saveNow?.(), SAVE_DELAY_MS)
    },
    { deep: true },
  )

  // Leaving the page (reload, closing the tab, typing another address) before the delay is over: send the save
  // anyway. A keepalive request outlives the page; the session cookie goes with it (same site).
  window.addEventListener('pagehide', () => {
    if (!timer) return
    clearTimeout(timer)
    timer = undefined
    const body = JSON.stringify({ chips: asInput(), ...(experience.value ? { experience: experience.value } : {}) })
    fetch('/api/v2/me/board', { method: 'PUT', keepalive: true, headers: { 'content-type': 'application/json' }, body }).catch(
      () => undefined,
    )
  })
}

/** Writes a change that is still waiting to be saved, so a page that reads the saved board sees it. */
export async function flushSavedBoard() {
  if (timer) await saveNow?.()
  else await saving
}
