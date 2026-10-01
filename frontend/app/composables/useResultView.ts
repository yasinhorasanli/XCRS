import type { ExplanationStatus, Role } from '~/types/api'

/** A course as the results page shows it: once, linked to every role that recommended it. */
export interface CourseLink {
  roleIndex: number
  roleId: number
  role: string
  explanation: string | null
  status: ExplanationStatus
  concepts: string[]
}

export interface CourseGroup {
  course_id: number
  title: string
  url: string
  links: CourseLink[]
}

/** De-duplicate courses across roles, ordered by the first role that recommends them (fewer line crossings). */
export function groupCourses(roles: Role[]): CourseGroup[] {
  const groups = new Map<number, CourseGroup>()
  roles.forEach((role, roleIndex) => {
    for (const c of role.courses) {
      const group = groups.get(c.course_id) ?? { course_id: c.course_id, title: c.title, url: c.url, links: [] }
      group.links.push({
        roleIndex,
        roleId: role.role_id,
        role: role.role,
        explanation: c.explanation,
        status: role.explanation_status,
        concepts: c.concepts,
      })
      groups.set(c.course_id, group)
    }
  })
  return [...groups.values()]
}
