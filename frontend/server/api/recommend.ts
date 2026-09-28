import { RecommendationResult } from "~/models/result";

// Temporary adapter (ADR-0016): the current UI sends four comma-separated strings;
// the new backend takes structured lists at /api/v1/recommendations.
const apiUrl = process.env.XCRS_API_URL ?? "http://localhost:8000";

type Category = "took_and_liked" | "took_and_neutral" | "took_and_disliked" | "curious";

interface CourseV1 { title: string; url: string; explanation: string | null; concepts: string[] }
interface RoleV1 { role: string; score: number; explanation: string | null; courses: CourseV1[] }
interface ResponseV1 { request_id: string; status: string; model: string; roles: RoleV1[] }

const split = (text: string | undefined) =>
    (text ?? "").split(",").map((s) => s.trim()).filter((s) => s.length > 0);

export default defineEventHandler(async (event) => {
    const body = await readBody<Record<Category, string>>(event);

    const response = await $fetch<ResponseV1>(`${apiUrl}/api/v1/recommendations`, {
        method: "POST",
        body: {
            liked: split(body.took_and_liked),
            neutral: split(body.took_and_neutral),
            disliked: split(body.took_and_disliked),
            curious: split(body.curious),
        },
    });

    return {
        fileName: response.request_id,
        recommendations: [{
            model: response.model,
            roles: response.roles.map((r) => ({
                role: r.role,
                explanation: r.explanation ?? "",
                courses: r.courses.map((c) => ({
                    course: c.title,
                    url: c.url,
                    explanation: c.explanation ?? `Covers: ${c.concepts.slice(0, 4).join(", ")}`,
                })),
            })),
        }],
    } as RecommendationResult;
});
