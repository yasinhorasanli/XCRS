import { RecommendationResult } from "~/models/result";

// Temporary adapter (ADR-0016): the current UI sends four comma-separated strings;
// the new backend takes structured lists at /api/v1/recommendations.
const apiUrl = process.env.XCRS_API_URL ?? "http://localhost:8000";

type Category = "took_and_liked" | "took_and_neutral" | "took_and_disliked" | "curious";

interface CourseV1 { title: string; url: string; explanation: string | null; concepts: string[] }
interface RoleV1 { role: string; score: number; explanation: string | null; explanation_status: string; courses: CourseV1[] }
interface ResponseV1 { request_id: string; status: string; model: string; roles: RoleV1[] }

const split = (text: string | undefined) =>
    (text ?? "").split(",").map((s) => s.trim()).filter((s) => s.length > 0);

export default defineEventHandler(async (event) => {
    const body = await readBody<Record<Category, string>>(event);

    let response = await $fetch<ResponseV1>(`${apiUrl}/api/v1/recommendations`, {
        method: "POST",
        body: {
            liked: split(body.took_and_liked),
            neutral: split(body.took_and_neutral),
            disliked: split(body.took_and_disliked),
            curious: split(body.curious),
        },
    });

    // Explanations arrive in the background (ADR-0018). This old page can't show them later,
    // so wait here (up to 5 minutes) until no role is pending.
    for (let i = 0; i < 150 && response.roles.some((r) => r.explanation_status === "pending"); i++) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        response = await $fetch<ResponseV1>(`${apiUrl}/api/v1/recommendations/${response.request_id}`);
    }

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
