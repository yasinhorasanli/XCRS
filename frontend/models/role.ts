interface RoleRecommendation {
    role: string;
    score: number
    explanation: string;
    courses: CourseRecommendation[];
}