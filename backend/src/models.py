from pydantic import BaseModel
from typing import List, Dict

PALM_MODEL = "embedding-gecko-001"
VOYAGE_MODEL = "voyage-large-2"
OPENAI_MODEL = "text-embedding-3-small"
MISTRAL_MODEL = "mistral-embed"


# Model for course recommendation
class CourseRecommendation(BaseModel):
    course: str
    url: str
    explanation: str


# Model for role recommendation
class RoleRecommendation(BaseModel):
    role: str
    explanation: str
    courses: List[CourseRecommendation]


# Model for recommendation
class Recommendation(BaseModel):
    model: str
    roles: List[RoleRecommendation]


# Model for recommendation response
class RecommendationResponse(BaseModel):
    fileName: str
    recommendations: List[Recommendation]


# Model for recommendation request
class RecommendationRequest(BaseModel):
    took_and_liked: str
    took_and_neutral: str
    took_and_disliked: str
    curious: str


# Sample recommendation data
sample_rec_data = {
    "model": "mock-model",
    "roles": [
        {
            "role": "Role A",
            "explanation": "Explanation for Role A",
            "courses": [
                {"course": "Course 1", "url": "Url 1", "explanation": "Explanation for Course 1"},
                {"course": "Course 2", "url": "Url 2", "explanation": "Explanation for Course 2"},
                {"course": "Course 3", "url": "Url 3", "explanation": "Explanation for Course 3"},
            ],
        },
        {
            "role": "Role B",
            "explanation": "Explanation for Role B",
            "courses": [
                {"course": "Course 4", "url": "Url 4", "explanation": "Explanation for Course 4"},
                {"course": "Course 5", "url": "Url 5", "explanation": "Explanation for Course 5"},
                {"course": "Course 6", "url": "Url 6", "explanation": "Explanation for Course 6"},
            ],
        },
        {
            "role": "Role C",
            "explanation": "Explanation for Role C",
            "courses": [
                {"course": "Course 7", "url": "Url 7", "explanation": "Explanation for Course 7"},
                {"course": "Course 8", "url": "Url 8", "explanation": "Explanation for Course 8"},
                {"course": "Course 9", "url": "Url 9", "explanation": "Explanation for Course 9"},
            ],
        },
    ],
}
