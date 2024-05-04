from pydantic import BaseModel
from typing import List, Dict

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
    recommendations: List[Recommendation]

# Model for recommendation request
class RecommendationRequest(BaseModel):
    took_and_liked: str
    took_and_neutral: str
    took_and_disliked: str
    curious: str