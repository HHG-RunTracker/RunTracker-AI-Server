from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from app.services.recommendation_service import calculate_recommendations

router = APIRouter(
    prefix="/recommend",
    tags=["Recommendations"]
)

# 사용자 기록
class UserRecord(BaseModel):
    course_id: int
    ran_distance: float
    difficulty: str
    latitude: float
    longitude: float

# 주변 반경 코스
class Course(BaseModel):
    course_id: int
    distance: float
    difficulty: str
    latitude: float
    longitude: float

# Request
class RecommendationRequest(BaseModel):
    user_records: List[UserRecord]
    nearby_courses: List[Course]

@router.post("/")
def get_recommendations(request: RecommendationRequest):
    """
    사용자 기록과 주변 코스 목록을 받아 맞춤형 코스를 추천합니다.
    """
    # Pydantic 모델을 서비스 계층에 전달하기 위해 다시 dict 리스트로 변환
    user_records_dict = [record.model_dump() for record in request.user_records]
    nearby_courses_dict = [course.model_dump() for course in request.nearby_courses]
    
    # 두 목록 중 하나라도 비어있으면 계산 불가
    if not user_records_dict or not nearby_courses_dict:
        raise HTTPException(status_code=400, detail="유저 기록과 주변 코스 목록이 모두 필요합니다.")

    recommendations = calculate_recommendations(user_records_dict, nearby_courses_dict)
    
    # 추천할 코스가 없으면 빈 리스트 반환
    if recommendations is None or recommendations.empty:
        return []

    return recommendations.to_dict('records')