from fastapi import APIRouter
from typing import List
from pydantic import BaseModel

from app.services import animal_finder

router = APIRouter()

class Location(BaseModel):
    latitude: float
    longitude: float

# Response
class AnimalCourseResult(BaseModel):
    animal_name: str
    score: float
    path: List[Location]

@router.get(
    "/animal-courses",
    response_model=List[AnimalCourseResult],
    tags=["Courses"]
)

def find_animal_courses_endpoint(
    latitude: float, 
    longitude: float, 
    radius: int = 1000, 
    threshold: float = 1.5
):
    """
    주어진 좌표 주변에서 동물 모양과 유사한 코스를 찾아 반환합니다.
    """
    found_courses = animal_finder.find_animal_courses(latitude, longitude, radius, threshold)
    return found_courses