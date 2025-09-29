from fastapi import FastAPI
from app.routers import recommendations
from app.routers import animal_courses

app = FastAPI(
    title="RunTracker AI API",
    description="RunTracker FastAPI Server API"
)

# 추천 API
app.include_router(recommendations.router)
# 동물형 코스 생성 API
app.include_router(animal_courses.router)

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "RunTracker AI Server"}