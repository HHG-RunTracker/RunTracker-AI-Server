from fastapi import FastAPI
from app.routers import recommendations # 라우터 import

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="Running Course Recommendation API",
    description="RunTracker AI API"
)

# recommendations 라우터를 메인 앱에 포함
app.include_router(recommendations.router)

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "RunTracker AI Server"}