# 1. 공식 Python 베이스 이미지 선택 (slim 버전으로 가볍게)
FROM python:3.11-slim

# 2. 컨테이너 내의 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 파일 먼저 복사 (이 부분이 변경되지 않으면 Docker 캐시가 사용되어 빌드 속도 향상)
COPY requirements.txt .

# 4. 의존성 설치 (pip 캐시를 사용하지 않아 이미지 용량을 줄임)
RUN pip install --no-cache-dir -r requirements.txt

# 5. 애플리케이션 소스 코드 복사
# 로컬의 ./app 디렉토리를 컨테이너의 /app/app 디렉토리로 복사
COPY ./app ./app

# 6. 컨테이너가 시작될 때 실행할 명령어 정의
# --host 0.0.0.0는 컨테이너 외부에서의 접속을 허용하기 위해 필수
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]