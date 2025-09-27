import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def calculate_recommendations(user_records: list[dict], nearby_courses: list[dict]):
    """
    사용자 기록과 주변 코스 목록(dict 리스트)을 받아 추천 코스 ID와 유사도를 반환합니다.
    """
    # 1. 입력 데이터를 DataFrame으로 변환
    user_runs = pd.DataFrame(user_records)
    courses_to_recommend = pd.DataFrame(nearby_courses)

    # 추천할 코스가 없으면 빈 결과를 반환
    if courses_to_recommend.empty:
        return pd.DataFrame(columns=['course_id', 'similarity'])

    # 2. 사용자 프로필 생성 (user_runs 데이터만 사용)
    difficulty_mapping = {'쉬움': 1, '보통': 2, '어려움': 3}
    user_runs['difficulty_numeric'] = user_runs['difficulty'].map(difficulty_mapping)

    mode_result = user_runs['difficulty_numeric'].mode()
    # 최빈값이 없는 경우를 대비하여 기본값(1: 쉬움) 설정
    user_difficulty = mode_result[0] if not mode_result.empty else 1
    
    user_profile = pd.Series({
        'distance': user_runs['ran_distance'].mean(),
        'difficulty_numeric': user_difficulty,
        'latitude': user_runs['latitude'].mean(),
        'longitude': user_runs['longitude'].mean()
    })

    # 3. 추천 대상 코스 데이터 준비
    courses_to_recommend['difficulty_numeric'] = courses_to_recommend['difficulty'].map(difficulty_mapping)

    # 4. 유사도 계산을 위한 벡터화 및 정규화
    features = ['distance', 'difficulty_numeric', 'latitude', 'longitude']
    
    user_profile_df = user_profile.to_frame().T
    
    # 정규화를 위해 추천 대상 코스와 사용자 프로필을 하나로 합침
    combined_data = pd.concat([courses_to_recommend[features], user_profile_df], ignore_index=True)
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(combined_data)
    
    # 다시 분리
    scaled_courses = scaled_features[:-1]
    scaled_user_profile = scaled_features[-1].reshape(1, -1)
    
    # 코사인 유사도 계산
    similarity_scores = cosine_similarity(scaled_user_profile, scaled_courses)
    
    # 5. 최종 추천 목록 생성
    courses_to_recommend['similarity'] = similarity_scores[0]
    
    # 사용자가 이미 달려본 코스는 추천 목록에서 제외
    recommended_courses = courses_to_recommend[~courses_to_recommend['course_id'].isin(user_runs['course_id'])]
    
    # 유사도 점수가 높은 순으로 정렬하여 상위 3개 선택
    final_recommendations = recommended_courses.sort_values(
        by='similarity', ascending=False
    ).head(3)
    
    return final_recommendations[['course_id', 'similarity']]