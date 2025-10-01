import pandas as pd
import numpy as np  # numpy를 import 합니다.
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def calculate_recommendations(user_records: list[dict], nearby_courses: list[dict]):
    """
    사용자 기록과 주변 코스 목록(dict 리스트)을 받아 추천 코스 ID와 유사도를 반환합니다.
    """
    # 1. 입력 데이터를 DataFrame으로 변환
    user_runs = pd.DataFrame(user_records)
    courses_to_recommend = pd.DataFrame(nearby_courses)

    # [수정] 추천할 코스가 없거나 사용자 기록이 없으면 빈 결과를 반환
    if courses_to_recommend.empty or user_runs.empty:
        return pd.DataFrame(columns=['course_id', 'similarity'])

    # 2. 사용자 프로필 생성
    difficulty_mapping = {'쉬움': 1, '보통': 2, '어려움': 3}
    user_runs['difficulty_numeric'] = user_runs['difficulty'].map(difficulty_mapping)

    mode_result = user_runs['difficulty_numeric'].mode()
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
    
    combined_data = pd.concat([courses_to_recommend[features], user_profile_df], ignore_index=True)
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(combined_data)
    
    scaled_courses = scaled_features[:-1]
    scaled_user_profile = scaled_features[-1].reshape(1, -1)

    scaled_user_profile = np.nan_to_num(scaled_user_profile)
    scaled_courses = np.nan_to_num(scaled_courses)
    
    # 코사인 유사도 계산
    similarity_scores = cosine_similarity(scaled_user_profile, scaled_courses)
    
    # 5. 최종 추천 목록 생성
    courses_to_recommend['similarity'] = similarity_scores[0]
    
    recommended_courses = courses_to_recommend[~courses_to_recommend['course_id'].isin(user_runs['course_id'])]
    
    final_recommendations = recommended_courses.sort_values(
        by='similarity', ascending=False
    ).head(3)
    
    return final_recommendations[['course_id', 'similarity']]