import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def calculate_recommendations_Record(user_records: list[dict], nearby_courses: list[dict]):
    """
    사용자 기록과 주변 코스 목록(dict 리스트)을 받아 추천 코스 ID와 유사도를 반환합니다.
    """
    # 1. 입력 데이터를 DataFrame으로 변환
    user_runs = pd.DataFrame(user_records)
    courses_to_recommend = pd.DataFrame(nearby_courses)

    if courses_to_recommend.empty:
        return pd.DataFrame(columns=['course_id', 'similarity'])

    # 2. 사용자 프로필 생성
    difficulty_mapping = {'EASY': 1, 'MEDIUM': 2, 'HARD': 3}
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
    features = ['difficulty_numeric', 'latitude', 'longitude']
    if user_profile['distance'] > 0:
        features.insert(0, 'distance')
    
    user_profile_df = user_profile.to_frame().T
    
    combined_data = pd.concat(
        [courses_to_recommend[features], user_profile_df[features]], 
        ignore_index=True
    )
    
    # 데이터가 1개뿐이거나, 특정 피처의 모든 값이 동일하면 정규화 시 NaN이 발생하므로 예외 처리
    if len(combined_data) < 2 or any(combined_data[f].nunique() == 1 for f in features):
        # 이 경우 추천 로직을 수행할 수 없으므로 빈 DataFrame 반환
        return pd.DataFrame(columns=['course_id', 'similarity'])
    # ---------------------------
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(combined_data)
    
    scaled_courses = scaled_features[:-1]
    scaled_user_profile = scaled_features[-1].reshape(1, -1)
    
    similarity_scores = cosine_similarity(scaled_user_profile, scaled_courses)
    
    # 5. 최종 추천 목록 생성
    courses_to_recommend['similarity'] = similarity_scores[0]
    
    recommended_courses = courses_to_recommend[
        ~courses_to_recommend['course_id'].isin(user_runs['course_id'])
    ]
    
    final_recommendations = recommended_courses.sort_values(
        by='similarity', ascending=False
    ).head(3)
    
    return final_recommendations[['course_id', 'similarity']]

def calculate_recommendations_Setting(user_setting: dict, nearby_courses: list[dict]):
    """
    사용자 설정(거리, 난이도)과 주변 코스 목록을 받아 맞춤형 코스를 추천합니다.
    """
    # 1. 입력 데이터를 DataFrame으로 변환
    user_profile_df = pd.DataFrame([user_setting])
    courses_df = pd.DataFrame(nearby_courses)

    if courses_df.empty:
        return pd.DataFrame()

    # 2. 난이도를 숫자 데이터로 변환
    difficulty_mapping = {'EASY': 1, 'MEDIUM': 2, 'HARD': 3}
    user_profile_df['difficulty_numeric'] = user_profile_df['difficulty'].map(difficulty_mapping)
    courses_df['difficulty_numeric'] = courses_df['difficulty'].map(difficulty_mapping)

    # 3. 특성 선택
    if user_setting.get('distance', 0) > 0:
        features = ['distance', 'difficulty_numeric']
    else:
        features = ['difficulty_numeric']
        
    user_vector = user_profile_df[features]
    course_vectors = courses_df[features]

    # 4. 정규화
    combined_data = pd.concat([user_vector, course_vectors], ignore_index=True)
    
    if len(combined_data) < 2 or any(combined_data[f].nunique() == 1 for f in features):
        # 모든 코스의 난이도가 같을 경우, 가장 가까운 순서 등으로 대체 로직을 구현하거나
        # 여기서는 단순히 유사도 기반 추천이 불가능하므로 빈 DataFrame을 반환합니다.
        # (만약 features가 difficulty_numeric 하나인데 모든 코스가 보통이면 이 조건에 해당)
        return pd.DataFrame()
    # ---------------------------
        
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(combined_data)
    
    scaled_user_profile = scaled_features[0].reshape(1, -1)
    scaled_courses = scaled_features[1:]

    # 5. 코사인 유사도 계산
    similarity_scores = cosine_similarity(scaled_user_profile, scaled_courses)
    
    # 6. 최종 추천 목록 생성
    courses_df['similarity'] = similarity_scores[0]
    
    final_recommendations = courses_df.sort_values(
        by='similarity', ascending=False
    ).head(3)
    
    return final_recommendations