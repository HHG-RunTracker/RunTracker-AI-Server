import os
import json
import cv2
import numpy as np
import osmnx as ox

# --- 서버 시작 시 1회만 실행: 동물 벡터 미리 로드 ---
ANIMAL_VECTORS = {}
vectors_folder = 'animal_vectors'
if os.path.isdir(vectors_folder):
    for filename in os.listdir(vectors_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(vectors_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                animal_name = data['animal_name']
                ANIMAL_VECTORS[animal_name] = np.array(data['vector'])
print(f"✅ 총 {len(ANIMAL_VECTORS)}개의 동물 벡터를 로드했습니다.")
# ----------------------------------------------

def normalize_path(path_vector, size=100):
    """
    경로 벡터를 특정 크기(size x size)의 상자 안에 맞도록 정규화합니다.
    """
    path_array = np.array(path_vector, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(path_array.reshape(-1, 1, 2))
    if w == 0 or h == 0:
        return path_array.astype(np.int32)
    path_array[:, 0] -= x
    path_array[:, 1] -= y
    scale = size / max(w, h)
    normalized_path = (path_array * scale).astype(np.int32)
    return normalized_path

def find_animal_courses(lat: float, lon: float, radius: int, threshold: float):
    """
    주어진 좌표 주변에서 동물 모양과 유사한 코스를 찾아 반환하는 핵심 서비스 함수
    """
    center_point = (lat, lon)
    
    # 1. OSMnx로 경로 데이터 가져오기 및 중복 제거
    try:
        print(f"좌표 ({lat}, {lon}) 주변 {radius}m 반경 탐색 시작...")
        graph = ox.graph_from_point(center_point, dist=radius, network_type='walk')
        edges_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        
        unique_paths_set = set()
        map_path_vectors = []
        for _, row in edges_gdf.iterrows():
            line = row['geometry']
            if line:
                coords_list = list(line.coords)
                sorted_coords = sorted(coords_list)
                path_tuple = tuple(map(tuple, sorted_coords))
                if path_tuple not in unique_paths_set:
                    unique_paths_set.add(path_tuple)
                    map_path_vectors.append(np.array(coords_list))
        print(f"고유 경로 {len(map_path_vectors)}개 발견.")
    except Exception as e:
        print(f"OSMnx 오류: {e}")
        return []

    # 2. 모든 동물 벡터와 경로 비교
    all_found_courses = []
    
    # 미리 로드된 동물 벡터를 순회
    for animal_name, animal_vector in ANIMAL_VECTORS.items():
        animal_vector_normalized = normalize_path(animal_vector)
        
        for path_vector in map_path_vectors:
            if len(path_vector) < 3: continue
            
            path_vector_normalized = normalize_path(path_vector)
            distance = cv2.matchShapes(animal_vector_normalized, path_vector_normalized, cv2.CONTOURS_MATCH_I1, 0.0)
            
            if distance < threshold:
                path_as_dicts = [{"latitude": lat, "longitude": lon} for lon, lat in path_vector]

                all_found_courses.append({
                    "animal_name": animal_name,
                    "score": distance,
                    "path": path_as_dicts
                })

    # 3. 결과 정렬 후 반환
    sorted_courses = sorted(all_found_courses, key=lambda x: x['score'])
    print(f"최종 {len(sorted_courses)}개의 유사 코스 발견.")
    return sorted_courses