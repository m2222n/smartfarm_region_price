"""
프로젝트 설정 모듈

모든 하드코딩된 값들을 중앙 관리합니다.
- 파일 경로
- 컬럼명
- 모델 하이퍼파라미터
- 시각화 설정
"""

from pathlib import Path
from typing import Dict, List


# =============================================================================
# 경로 설정
# =============================================================================

def get_project_root() -> Path:
    """프로젝트 루트 경로 반환"""
    return Path(__file__).parent.parent


def get_data_path() -> Path:
    """데이터 디렉토리 경로 반환"""
    return get_project_root() / "data"


def get_processed_path() -> Path:
    """전처리된 데이터 경로 반환"""
    return get_data_path() / "processed"


def get_geo_path() -> Path:
    """지리 데이터 경로 반환"""
    return get_data_path() / "geo"


def get_output_path() -> Path:
    """출력 디렉토리 경로 반환"""
    return get_project_root() / "outputs"


# =============================================================================
# 농작물 설정
# =============================================================================

# 지원하는 농작물 목록 {영문키: 한글명}
CROPS: Dict[str, str] = {
    "apple": "사과",
    "onion": "양파",
    "baechu": "배추",
    "radish": "무",
    "gyul": "감귤",
    "peach": "복숭아",
}


# =============================================================================
# 데이터 컬럼 설정
# =============================================================================

# 가격 예측용 피처 컬럼
FEATURE_COLUMNS: List[str] = [
    "평균기온(°C)",
    "일강수량(mm)",
    "최대 풍속(m/s)",
    "평균 상대습도(%)",
    "합계 일조시간(hr)",
    "일교차(°C)",
    "거래량",
]

# 가격 예측 타겟 컬럼
TARGET_COLUMN: str = "가격/단량"

# 토양 성분 컬럼
SOIL_COLUMNS: List[str] = [
    "산도",
    "유효인산",
    "유기물",
    "마그네슘",
    "칼륨",
    "칼슘",
    "전기전도도",
]

# 토양 적합도 컬럼
SOIL_SCORE_COLUMNS: Dict[str, str] = {
    "best": "면적당 최적지",
    "good": "면적당 적지",
}


# =============================================================================
# 파일명 설정
# =============================================================================

# 농작물별 가격 데이터 파일명 패턴
CROP_FILE_PATTERN: str = "final_{crop}.csv"

# 토양 데이터 파일명
SOIL_RATIO_FILE: str = "final_soil_ratio.csv"
SOIL_DETAIL_FILE: str = "final_soil_except2_df.csv"

# GeoJSON 파일명
GEOJSON_FILE: str = "HangJeongDong_ver20230701.geojson"

# 연계표 파일명
LINK_TABLE_FILE: str = "region_code_mapping.csv"

# 연계표 컬럼명 (인코딩 문제 대비용)
LINK_TABLE_COLUMNS: List[str] = ["동이름", "행정동코드", "법정동코드"]


# =============================================================================
# 인코딩 설정
# =============================================================================

# CSV 파일 기본 인코딩
DEFAULT_ENCODING: str = "utf-8-sig"


# =============================================================================
# 모델 하이퍼파라미터 설정
# =============================================================================

MODEL_PARAM_GRIDS: Dict[str, Dict] = {
    "linear": {
        "fit_intercept": [True, False],
    },
    "ridge": {
        "alpha": [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100, 200, 1000, 2000],
    },
    "lasso": {
        "alpha": [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100, 200, 1000, 2000],
    },
    "polynomial": {
        "polynomial__degree": [1, 2, 3],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
}

# 모델 학습 기본 설정
MODEL_DEFAULTS: Dict[str, any] = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
}


# =============================================================================
# 시각화 설정
# =============================================================================

# 지도 기본 설정
MAP_CONFIG: Dict[str, any] = {
    "center": [36.4109466, 128.1590828],  # 대한민국 중심 좌표
    "zoom": 7,
    "tiles": "cartodbpositron",
}

# 지도 마커 색상
MARKER_COLORS: Dict[str, str] = {
    "default": "green",
    "best": "red",
    "good": "orange",
}
