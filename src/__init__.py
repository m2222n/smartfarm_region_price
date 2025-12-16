"""
농작물 재배 추천 지역 및 가격 예측 프로젝트

- 토양 데이터 기반 전국 2,072개 행정동 농작물 재배 최적 지역 시각화
- 기상 데이터 기반 6개 농작물(사과, 양파, 배추, 무, 감귤, 복숭아) 가격 예측 모델
"""

from .config import (
    CROPS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SOIL_COLUMNS,
)
from .data_loader import (
    load_crop_data,
    load_all_crops,
    load_soil_data,
    get_feature_columns,
    get_target_column,
    get_crop_names,
)
from .price_model import (
    CropPricePredictor,
    ModelResult,
    compare_models,
)
from .visualization import (
    CropRegionVisualizer,
    create_crop_map,
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "CROPS",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "SOIL_COLUMNS",
    # Data Loader
    "load_crop_data",
    "load_all_crops",
    "load_soil_data",
    "get_feature_columns",
    "get_target_column",
    "get_crop_names",
    # Price Model
    "CropPricePredictor",
    "ModelResult",
    "compare_models",
    # Visualization
    "CropRegionVisualizer",
    "create_crop_map",
]
