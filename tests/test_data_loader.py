"""데이터 로더 테스트"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    load_crop_data,
    load_all_crops,
    get_crop_names,
    get_feature_columns,
    get_target_column,
)
from src.config import CROPS, FEATURE_COLUMNS, TARGET_COLUMN


class TestLoadCropData:
    """load_crop_data 함수 테스트"""

    def test_load_apple_data(self):
        """사과 데이터 로드 테스트"""
        df = load_crop_data("apple")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_all_supported_crops(self):
        """모든 지원 농작물 로드 테스트"""
        for crop in CROPS.keys():
            df = load_crop_data(crop)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0

    def test_invalid_crop_raises_error(self):
        """지원하지 않는 농작물 에러 테스트"""
        with pytest.raises(ValueError, match="지원하지 않는 농작물"):
            load_crop_data("invalid_crop")

    def test_data_columns(self):
        """데이터 컬럼 확인 테스트"""
        df = load_crop_data("apple")
        assert TARGET_COLUMN in df.columns


class TestLoadAllCrops:
    """load_all_crops 함수 테스트"""

    def test_returns_dict(self):
        """딕셔너리 반환 테스트"""
        result = load_all_crops()
        assert isinstance(result, dict)
        assert len(result) == len(CROPS)

    def test_all_crops_loaded(self):
        """모든 농작물 로드 확인"""
        result = load_all_crops()
        for crop in CROPS.keys():
            assert crop in result
            assert isinstance(result[crop], pd.DataFrame)


class TestConfigFunctions:
    """설정 함수 테스트"""

    def test_get_crop_names(self):
        """농작물 이름 반환 테스트"""
        crops = get_crop_names()
        assert isinstance(crops, dict)
        assert "apple" in crops
        assert crops["apple"] == "사과"

    def test_get_feature_columns(self):
        """피처 컬럼 반환 테스트"""
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_get_target_column(self):
        """타겟 컬럼 반환 테스트"""
        target = get_target_column()
        assert target == TARGET_COLUMN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
