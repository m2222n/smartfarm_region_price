"""가격 예측 모델 테스트"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_crop_data
from src.price_model import CropPricePredictor, ModelResult


class TestCropPricePredictor:
    """CropPricePredictor 클래스 테스트"""

    @pytest.fixture
    def predictor(self):
        """테스트용 predictor 생성"""
        df = load_crop_data("apple")
        return CropPricePredictor(df, "apple")

    def test_init(self, predictor):
        """초기화 테스트"""
        assert predictor.crop_name == "apple"
        assert predictor.X is not None
        assert predictor.y is not None

    def test_train_test_split(self, predictor):
        """학습/테스트 분할 테스트"""
        total = len(predictor.X)
        train_size = len(predictor.X_train)
        test_size = len(predictor.X_test)
        assert train_size + test_size == total

    def test_train_linear(self, predictor):
        """Linear 모델 학습 테스트"""
        result = predictor.train_linear()
        assert isinstance(result, ModelResult)
        assert result.model_name == "linear"
        assert result.rmse > 0

    def test_train_ridge(self, predictor):
        """Ridge 모델 학습 테스트"""
        result = predictor.train_ridge()
        assert isinstance(result, ModelResult)
        assert result.model_name == "ridge"

    def test_train_lasso(self, predictor):
        """Lasso 모델 학습 테스트"""
        result = predictor.train_lasso()
        assert isinstance(result, ModelResult)
        assert result.model_name == "lasso"

    def test_predict(self, predictor):
        """예측 테스트"""
        predictor.train_linear()
        predictions = predictor.predict(predictor.X_test, "linear")
        assert len(predictions) == len(predictor.X_test)

    def test_summary(self, predictor):
        """요약 테이블 테스트"""
        predictor.train_linear()
        predictor.train_ridge()
        summary = predictor.summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2


class TestModelResult:
    """ModelResult 데이터클래스 테스트"""

    def test_create_result(self):
        """결과 객체 생성 테스트"""
        result = ModelResult(
            model_name="test",
            best_params={"alpha": 1.0},
            mse=100.0,
            rmse=10.0,
            mae=8.0,
            r2=0.85
        )
        assert result.model_name == "test"
        assert result.rmse == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
