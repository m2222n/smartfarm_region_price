"""
농작물 가격 예측 모델 모듈

Linear, Ridge, Lasso, Polynomial, Random Forest 회귀 모델을 활용한
6개 농작물 가격 예측
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .config import (
    TARGET_COLUMN,
    MODEL_PARAM_GRIDS,
    MODEL_DEFAULTS,
    get_output_path,
)


@dataclass
class ModelResult:
    """모델 평가 결과"""
    model_name: str
    best_params: Dict[str, Any]
    mse: float
    rmse: float
    mae: float
    r2: float
    cv_score: Optional[float] = None


class CropPricePredictor:
    """
    농작물 가격 예측 모델

    기상 데이터(기온, 강수량, 풍속, 습도, 일조시간, 일교차, 거래량)를 활용하여
    농작물 가격을 예측합니다.

    지원 모델:
        - Linear Regression
        - Ridge Regression (L2 정규화)
        - Lasso Regression (L1 정규화)
        - Polynomial Regression
        - Random Forest Regressor

    Example:
        >>> from src.data_loader import load_crop_data
        >>> df = load_crop_data("apple")
        >>> predictor = CropPricePredictor(df, "apple")
        >>> predictor.train_all_models()
        >>> predictor.get_best_model()
        ModelResult(model_name='lasso', rmse=651.71, ...)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        crop_name: str,
        target_col: str = TARGET_COLUMN,
        test_size: float = MODEL_DEFAULTS["test_size"],
        random_state: int = MODEL_DEFAULTS["random_state"]
    ):
        """
        Args:
            data: 농작물 가격 데이터 (기상 피처 + 타겟)
            crop_name: 농작물 이름
            target_col: 예측 대상 컬럼명
            test_size: 테스트 데이터 비율
            random_state: 랜덤 시드
        """
        self.crop_name = crop_name
        self.target_col = target_col
        self.random_state = random_state
        self.cv_folds = MODEL_DEFAULTS["cv_folds"]

        # 피처/타겟 분리
        self.X = data.drop(target_col, axis=1)
        self.y = data[target_col]
        self.feature_names = list(self.X.columns)

        # 학습/테스트 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # 결과 저장
        self.results: Dict[str, ModelResult] = {}
        self.trained_models: Dict[str, Any] = {}

    def _evaluate_model(
        self,
        model: Any,
        model_name: str,
        best_params: Dict[str, Any]
    ) -> ModelResult:
        """모델 평가"""
        y_pred = model.predict(self.X_test)

        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # 교차 검증 (Random Forest는 시간이 오래 걸려 생략)
        cv_score = None
        if model_name != "random_forest":
            cv_scores = cross_val_score(
                model, self.X, self.y,
                cv=self.cv_folds,
                scoring="neg_mean_squared_error"
            )
            cv_score = np.sqrt(-cv_scores.mean())

        return ModelResult(
            model_name=model_name,
            best_params=best_params,
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            cv_score=cv_score
        )

    def _train_model(
        self,
        model: Any,
        model_name: str,
        param_grid: Dict[str, Any]
    ) -> ModelResult:
        """모델 학습 공통 로직"""
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.cv_folds,
            scoring="neg_mean_squared_error",
            n_jobs=-1 if model_name == "random_forest" else 1
        )
        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_
        self.trained_models[model_name] = best_model

        result = self._evaluate_model(best_model, model_name, grid_search.best_params_)
        self.results[model_name] = result
        return result

    def train_linear(self) -> ModelResult:
        """선형 회귀 모델 학습"""
        return self._train_model(
            LinearRegression(),
            "linear",
            MODEL_PARAM_GRIDS["linear"]
        )

    def train_ridge(self) -> ModelResult:
        """Ridge 회귀 모델 학습 (L2 정규화)"""
        return self._train_model(
            Ridge(),
            "ridge",
            MODEL_PARAM_GRIDS["ridge"]
        )

    def train_lasso(self) -> ModelResult:
        """Lasso 회귀 모델 학습 (L1 정규화)"""
        return self._train_model(
            Lasso(),
            "lasso",
            MODEL_PARAM_GRIDS["lasso"]
        )

    def train_polynomial(self) -> ModelResult:
        """다항 회귀 모델 학습"""
        pipeline = Pipeline([
            ("polynomial", PolynomialFeatures()),
            ("linear_regression", LinearRegression())
        ])
        return self._train_model(
            pipeline,
            "polynomial",
            MODEL_PARAM_GRIDS["polynomial"]
        )

    def train_random_forest(self) -> ModelResult:
        """Random Forest 회귀 모델 학습"""
        return self._train_model(
            RandomForestRegressor(random_state=self.random_state),
            "random_forest",
            MODEL_PARAM_GRIDS["random_forest"]
        )

    def train_all_models(self, verbose: bool = True) -> Dict[str, ModelResult]:
        """
        모든 모델 학습

        Args:
            verbose: 학습 과정 출력 여부

        Returns:
            {model_name: ModelResult} 딕셔너리
        """
        models = [
            ("linear", self.train_linear),
            ("ridge", self.train_ridge),
            ("lasso", self.train_lasso),
            ("polynomial", self.train_polynomial),
            ("random_forest", self.train_random_forest)
        ]

        for name, train_func in models:
            if verbose:
                print(f"[{self.crop_name}] {name} 모델 학습 중...")
            train_func()
            if verbose:
                print(f"  -> RMSE: {self.results[name].rmse:.2f}")

        return self.results

    def get_best_model(self) -> Tuple[str, ModelResult]:
        """
        RMSE 기준 최적 모델 반환

        Returns:
            (model_name, ModelResult) 튜플
        """
        if not self.results:
            raise ValueError("먼저 train_all_models()를 실행하세요.")

        best_name = min(self.results, key=lambda x: self.results[x].rmse)
        return best_name, self.results[best_name]

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Random Forest 피처 중요도 반환

        Returns:
            피처 중요도 DataFrame (내림차순 정렬)
        """
        if "random_forest" not in self.trained_models:
            return None

        rf_model = self.trained_models["random_forest"]
        importance = rf_model.feature_importances_

        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

    def predict(self, X: pd.DataFrame, model_name: str = "random_forest") -> np.ndarray:
        """
        새로운 데이터에 대한 가격 예측

        Args:
            X: 입력 피처 DataFrame
            model_name: 사용할 모델 이름

        Returns:
            예측 가격 배열
        """
        if model_name not in self.trained_models:
            raise ValueError(f"학습되지 않은 모델입니다: {model_name}")

        return self.trained_models[model_name].predict(X)

    def save_model(self, model_name: str, filepath: Optional[str] = None) -> Path:
        """
        모델 저장

        Args:
            model_name: 저장할 모델 이름
            filepath: 저장 경로 (기본값: outputs/models/{crop}_{model}.pkl)

        Returns:
            저장된 파일 경로
        """
        if model_name not in self.trained_models:
            raise ValueError(f"학습되지 않은 모델입니다: {model_name}")

        if filepath is None:
            output_dir = get_output_path() / "models"
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f"{self.crop_name}_{model_name}.pkl"
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self.trained_models[model_name], f)

        return filepath

    def summary(self) -> pd.DataFrame:
        """
        모든 모델 결과 요약 테이블

        Returns:
            모델별 성능 지표 DataFrame
        """
        if not self.results:
            raise ValueError("먼저 train_all_models()를 실행하세요.")

        rows = []
        for name, result in self.results.items():
            rows.append({
                "Model": name,
                "RMSE": round(result.rmse, 2),
                "MAE": round(result.mae, 2),
                "R²": round(result.r2, 4),
                "Best Params": str(result.best_params)
            })

        return pd.DataFrame(rows).sort_values("RMSE")


def compare_models(crops_data: Dict[str, pd.DataFrame], verbose: bool = True) -> pd.DataFrame:
    """
    모든 농작물에 대해 모델 비교

    Args:
        crops_data: {crop_name: DataFrame} 딕셔너리
        verbose: 학습 과정 출력 여부

    Returns:
        농작물별 최적 모델 요약 DataFrame
    """
    results = []

    for crop_name, data in crops_data.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"농작물: {crop_name}")
            print("="*50)

        predictor = CropPricePredictor(data, crop_name)
        predictor.train_all_models(verbose=verbose)

        best_name, best_result = predictor.get_best_model()

        results.append({
            "농작물": crop_name,
            "최적 모델": best_name,
            "RMSE": round(best_result.rmse, 2),
            "R²": round(best_result.r2, 4),
            "가격 표준편차": round(data[TARGET_COLUMN].std(), 2)
        })

    return pd.DataFrame(results)
