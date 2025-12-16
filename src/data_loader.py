"""
데이터 로딩 모듈

농작물 가격 데이터 및 토양 데이터 로딩 유틸리티
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Union

from .config import (
    CROPS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    CROP_FILE_PATTERN,
    SOIL_RATIO_FILE,
    DEFAULT_ENCODING,
    get_processed_path,
)


def load_crop_data(
    crop: str,
    data_path: Optional[Path] = None,
    encoding: str = DEFAULT_ENCODING
) -> pd.DataFrame:
    """
    특정 농작물의 가격 데이터 로드

    Args:
        crop: 농작물 이름 (apple, onion, baechu, radish, gyul, peach)
        data_path: 데이터 디렉토리 경로 (기본값: data/processed)
        encoding: 파일 인코딩

    Returns:
        주차별 기상 데이터 + 가격 데이터가 포함된 DataFrame

    Raises:
        ValueError: 지원하지 않는 농작물인 경우
        FileNotFoundError: 데이터 파일이 없는 경우

    Example:
        >>> df = load_crop_data("apple")
        >>> df.columns.tolist()
        ['평균기온(°C)', '일강수량(mm)', ..., '가격/단량']
    """
    if crop not in CROPS:
        raise ValueError(
            f"지원하지 않는 농작물입니다: {crop}. "
            f"지원 목록: {list(CROPS.keys())}"
        )

    if data_path is None:
        data_path = get_processed_path()

    filename = CROP_FILE_PATTERN.format(crop=crop)
    file_path = data_path / filename

    if not file_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")

    df = pd.read_csv(file_path, encoding=encoding)

    # 첫 번째 컬럼(인덱스)과 '주차' 컬럼 처리
    if df.columns[0] == "Unnamed: 0":
        df = df.iloc[:, 1:]

    if "주차" in df.columns:
        df = df.set_index("주차")

    # 결측치 처리
    df = df.fillna(0)

    return df


def load_all_crops(data_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    모든 농작물 데이터 로드

    Args:
        data_path: 데이터 디렉토리 경로 (기본값: data/processed)

    Returns:
        {crop_name: DataFrame} 형태의 딕셔너리

    Example:
        >>> crops_data = load_all_crops()
        >>> crops_data["apple"].shape
        (248, 8)
    """
    if data_path is None:
        data_path = get_processed_path()

    return {crop: load_crop_data(crop, data_path) for crop in CROPS}


def load_soil_data(
    data_path: Optional[Path] = None,
    encoding: str = DEFAULT_ENCODING
) -> pd.DataFrame:
    """
    토양 적합도 데이터 로드 (지도 시각화용)

    Args:
        data_path: 데이터 디렉토리 경로 (기본값: data/processed)
        encoding: 파일 인코딩

    Returns:
        지역별 토양 적합도 DataFrame

    Raises:
        FileNotFoundError: 토양 데이터 파일이 없는 경우
    """
    if data_path is None:
        data_path = get_processed_path()

    file_path = data_path / SOIL_RATIO_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"토양 데이터를 찾을 수 없습니다: {file_path}")

    return pd.read_csv(file_path, encoding=encoding)


def get_feature_columns() -> List[str]:
    """기상 피처 컬럼명 반환"""
    return FEATURE_COLUMNS.copy()


def get_target_column() -> str:
    """타겟 컬럼명 반환"""
    return TARGET_COLUMN


def get_crop_names() -> Dict[str, str]:
    """농작물 목록 반환 {영문: 한글}"""
    return CROPS.copy()
