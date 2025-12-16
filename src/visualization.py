"""
농작물 재배 추천 지역 시각화 모듈

Folium을 활용한 전국 단위 농작물 최적 재배 지역 지도 시각화
- 농작물 → 최적 지역 검색
- 지역 → 추천 농작물 검색

Note:
    GeoJSON의 행정동코드(adm_cd2)와 토양 데이터의 법정동코드가 다르므로
    연계표.csv를 통해 매핑합니다.
"""

from __future__ import annotations

import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from pathlib import Path
from typing import Optional, List, Union

from .config import (
    SOIL_COLUMNS,
    SOIL_SCORE_COLUMNS,
    SOIL_RATIO_FILE,
    GEOJSON_FILE,
    LINK_TABLE_FILE,
    LINK_TABLE_COLUMNS,
    DEFAULT_ENCODING,
    MAP_CONFIG,
    get_data_path,
    get_geo_path,
    get_project_root,
    get_output_path,
)


class CropRegionVisualizer:
    """
    농작물 재배 추천 지역 시각화 클래스

    전국 2,072개 행정동의 토양 데이터를 기반으로
    농작물별 최적 재배 지역을 지도로 시각화합니다.

    Example:
        >>> viz = CropRegionVisualizer()
        >>> viz.load_data()
        >>>
        >>> # 농작물 → 최적 지역
        >>> regions = viz.search_by_crop("사과", top_n=30)
        >>> viz.create_map(regions, "사과 최적 재배 지역")
        >>>
        >>> # 지역 → 추천 농작물
        >>> crops = viz.search_by_region("경상북도", "안동시", top_n=15)
    """

    # 토양 데이터 컬럼명
    CROP_NAME_COL = "작물이름"
    CODE_COL = "법정동코드"

    def __init__(
        self,
        data_path: Optional[Path] = None,
        project_root: Optional[Path] = None
    ):
        """
        Args:
            data_path: 데이터 디렉토리 경로 (기본값: data/)
            project_root: 프로젝트 루트 경로 (연계표.csv 위치)
        """
        self.project_root = project_root or get_project_root()
        self.data_path = data_path or get_data_path()

        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.soil_data: Optional[pd.DataFrame] = None
        self.link_table: Optional[pd.DataFrame] = None
        self.merged_data: Optional[gpd.GeoDataFrame] = None

    def load_data(self) -> "CropRegionVisualizer":
        """
        GeoJSON, 토양 데이터, 연계표 로드

        Returns:
            self (체이닝 지원)

        Raises:
            FileNotFoundError: 필수 파일이 없는 경우
        """
        self._load_geojson()
        self._load_soil_data()
        self._load_link_table()
        self._merge_data()

        return self

    def _load_geojson(self) -> None:
        """GeoJSON 로드 (행정동 경계)"""
        geojson_path = get_geo_path() / GEOJSON_FILE
        if not geojson_path.exists():
            raise FileNotFoundError(f"GeoJSON 파일을 찾을 수 없습니다: {geojson_path}")

        self.gdf = gpd.read_file(geojson_path)

    def _load_soil_data(self) -> None:
        """토양 적합도 데이터 로드"""
        soil_path = self.data_path / "processed" / SOIL_RATIO_FILE
        if not soil_path.exists():
            raise FileNotFoundError(f"토양 데이터를 찾을 수 없습니다: {soil_path}")

        self.soil_data = pd.read_csv(soil_path, encoding=DEFAULT_ENCODING)

    def _load_link_table(self) -> None:
        """연계표 로드 (행정동코드 ↔ 법정동코드 매핑)"""
        link_path = self.project_root / LINK_TABLE_FILE
        if link_path.exists():
            self.link_table = pd.read_csv(link_path, encoding=DEFAULT_ENCODING)
            # 컬럼명 정규화 (인코딩 문제 대비)
            self.link_table.columns = LINK_TABLE_COLUMNS
        else:
            print(f"Warning: 연계표 파일을 찾을 수 없습니다: {link_path}")
            print("직접 코드 매핑을 시도합니다.")
            self.link_table = None

    def _merge_data(self) -> None:
        """
        GeoJSON과 토양 데이터 병합

        연계표가 있으면: GeoJSON(adm_cd2) → 연계표(행정동코드→법정동코드) → 토양(법정동코드)
        연계표가 없으면: 직접 매핑 시도
        """
        if self.gdf is None or self.soil_data is None:
            raise ValueError("먼저 load_data()를 실행하세요.")

        # 코드 타입 통일 (문자열)
        self.gdf["adm_cd2"] = self.gdf["adm_cd2"].astype(str).str.strip()
        self.soil_data[self.CODE_COL] = self.soil_data[self.CODE_COL].astype(str).str.strip()

        if self.link_table is not None:
            merged = self._merge_with_link_table()
        else:
            merged = self._merge_direct()

        self.merged_data = gpd.GeoDataFrame(merged, geometry="geometry")
        self._calculate_score()

        print(f"병합 완료: {len(self.merged_data)} 레코드")

    def _merge_with_link_table(self) -> pd.DataFrame:
        """연계표를 통한 매핑"""
        link_cols = LINK_TABLE_COLUMNS[1:]  # ["행정동코드", "법정동코드"]

        self.link_table[link_cols[0]] = self.link_table[link_cols[0]].astype(str).str.strip()
        self.link_table[link_cols[1]] = self.link_table[link_cols[1]].astype(str).str.strip()

        # 1단계: GeoJSON + 연계표 (행정동코드 기준)
        gdf_with_link = pd.merge(
            self.gdf,
            self.link_table[link_cols].drop_duplicates(),
            left_on="adm_cd2",
            right_on=link_cols[0],
            how="left"
        )

        # 2단계: 연계표 결과 + 토양 데이터 (법정동코드 기준)
        gdf_with_link = gdf_with_link.rename(columns={link_cols[1]: "연계_법정동코드"})

        return pd.merge(
            gdf_with_link,
            self.soil_data,
            left_on="연계_법정동코드",
            right_on=self.CODE_COL,
            how="inner"
        )

    def _merge_direct(self) -> pd.DataFrame:
        """직접 매핑 (연계표 없는 경우)"""
        return pd.merge(
            self.gdf,
            self.soil_data,
            left_on="adm_cd2",
            right_on=self.CODE_COL,
            how="inner"
        )

    def _calculate_score(self) -> None:
        """최적지 점수 계산 (최적지 가중치 2배)"""
        best_col = SOIL_SCORE_COLUMNS["best"]
        good_col = SOIL_SCORE_COLUMNS["good"]

        if best_col in self.merged_data.columns and good_col in self.merged_data.columns:
            self.merged_data["최적지_점수"] = (
                self.merged_data[best_col] * 2 + self.merged_data[good_col]
            ) / 2

    def get_available_crops(self) -> List[str]:
        """사용 가능한 농작물 목록 반환"""
        if self.merged_data is None:
            raise ValueError("먼저 load_data()를 실행하세요.")

        if self.CROP_NAME_COL in self.merged_data.columns:
            return sorted(self.merged_data[self.CROP_NAME_COL].unique().tolist())
        return []

    def get_available_regions(self) -> pd.DataFrame:
        """사용 가능한 지역 목록 반환 (시도, 법정동)"""
        if self.merged_data is None:
            raise ValueError("먼저 load_data()를 실행하세요.")

        return (
            self.merged_data[["sidonm", "adm_nm"]]
            .drop_duplicates()
            .sort_values(["sidonm", "adm_nm"])
        )

    def search_by_crop(
        self,
        crop_name: str,
        top_n: int = 50
    ) -> gpd.GeoDataFrame:
        """
        농작물 기준 최적 재배 지역 검색

        Args:
            crop_name: 농작물 이름 (예: "사과", "배추", "감귤")
            top_n: 반환할 지역 수

        Returns:
            최적 지역 GeoDataFrame (최적지 점수 내림차순)

        Raises:
            ValueError: 농작물을 찾을 수 없는 경우
        """
        if self.merged_data is None:
            raise ValueError("먼저 load_data()를 실행하세요.")

        filtered = self.merged_data[
            self.merged_data[self.CROP_NAME_COL] == crop_name
        ].copy()

        if filtered.empty:
            available = self.get_available_crops()[:10]
            raise ValueError(
                f"'{crop_name}' 농작물을 찾을 수 없습니다. "
                f"사용 가능: {available}..."
            )

        return (
            filtered
            .sort_values("최적지_점수", ascending=False)
            .drop_duplicates(self.CODE_COL)
            .head(top_n)
        )

    def search_by_region(
        self,
        sido: str,
        dong: str,
        top_n: int = 15
    ) -> gpd.GeoDataFrame:
        """
        지역 기준 추천 농작물 검색

        Args:
            sido: 시도명 (예: "경상북도", "전라남도")
            dong: 법정동명 (예: "안동시", "목포시")
            top_n: 반환할 농작물 수

        Returns:
            추천 농작물 GeoDataFrame (최적지 점수 내림차순)

        Raises:
            ValueError: 지역을 찾을 수 없는 경우
        """
        if self.merged_data is None:
            raise ValueError("먼저 load_data()를 실행하세요.")

        filtered = self.merged_data[
            (self.merged_data["sidonm"] == sido) &
            (self.merged_data["adm_nm"].str.contains(dong, na=False))
        ].copy()

        if filtered.empty:
            raise ValueError(f"'{sido} {dong}' 지역을 찾을 수 없습니다.")

        return (
            filtered
            .sort_values("최적지_점수", ascending=False)
            .drop_duplicates(self.CROP_NAME_COL)
            .head(top_n)
        )

    def create_map(
        self,
        data: gpd.GeoDataFrame,
        title: str = "농작물 재배 추천 지역",
        show_markers: bool = True,
        show_boundaries: bool = True,
        show_ranking: bool = True,
        tiles: Optional[str] = None
    ) -> folium.Map:
        """
        지도 시각화 생성

        Args:
            data: 시각화할 GeoDataFrame
            title: 지도 제목
            show_markers: 마커 클러스터 표시 여부
            show_boundaries: 행정 경계 표시 여부
            show_ranking: 순위 차트 패널 표시 여부
            tiles: 지도 타일 스타일

        Returns:
            Folium Map 객체
        """
        if tiles is None:
            tiles = MAP_CONFIG["tiles"]

        m = folium.Map(
            location=MAP_CONFIG["center"],
            zoom_start=MAP_CONFIG["zoom"],
            tiles=tiles
        )

        self._add_title(m, title)

        if show_boundaries:
            self._add_boundaries(m, data)

        if show_markers and "geometry" in data.columns:
            self._add_markers(m, data)

        if show_ranking:
            self._add_ranking_panel(m, data, title)

        return m

    def _add_title(self, m: folium.Map, title: str) -> None:
        """지도에 제목 추가"""
        title_html = f'''
            <div style="position: fixed;
                        top: 10px; left: 50px;
                        z-index: 9999;
                        background-color: white;
                        padding: 10px;
                        border-radius: 5px;
                        box-shadow: 2px 2px 5px gray;">
                <h4>{title}</h4>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

    def _add_ranking_panel(
        self,
        m: folium.Map,
        data: gpd.GeoDataFrame,
        title: str
    ) -> None:
        """지도에 순위 차트 패널 추가"""
        best_col = SOIL_SCORE_COLUMNS["best"]
        good_col = SOIL_SCORE_COLUMNS["good"]

        # 순위 데이터 준비 (최대 15개)
        ranking_data = data.head(15).copy()

        # 순위 테이블 HTML 생성
        rows_html = ""
        for idx, row in enumerate(ranking_data.itertuples(), 1):
            region = getattr(row, "adm_nm", "알 수 없음")
            sido = getattr(row, "sidonm", "")
            best = getattr(row, best_col.replace(" ", "_").replace("당", "당"), 0) if hasattr(row, best_col.replace(" ", "_")) else row._asdict().get(best_col, 0)
            good = getattr(row, good_col.replace(" ", "_").replace("당", "당"), 0) if hasattr(row, good_col.replace(" ", "_")) else row._asdict().get(good_col, 0)

            # DataFrame에서 직접 값 가져오기
            best_val = ranking_data.iloc[idx-1].get(best_col, 0)
            good_val = ranking_data.iloc[idx-1].get(good_col, 0)
            region_name = ranking_data.iloc[idx-1].get("adm_nm", "알 수 없음")
            sido_name = ranking_data.iloc[idx-1].get("sidonm", "")

            # 막대 그래프 너비 계산 (최대값 기준 비율)
            max_score = ranking_data[best_col].max() if best_col in ranking_data.columns else 1
            bar_width = (best_val / max_score * 100) if max_score > 0 else 0

            medal = ""
            if idx == 1:
                medal = "🥇"
            elif idx == 2:
                medal = "🥈"
            elif idx == 3:
                medal = "🥉"

            rows_html += f'''
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 4px 8px; text-align: center; font-weight: bold;">{medal}{idx}</td>
                    <td style="padding: 4px 8px; font-size: 11px;">
                        <div>{region_name}</div>
                        <div style="font-size: 10px; color: #888;">{sido_name}</div>
                    </td>
                    <td style="padding: 4px 8px; width: 100px;">
                        <div style="background: linear-gradient(90deg, #3498db {bar_width}%, #ecf0f1 {bar_width}%);
                                    height: 16px; border-radius: 3px;"></div>
                    </td>
                    <td style="padding: 4px 8px; text-align: right; font-size: 11px; font-weight: bold;">{best_val:.2f}</td>
                </tr>
            '''

        ranking_html = f'''
            <div id="ranking-panel" style="
                position: fixed;
                top: 60px; right: 10px;
                z-index: 9999;
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                max-height: 80vh;
                overflow-y: auto;
                width: 320px;
                font-family: 'Malgun Gothic', sans-serif;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: #2c3e50;">📊 Top 15 순위</h4>
                    <button onclick="document.getElementById('ranking-panel').style.display='none'"
                            style="border: none; background: #eee; border-radius: 50%; width: 24px; height: 24px; cursor: pointer;">✕</button>
                </div>
                <p style="font-size: 11px; color: #7f8c8d; margin-bottom: 10px;">
                    면적당 최적지 비율 기준
                </p>
                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <thead>
                        <tr style="background: #f8f9fa; border-bottom: 2px solid #ddd;">
                            <th style="padding: 6px; text-align: center;">순위</th>
                            <th style="padding: 6px; text-align: left;">지역</th>
                            <th style="padding: 6px; text-align: center;">비율</th>
                            <th style="padding: 6px; text-align: right;">점수</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
                <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #eee;">
                    <p style="font-size: 10px; color: #95a5a6; margin: 0;">
                        📍 마커 클릭 시 상세 정보 확인<br>
                        데이터: 농촌진흥청 흙토람
                    </p>
                </div>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(ranking_html))

    def _add_boundaries(self, m: folium.Map, data: gpd.GeoDataFrame) -> None:
        """행정 경계 추가"""
        folium.GeoJson(
            data,
            style_function=lambda x: {
                "fillColor": "#3498db",
                "color": "#2c3e50",
                "weight": 1,
                "fillOpacity": 0.4
            }
        ).add_to(m)

    def _add_markers(self, m: folium.Map, data: gpd.GeoDataFrame) -> None:
        """마커 클러스터 추가"""
        marker_cluster = MarkerCluster().add_to(m)
        centroids = data.geometry.centroid

        for idx, (lat, lon) in enumerate(zip(centroids.y, centroids.x)):
            row = data.iloc[idx]
            popup_html = self._create_popup_html(row)

            folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color="green", icon="leaf")
            ).add_to(marker_cluster)

    def _create_popup_html(self, row: pd.Series) -> str:
        """마커 팝업 HTML 생성"""
        region_name = row.get("adm_nm", row.get("법정동", "알 수 없음"))
        crop_name = row.get(self.CROP_NAME_COL, "")
        best_score = row.get(SOIL_SCORE_COLUMNS["best"], 0)
        good_score = row.get(SOIL_SCORE_COLUMNS["good"], 0)

        html = f"""
        <div style="font-family: 'Malgun Gothic', sans-serif;">
            <h4 style="margin: 0; color: #2c3e50;">{region_name}</h4>
        """

        if crop_name:
            html += f"<p><strong>농작물:</strong> {crop_name}</p>"

        html += f"""
            <p><strong>면적당 최적지:</strong> {best_score:.2f}</p>
            <p><strong>면적당 적지:</strong> {good_score:.2f}</p>
            <hr>
            <p style="font-size: 12px; color: #7f8c8d;"><strong>토양 성분</strong></p>
        """

        for col in SOIL_COLUMNS:
            if col in row.index:
                value = row[col]
                if pd.notna(value):
                    html += f"<p style='margin: 2px 0; font-size: 11px;'>{col}: {value:.2f}</p>"

        html += "</div>"
        return html

    def save_map(
        self,
        m: folium.Map,
        filename: str = "map.html",
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        지도 HTML 파일 저장

        Args:
            m: Folium Map 객체
            filename: 저장할 파일명
            output_dir: 출력 디렉토리 (기본값: outputs/maps/)

        Returns:
            저장된 파일 경로
        """
        if output_dir is None:
            output_dir = get_output_path() / "maps"

        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename
        m.save(str(filepath))

        return filepath


def create_crop_map(
    crop_name: str,
    top_n: int = 30,
    save: bool = True
) -> folium.Map:
    """
    농작물 최적 재배 지역 지도 생성 (간편 함수)

    Args:
        crop_name: 농작물 이름
        top_n: 표시할 지역 수
        save: HTML 파일 저장 여부

    Returns:
        Folium Map 객체
    """
    viz = CropRegionVisualizer()
    viz.load_data()

    regions = viz.search_by_crop(crop_name, top_n=top_n)
    m = viz.create_map(regions, title=f"{crop_name} 최적 재배 지역 Top {top_n}")

    if save:
        filepath = viz.save_map(m, f"map_{crop_name}.html")
        print(f"지도가 저장되었습니다: {filepath}")

    return m
