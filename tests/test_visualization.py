"""시각화 모듈 테스트"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import CropRegionVisualizer, create_crop_map


class TestCropRegionVisualizer:
    """CropRegionVisualizer 클래스 테스트"""

    @pytest.fixture
    def visualizer(self):
        """테스트용 visualizer 생성"""
        viz = CropRegionVisualizer()
        viz.load_data()
        return viz

    def test_load_data(self, visualizer):
        """데이터 로드 테스트"""
        assert visualizer.gdf is not None
        assert visualizer.soil_data is not None
        assert visualizer.merged_data is not None

    def test_get_available_crops(self, visualizer):
        """사용 가능 농작물 목록 테스트"""
        crops = visualizer.get_available_crops()
        assert isinstance(crops, list)
        assert len(crops) > 0
        assert "사과" in crops

    def test_search_by_crop(self, visualizer):
        """농작물 검색 테스트"""
        regions = visualizer.search_by_crop("사과", top_n=10)
        assert len(regions) <= 10
        assert "최적지_점수" in regions.columns

    def test_search_by_crop_invalid(self, visualizer):
        """존재하지 않는 농작물 검색 테스트"""
        with pytest.raises(ValueError, match="찾을 수 없습니다"):
            visualizer.search_by_crop("존재하지않는농작물")

    def test_create_map(self, visualizer):
        """지도 생성 테스트"""
        regions = visualizer.search_by_crop("사과", top_n=5)
        m = visualizer.create_map(regions, "테스트 지도")
        assert m is not None

    def test_method_chaining(self):
        """메서드 체이닝 테스트"""
        viz = CropRegionVisualizer().load_data()
        assert viz.merged_data is not None


class TestCreateCropMap:
    """create_crop_map 함수 테스트"""

    def test_create_map_without_save(self):
        """저장 없이 지도 생성 테스트"""
        m = create_crop_map("사과", top_n=5, save=False)
        assert m is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
