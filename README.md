# 농작물 재배 추천 지역 및 가격 예측 프로젝트

전국 2,072개 행정동의 토양 데이터를 활용한 **농작물 재배 최적 지역 시각화**와 기상 데이터 기반 **6개 농작물 가격 예측 모델** 프로젝트입니다.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Selenium](https://img.shields.io/badge/Selenium-4.8+-green.svg)](https://selenium.dev)
[![Folium](https://img.shields.io/badge/Folium-0.14+-orange.svg)](https://python-visualization.github.io/folium/)

> **🗺️ [인터랙티브 지도 보기](https://m2222n.github.io/smartfarm_region_price/maps/)** - 사과, 감귤, 배추 최적 재배 지역

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **기간** | 2023.06.20 ~ 2023.07.31 (6주) |
| **인원** | 3명 |
| **역할** | 데이터 수집, 데이터 웨어하우스 구축, 지도 시각화, 머신러닝 모델링 (A to Z) |

## 기술 스택

| 분류 | 기술 |
|------|------|
| **데이터 수집** | Python, Selenium, REST API (KAMIS, 기상청) |
| **데이터 분석** | Pandas, NumPy, GeoPandas |
| **시각화** | Folium, Matplotlib, Seaborn |
| **모델링** | Scikit-learn (Linear, Ridge, Lasso, Polynomial, Random Forest) |
| **인프라** | AWS, SQL |

---

## 주요 기능

### 1. 농작물 재배 추천 지역 시각화

전국 토양 데이터(농촌진흥청)를 Selenium으로 크롤링하여 **64개 농작물**의 최적 재배 지역을 지도로 시각화합니다.

#### 데이터 수집 규모 (블로그 기록 기준)

| 항목 | 수치 |
|------|------|
| **총 파라미터** | 321,664개 (64개 농작물 × 전국 읍면동) |
| **파라미터 수집 시간** | 약 1.5시간 |
| **데이터 변환 시간** | 약 5.5시간 |
| **결과 파일** | 33개 CSV로 분할 저장 |

#### 기능 A: 농작물 → 최적 지역 검색
```python
# 예시: 사과가 잘 자라는 전국 지역 Top 30
from src.visualization import CropRegionVisualizer

viz = CropRegionVisualizer()
viz.load_data()
regions = viz.search_by_crop("사과", top_n=30)
```

#### 기능 B: 지역 → 추천 농작물 검색
```python
# 예시: 경상북도 안동에서 잘 자라는 농작물 Top 15
crops = viz.search_by_region("경상북도", "안동", top_n=15)
```

#### 토양 성분 정보
- pH(산도), 유기물, 유효인산, 칼륨, 칼슘, 마그네슘, 전기전도도

### 2. 농작물 가격 예측 모델

기상 데이터(기온, 강수량, 풍속, 습도, 일조시간)와 거래량을 활용하여 **6개 농작물**의 주간 가격을 예측합니다.

#### 대상 농작물
| 농작물 | 데이터 기간 | 샘플 수 |
|--------|-------------|---------|
| 사과 | 2018-2022 | 248주 |
| 양파 | 2018-2022 | 249주 |
| 배추 | 2018-2022 | 249주 |
| 무 | 2018-2022 | 249주 |
| 감귤 | 2018-2022 | 248주 |
| 복숭아 | 2018-2022 | 120주 |

#### 모델 성능 (RMSE 기준)

| 농작물 | Linear | Ridge | Lasso | Random Forest | 가격 표준편차 |
|--------|--------|-------|-------|---------------|--------------|
| 사과 | 685.8 | 659.0 | **651.7** | 666.9 | 613.9 |
| 양파 | 267.6 | 265.1 | 264.4 | **234.5** | 334.7 |
| 배추 | - | - | - | - | - |
| 감귤 | - | - | - | - | - |

> **양파, 감귤**: RMSE가 가격 표준편차보다 낮아 예측력 유의미

---

## 프로젝트 구조

```
smartfarm_region_price/
├── README.md
├── requirements.txt
├── region_code_mapping.csv      # 행정동↔법정동 코드 매핑 테이블
├── data/
│   ├── processed/               # 전처리된 데이터
│   │   ├── final_apple.csv      # 사과 가격+기상 데이터
│   │   ├── final_onion.csv      # 양파 가격+기상 데이터
│   │   ├── final_baechu.csv     # 배추 가격+기상 데이터
│   │   ├── final_radish.csv     # 무 가격+기상 데이터
│   │   ├── final_gyul.csv       # 감귤 가격+기상 데이터
│   │   ├── final_peach.csv      # 복숭아 가격+기상 데이터
│   │   ├── final_soil_ratio.csv # 토양 적합도
│   │   └── final_soil_detail.csv # 토양 성분 상세
│   └── geo/
│       └── HangJeongDong_ver20230701.geojson  # 행정동 경계
├── src/
│   ├── __init__.py
│   ├── config.py            # 설정값 중앙관리
│   ├── data_loader.py       # 데이터 로딩
│   ├── price_model.py       # 가격 예측 모델
│   └── visualization.py     # 지도 시각화
├── notebooks/
│   ├── 01_data_crawling.ipynb       # Selenium 크롤링 (참고용)
│   ├── 02_region_visualization.ipynb # 지도 시각화 데모
│   └── 03_price_prediction.ipynb    # 가격 예측 분석
├── outputs/
│   └── maps/
│       └── map.html         # 인터랙티브 지도
└── docs/
    └── presentation.pdf     # 발표자료
```

## 데이터 소스

| 데이터 | 출처 | 수집 방법 | 비고 |
|--------|------|----------|------|
| 토양 적합도 | 농촌진흥청 흙토람 | Selenium 크롤링 | 32만+ 파라미터, 7시간 소요 |
| 농산물 가격 | KAMIS 농산물유통정보 | REST API | 2018-2022년 주간 데이터 |
| 기상 데이터 | 기상청 종관기상관측 | REST API | 212개 관측지점, 5년, 39만행 |
| 행정동 경계 | 통계청 | GeoJSON | 2023.07.01 기준 |
| 행정동↔법정동 연계표 | 행정안전부 | CSV | 코드 매핑용 |

---

## 실행 방법

### 환경 설정

```bash
# 1. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt
```

### 가격 예측 모델 실행

```python
from src.data_loader import load_all_crops
from src.price_model import compare_models

# 6개 농작물 데이터 로드
crops_data = load_all_crops()

# 모델 비교 (Linear, Ridge, Lasso, Random Forest)
results = compare_models(crops_data)
print(results)
```

### 지도 시각화 실행

```python
from src.visualization import CropRegionVisualizer

viz = CropRegionVisualizer()
viz.load_data()

# 사과 최적 재배 지역 Top 30
regions = viz.search_by_crop("사과", top_n=30)
m = viz.create_map(regions, title="사과 최적 재배 지역")
viz.save_map(m, "map_apple.html")
```

### Jupyter 노트북 실행

```bash
jupyter notebook notebooks/
```

## 주요 성과

- 전국 **2,072개 법정동**에서 **64개 농작물**의 최적 재배 지역 시각화
- 양파, 감귤 가격 예측 모델에서 **RMSE가 표준편차 이하**로 유의미한 예측력 확인
- Selenium을 활용한 **32만개 파라미터 대규모 크롤링** 파이프라인 구축 (7시간 소요)

---

## 한계점 및 개선 방향

### 한계점
| 한계 | 설명 |
|------|------|
| **전국 평균 기상 데이터** | 지역별 세분화된 기상 데이터 필요 |
| **외부 요인 미반영** | 수입량, 정책 변화, 재해 등 |
| **데이터 기간** | 2018-2022년으로 최신 트렌드 미반영 |

### 개선 방향
- [ ] 시계열 모델 (LSTM, Prophet) 적용
- [ ] 지역별 기상 데이터 연동 (관측지점별)
- [ ] 실시간 API 연동 대시보드 구축

---

## 기술적 도전과 해결

### 1. 대규모 Selenium 크롤링

**도전:** 32만개 파라미터를 안정적으로 수집해야 함

**해결 방법:**
- 10,000개 단위 배치 처리로 메모리 관리
- `time.sleep(2)` 페이지 로딩 대기
- 에러 발생 시 로깅 후 계속 진행
- 33개 CSV 파일로 분할 저장

```python
# 배치 처리 예시
for param in tqdm(parameter_list[10001:20001]):
    try:
        response = requests.post(GET_CODE_URL, data=param)
        code_list.append(response.json())
    except Exception as e:
        print("ERROR:", param)
```

### 2. 행정동↔법정동 코드 매핑

**도전:** GeoJSON(행정동)과 토양 데이터(법정동)의 코드 체계가 다름

**해결 방법:**
- 행정안전부 연계표(CSV) 활용
- 다대일(N:1) 매핑 처리 (여러 법정동 → 하나의 행정동)

### 3. 팀원 이탈 대응

**도전:** 4명 중 1명이 중도 하차하여 업무 재분배 필요

**해결 방법:**
- 전체 파이프라인(데이터 수집 → 전처리 → 모델링)을 혼자 담당
- End-to-End 역량 확보

---

## 기여도

| 역할 | 기여도 |
|------|--------|
| 데이터 수집 (Selenium, API) | 80% |
| 지도 시각화 (Folium) | 80% |
| ML 모델링 | 80% |
| 데이터 웨어하우스 구축 | 70% |

---

## 참고 자료

- [프로젝트 블로그 기록](https://m2222n.tistory.com/)
- [농촌진흥청 흙토람](http://soil.rda.go.kr/)
- [KAMIS 농산물유통정보](https://www.kamis.or.kr/)
- [기상청 기상자료개방포털](https://data.kma.go.kr/)

## 라이선스

이 프로젝트는 학습 목적으로 제작되었습니다.
