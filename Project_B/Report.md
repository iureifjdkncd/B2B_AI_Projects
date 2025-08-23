## 확률적 예측 기반 제당 공정 품질 추론 및 공조건 최적화 시스템
(S사 제당공정 기업 / AI 바우처 사업 / 2024.05 ~ 2025.03)

---
### 디렉토리 구조
- `1.) ML유량예측모델.ipynb` : 포충 1·2탑 ML 공급유량 예측
- `2.) FL_BRIX예측모델.ipynb` : MVR 공정 FL_BRIX 농도 예측
- `3.) 실시간 추론 작업형.ipynb` : 실시간 추론 검증 및 점검
- `main.ipynb` : FastAPI 기반 추론 서버

---
### 사용 환경
- Python 3.9.13
- pandas(1.5.3), numpy(1.23.1), scipy(1.9.1), tensorflow(2.7.0), scikit-learn(1.2.2), pymongo(4.10.1), lightgbm(3.3.5), xgboost(1.7.4), fastapi(0.115.12), uvicorn(0.34.3)

---
### 문제 정의
- 1.) **다음 시점 품질 예측**
  - 공정 구간: [원당 → 용해 → 포충 → 여과 → 정제 → 이중효용관 → MVR]
  - 예측 대상: 포충 1·2탑 ML 공급유량, MVR 공정 FL_BRIX 농도
  - 목표: 10분 후 3개 품질 변수 동시 예측
- 2.) **목표값 기반 최적 조건 추천**
  - 사용자 정의 혹은 예측 목표값 대비 **같거나 큰 품질을 가진 과거 조건 탐색 필요**
  - Data Drift 및 불확실성을 고려한 **적응형 예측 및 추천 로직** 필요
---

### 주요 전처리 
- 1.) **시계열 데이터 정의**
  - ML 접근: Target값을 1시점 뒤로 Shift
  - DL 접근: (Batch, TimeStamp, Features) 형태의 3D 배열 구성
  - Train/Test 순차적 분할 (랜덤 셔플 불가)
- 2.) **부분적 제어**
  - 점검 기간 데이터 제거
  - 필요 시 공정 시퀀스 특성에 따라 입력 데이터 시차 지연 적용
- 3.) **Recipe 데이터 구축**
  - 조건 탐색을 위해 입력 Feature들의 합산/재배열 기반 **Recipe Data** 생성

---

### 학습 프로세스  

- 1.) **ML/DL 기반 점추정 & 확률 추정**
  - ML: LightGBM, XGBoost, GBM, HistGBM → Quantile Regression 적용
    - 각 Target 품질의 CV(변동계수)에 기반하여 Lower/Upper Percentile 정의
  - DL: MC Dropout 기반 LSTM 계열 모델(N=100 Simulation)
    - 예측 분포를 mean ± k·std 형태로 정량화
    - K → 각 Target 품질의 CV(변동계수)에 기반
- 2.) **예측 모델 검증**
  - ML/DL 모델 각각의 추정 결과 비교
  - 점추정 vs 확률 분포 기반 성능 차이 분석
- 3.) **ML공급유량1,2 / FL_BRIX 중 예측 결과 예시**
     <img width="695" height="100" alt="화면 캡처 2025-07-29 171327" src="https://github.com/user-attachments/assets/b4bcce22-d918-475d-a890-154436c55572" />

---

### 실시간 추론 프로세스 
- 1.) **최초 예측값 Ensemble**
  - MongoDB에서 실시간 데이터(최근 100개) 조회 및 전처리
  - 각 ML/DL 모델의 예측값을 수집 후 KDE 기반 Ensemble 수행
  - KDE 분포의 변동계수(CV) 기반으로 Upper/Lower Boundary 설정
  - 최종 평균 추정치(Final Mean)를 산출하여 Data Drift에 강건한 예측 확보
- 2.) **목표값 대비 예측값 보정**
  - 사용자 정의 or 예측 목표값 입력 (ML 공급유량 1·2, FL_BRIX 농도)
  - 각 KDE 분포에서 목표 이상 값 1차 선택
  - **Gamma-KDE 보정** 수행:
    - 최근 100개 각 Target 실측값의 표준편차로 Gamma 분포 파라미터 추정
    - KDE + Gamma-KDE 이중 보정을 통해 변동성 반영
  - Gamma-KDE 기반 분포에서 목표 이상 값 최종 선택
- 3.) **목표값 기반 최근사 예측값 추적 예시**
  - 목표 FL_BRIX농도=76.5 / KDE예측값 = 76.401 / Gamma-KDE예측값 = 76.5
     <img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/64338589-0b0c-44d4-a568-93f511cf2b33" />
- 4.) **목표값 대비 최적 과거 다변량 공정조건 선택**
  - (1-3)과정을 통해서 **목표값 → 최근사 예측값**으로 기준 확립
  - **Recipe Data 기반 조건 탐색**
    - 전체 Recipe 데이터(**Recipe Full**)와 Min/Max Filtering 데이터(**Recipe Info**) 활용
  - **df_origin / df_cluster 이중 트랙 기반 탐색**
    - 동일 원산지 조건 → df_origin 필터링
    - 원산지+원당 기준 K-Means Clustering 수행 → 실시간 데이터의 소속 Cluster 탐색
    - 소속 Cluster 내 조건 부족 시, 최근접 Cluster 데이터로 대체
  - **공정조건 경로탐색 로직**
    - 목표 이상 다수 품질값을 동시에 만족하는 과거 공정조건 후보군 생성
    - 후보군 중 최종 **FL_BRIX 농도를 기준**으로 단일 최적 조건 선택
    - 조건 과대/과소 추정을 방지하기 위한 예외 처리 함수 적용
  - **지속가능성 확보**
    - 실시간 입력 데이터는 **Recipe DB**에 실시간 업데이트하여 장기 추론 가능성 강화

---

### 전체 프로세스 예시 

- 1.) 실시간 100개 데이터 조회 후, 품질의 표준편차·최근 품질값 및 목표값 정의 

     <img width="500" height="100" alt="image" src="https://github.com/user-attachments/assets/fd4fb9a4-0fa0-4370-96f3-9b3b0d02f88b" />

- 2.) 목표값 기반 예측값 선택 (KDE + Gamma-KDE 보정 반영)

     <img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/6393e371-eb29-4752-a36a-93df0f6be041" />

- 3.) 최적 과거 공정조건 탐색 및 최종 추천

     <img width="1000" height="300" alt="image" src="https://github.com/user-attachments/assets/2719e071-d3b5-472c-addc-55d24a743e6c" />

     <img width="1000" height="200" alt="image" src="https://github.com/user-attachments/assets/f409db70-77ce-436e-9de7-5c0300569361" />

---



  

