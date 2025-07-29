## 프로젝트 정리

### 디렉토리 구조
- `1.) ML유량예측모델.ipynb` : 포충1,2탑 ML공급유량 예측
- `2.) FL_BRIX예측모델.ipynb` : FL_BRIX 농도 예측
- `3.) 실시간 추론 작업형.ipynb` : 추론 점검용  
- `main.py` : 추론용 FastAPI

---

### 사용 환경
- Python 3.9.13
- pandas 1.5.3
- numpy 1.23.1
- scipy 1.9.1
- tensorflow 2.7.0
- scikit-learn 1.2.2
- pymongo 4.10.1
- lightgbm 3.3.5
- xgboost 1.7.4
- fastapi 0.115.12
- uvicorn 0.34.3

---

### 문제 정의
- 1.) 다음 시점(10분 이후) 3개의 제당공정 품질 예측값 제공
  
   → [원당-용해-포충-여과-정제-이중효용관-MVR] 공정에서 Target은 포충1,2탑 ML공급유량/MVR공정의 FL_BRIX 농도 


- 2.) 총 3개의 사용자 정의 목표값 or 예측 품질 대비 같거나 큰 품질을 가졌던 과거 기반 다변량 공정 입력조건 추천

---

### 주요 전처리 
  - 1.) 시계열 데이터 정의

    → ML 접근 : Target값을 1시점 뒤로 Shift

    → DL 접근 : (Batch , TimeStamp , Features) 3D 배열

    → Train/Test 순차적 분할 ( Random Shuffle X )

  - 2.) 부분적 제어 
  
    → 특정 기간 제거(Ex. 점검기간) & 필요 시 공정 시퀀스 원리에 따라 입력데이터 시차지연 적용

  - 3.) 공정조건 탐색용 데이터 구축 ( Ex. 특정 입력값들의 합산, 재배열 )
    
---

### 학습 프로세스  

   - 1.) ML/DL 점추정 & 확률적 추정 모델 적용

       → ML : Tree계열 모델 (LightGBM,XGBoost,GBM,HistGBM) Quantile Regression

     (학습데이터의 각 Target품질의 변동계수(CV) 기반 Lower/Upper Percentile 정의 )

       → DL : Monte Carlo Dropout 기반 LSTM계열 모델 ( N=100 Simulation 출력 )

     (학습데이터의 각 Target품질의 변동계수(CV) 기반 mean ± K*std에서 K 정의 )

   - 2.) 예측 모델 검증 예시

     <img width="695" height="100" alt="화면 캡처 2025-07-29 171327" src="https://github.com/user-attachments/assets/b4bcce22-d918-475d-a890-154436c55572" />

---

### 실시간 추론 프로세스 1 ( 최초 예측값 Ensemble )

  - 1.) 실시간 MongoDB 데이터 수집

       → 공정 일련 프로세스에 대해서 100개씩 조회 & 용도에 맞게 데이터 형태 정의  

  - 2.) 각 품질 예측값 KDE Ensemble & 품질 범위 정량화

       → 점추정/확률 추정 결과들의 Min/Mean/Max 특징들로 KDE 모수 지정 & 분포 생성

       → KDE분포의 변동계수(Coefficient of Variance) 기반으로 Upper/Lower Boundary 지정 

---

### 실시간 추론 프로세스 2  ( 목표값 대비 최근사 예측값 선택 )

  - 1.) 총 3개의 목표 품질 (예측 or 사용자 정의) ML공급유량1,2 & FL_BRIX 농도 정의

  - 3.) 각 KDE예측 분포에서 목표 품질 대비 크거나 같은 값 1차 선택 

  - 4.) Gamma-KDE 기반 예측값 2차 보정

       → 1차 KDE예측값과 실시간 100개 품질값의 Std를 Gamma분포의 모수로 지정 & KDE분포 생성

       → Gamma-KDE분포의 변동계수(Coefficient of Variance) 기반으로 Upper/Lower Boundary 지정

       → 각 Gamma-KDE예측 분포에서 목표 품질 대비 크거나 같은 값 2차 선택  

  - 5.) Ex.) 목표 FL_BRIX농도=76.5 / KDE예측값 = 76.401 / Gamma-KDE예측값 = 76.5

     <img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/64338589-0b0c-44d4-a568-93f511cf2b33" />


---

### 실시간 추론 프로세스 3  ( 목표값 대비 최근사 예측값 기반 최적 과거 공정조건 선택 )  

  - 1.) 공정 조건 탐색용으로 변환된 학습데이터 선택 (Recipe Data)

       → 전체 데이터(Recipe Full) & 특정 공정조건 Min/Max Filtering 적용한 데이터 정의(Recipe Info)


  - 2.) 실시간 최근(마지막 행) 입력데이터 활용 기반 Recipe Data 정보 분리 (df_origin / df_cluster)

       → Recipe Info기준 현재와 동일 원산지 정보 가진 데이터 필터링 (df_origin)

       → 원산지 + 원당 조건 수치데이터로 군집 학습(K-Means Clustering)

       → 현재 입력정보에 대한 예측 군집에 해당하는 부분 데이터 df_origin에서 선택 (df_cluster)

    (개수 기준 미달 시 현재 군집 제외 df_origin에서 최근접 군집데이터 선택 )



  - 3.) df_origin & df_cluster 기준 목표 대비 최근사 예측값 집합에 대응하는 최적 공정조건 출력

       → 목표 ML공급유량 1,2 & FL_BRIX농도에 대응하는 공정조건 탐색 최적 경로 탐색방안 구축

       → 조건탐색 실패 및 과대/과소 추정 품질을 가진 공정조건 방지를 위한 예외처리 함수 구축 

---

### 전체 프로세스 예시 

  - 1.) 실시간 100개 데이터 MongoDB 조회 시 품질들의 표준편차 & 가장 최근 품질값 / 목표 품질값 정의  

     <img width="500" height="100" alt="image" src="https://github.com/user-attachments/assets/fd4fb9a4-0fa0-4370-96f3-9b3b0d02f88b" />

  - 2.) 목표값 기반 최근사 예측값 정의 

     <img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/6393e371-eb29-4752-a36a-93df0f6be041" />

  - 3.) 목표값 기반 최근사 예측값 기반 최적 과거 공정조건 탐색

     <img width="1000" height="319" alt="image" src="https://github.com/user-attachments/assets/2719e071-d3b5-472c-addc-55d24a743e6c" />


---



  

