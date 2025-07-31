## 프로젝트 정리 A 

### 디렉토리 구조
#### Injection_Prediction(Old)
- `1.) 사출데이터_전처리(Old).ipynb` : 데이터 수집 & 전처리
- `2.) 예측모델_생성(Old).ipynb` : 수집데이터 학습
- `3.) 추론코드(Old).ipynb` : 추론코드 점검용

#### Injection_Prediction(New)
- `1.) 사출데이터 전처리_비지도학습(New).ipynb` : 데이터수집/전처리/학습
- `2.) 사출_작업형_추론코드작성(New).ipynb` : 추론코드 점검용 
- `main.py : 추론용 FastAPI
- `prediction.py : 추론용 API 탑재용 py
- 'neonentPredictionAPI : 실시간 추론 FastAPI BAT
---

### 사용 환경
- Python 3.9.13
- pandas 1.5.3
- numpy 1.23.1
- scipy 1.9.1
- tensorflow 2.7.0
- scikit-learn 1.2.2
- lightgbm 3.3.5
- xgboost 1.7.4
- pymongo 4.8.0
- fastapi 0.115.12
- uvicorn 0.34.3

---

### 문제 정의
- 1.) 사출기 6대에 대한 품질 분석 

- 2.) 사출기 점검/ 수집 / Labeling 측정 방식 문제 발생으로 기존 학습/추론방식 활용 불가 이슈

   → 기존 사출+PLC데이터 활용 지도학습에서 사출데이터만 활용한 비지도학습 변경 
  
---

### 주요 전처리 

  - 1.) 기존 입력값 처리

     → 각 사출기에서 수집된 4개 온도 + 생산과정에서 수집되는 약 1초단위 PLC시계열 데이터 통계량 결합 (ppt 그림 추가)

  - 2.) 변경 입력값 처리

     → 기존 개별 1대1 라벨링기반 데이터에서 N Cavity = 1Shot(품목) 형식으로 데이터 집계

     → 각각의 N Cavity집약 기반 데이터에서 1개라도 불량유형 입력 시 불량으로 처리 (ppt 그림 추가)

---

### 학습 프로세스  
   - 1.) 지도학습 모델 구축 (기존)

       → 각 사출기마다 사출+PLC데이터 결합
     
       → Train Normal Data에서 IsolationForest 활용한 소수 이상치 제거 

       → Train/Test 각각 UnderSampling & 이후 Train OverSampling 진행으로 정상/불량 특징 가중치 비율 조정

       → Tree기반 ML모델 및 Stacking Classifier 구축 (예측 그림 추가)


   - 2.) 비지도학습 모델 구축 (변경)

       → 데이터 수집 방식 & Labeling 측정 문제 발생 ( Ex. 데이터 수집 10,000개당 불량 100개 미만 or 입력 X)

       → N Cavity= 1 Shot 형식 변환 이후 Normal데이터(N Cavity 불량률=0.0%) 대상으로 Basic AutoEncoder 적용 (예측 그림 추가)

      
---

### 실시간 추론 프로세스  

  - 1.) 실시간 MongoDB 수집 데이터 수집

       → 추론 단계에서는 변경된 수집방식인 Cavity단위 데이터 그대로 수집 (N Cavity = 1Shot 변경 X)

 
  - 2.) 기본 추론 ( Fixed Prediction )

       → 각 사출기당 실시간 1개의 Unique_Num(Working_No) 수집
    
       → 해당 학습모델 업로드

       → 학습된 mae loss 정보 기준으로 MinMaxScaler(clip=True)로 예측용 데이터 정규화값 발산 방지 


  - 3.) Distribution Adaptable Prediction

       → 각 사출기당 실시간 N개 데이터 수집 & NxD데이터 자체 MinMaxScaling 진행 

       → 해당 학습모델로 실시간 N개 데이터 예측 & Test_MAE_Loss 값 계산

       → Test_MAE_Loss값 기반 KDE분포 생성/임계값 Quantile 재계산으로 N개 데이터 개별 품질 예측 수행

       → 마지막 20개 데이터 대상 & 현재 입력대상 Unique_Num(Working_No)에 해당하는 데이터 품질 예측 결과 출력


   - 4.) Hybrid Anomaly Detection System

   - 5.) Recheck Anomalies Process 

 
    
