## 프로젝트 정리

### 디렉토리 구조
- `1.) 데이터수집_postgre.ipynb` : 데이터 수집용 쥬피터 노트북
- `2.) 학습데이터_구축_postgre.ipynb` : 학습용 데이터 정제
- `3.) 학습모델구축.ipynb` : 모델링 및 학습 
- `4.) 임시라벨링_데이터점검.ipynb` : 학습모델 기반 라벨링 및 검토
- `5.) RecipeAI.ipynb` : 최적 생성조건 데이터 생성
- `prediction_점검용.ipynb` : 추론 점검용
- `prediction.py` : 추론 API 탑재용 py
- `memae.py` : Memory Augmented AutoEncoder Custom 함수
- `main.py` : 추론용 FastAPI
- `predictionAPI.bat` : 실시간 추론 FastAPI BAT

---

### 사용 환경
- Python 3.9.21
- pandas 2.2.3
- numpy 1.23.1
- scipy 1.10.1
- tensorflow 2.7.0
- scikit-learn 1.2.2
- pymongo 4.10.1
- psycopg2 2.9.10
- fastapi 0.115.12
- uvicorn 0.34.3

---

### 문제 정의
- 1.) Postgre Raw 사출 생산데이터의 1대1 라벨링 부재
  
   → 데이터의 1일 Lot단위 생산 특징
  
   → 1일 단위 총 생산량 (정상 + 불량품질 총합 개수만 파악 가능)
  
   → 개별 데이터의 Good/Bad Qty의 수치는 총생산량 기반으로 기록 (개별 데이터의 품질 의미 X)

- 2.) Setting 기반 데이터 분포 조합에 따라서 분포의 차이를 나타냄

   → 각 maker당 동일 설비/품목은 여러 날짜(Lot)별로 1개 혹은 다수의 Setting에 의해 생산 진행
  
---

### 주요 전처리 
  - 1.) 설비/품목/1일 Lot생산단위 불량률 계산

     → 1대1 라벨링이 아닌 생산다위별로 정상/불량 정보 거시적으로 파악하여 최소한 학습가능한 데이터 선별 
    
    <img width="700" height="250" alt="그림1" src="https://github.com/user-attachments/assets/137fb13a-177d-4add-8443-98f7abe4f377" />

  - 2.) Setting 데이터 구분 프로세스 구축 

      → 다변량 Setting데이터의 Unique조합 계산 (Drop Duplicates)
    
      → K-Means 학습으로 고유 Setting 조합에 대한 전체 학습데이터 Cluster Numbering 부여

      → 3개의 maker에서 각 Facility-Item-Setting에 대한 정보 구분

    <img width="296" height="326" alt="화면 캡처 2025-07-28 131912" src="https://github.com/user-attachments/assets/bf60ecc7-8386-4bb4-aa06-07dd19738e8e" />

---

### 학습 프로세스  
   - 1.) Lot단위 불량률 & 고유 Setting 정보 정량화 기반 학습데이터 완성

       → 총불량률 <1.0%인 [Facility-Item-Setting] 데이터 학습 / 총불량률 >=1.0 검증
     
       → Lot 불량률 0.0% --> 1.0% 기준 완화로 정상품질에 대한 과적합 사전 방지 & 학습 유연성 부여

       → Lot 불량률 1.0% 미만인 Facility-Item-Setting정보 구분된 데이터 중 500개 미만은 학습에서 제거

     <img width="1918" height="475" alt="그림3" src="https://github.com/user-attachments/assets/b9ad7c9d-e64c-4d7f-ba4e-d368b5caf8a0" />

   - 2.) 학습 데이터 개수에 따른 학습모델 선택

       → IsolationForest, AutoEncoder, Memory Augmented AutoEncoder ( AE계열 Early Stopping & Best Weight 활용 )

       → Ex.) 특정 (설비/품목/Setting Cluster)의 분리학습 

       <img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/52b96386-97fd-439a-9f30-c39a8ad1529c" />

---

### 실시간 추론 프로세스  

  - 1.) 실시간 MongoDB 수집 데이터 품질 검토

       → 결측 정보(Ex. 품목 , Production, Setting) Forward/Backward/Mean/Mode Fill

       → 정보 결함으로 인한 예측 불가 예외처리 사전전방지 


  - 2.) 기본 추론 ( 학습이력이 있는 정보 )

       → 현재 Setting의 Cluster예측 & 해당 Cluster가 속한 학습된 Facility/Item정보 활용
    
       → 해당 학습모델 업로드

       → 학습된 mae loss 정보 기준으로 MinMaxScaler(clip=True)로 예측용 데이터 정규화값 발산 방지 

       → Trained Reconstruction MAE Loss 분포의 변동계수(Coefficient of Variance)로 임계값에 가중치 & Margin Limit 부여
    ( 과적합 방지 & 추론 유연성 부여 )

    <img width="343" height="150" alt="화면 캡처 2025-07-28 135612" src="https://github.com/user-attachments/assets/2d0c0c99-271e-4665-8bee-0b73e0742804" />


  - 3.) 적응형 추론 ( 학습이력이 없거나 기존 학습불가 Setting Cluster 정보 )

       → 해당 Maker 전체 혹은 동일 Facility/Item 학습정보 중 현재 Setting조합과 최근접 정보 탐색 ( Euclidean Distance)

       → 최근접 정보 기반 대체 추론용 학습모델로 현재 실시간 데이터 품질 추론 

       → 학습된 mae loss 정보 기준으로 MinMaxScaler(clip=True)로 예측용 데이터 정규화값 발산 방지

  - 4.) 추론 임계값 유연성 부어 

       → 기존 학습 당시 고정 임계값 & trained_mae_loss의 Max 중 최대값 선택

       → Trained_MAE_Loss의 변동계수(CV) 기반 기본 가중치 계산

       → 적응형 추론 단계에서는 최근접 Setting 탐색 Euclidean Distance 추가 가중치 계산 

       → 가중치 정보 기반으로 추론 단계에서 Threshold & Current test_mae_loss간 Margin 계산으로 불량예측 허용오차 계산


     <img width="343" height="150" alt="화면 캡처 2025-07-28 135636" src="https://github.com/user-attachments/assets/d5b4d90f-fbd6-41b6-a3e9-3effbe15f910" />


     <img width="343" height="150" alt="화면 캡처 2025-07-31 192107" src="https://github.com/user-attachments/assets/788e8d9c-435c-4cf7-a0c6-433cbdcebc5d" />

---

### 최적 생산 Setting조건 제공용 데이터 구축   

  - 1.) 기존 학습모델의 임계값 추론 기반으로 학습데이터 임시라벨링 부여 (예측 기반 라벨링)

  - 2.) 각 설비 & 품목 생산정보 중 최다 생산 이력 / 최소 예측불량률 이력을 가진 Cluster Setting정보 정리

  - 3.) Unique Setting 정보 & 해당 Setting으로 생산한 Production Data들의 Min/Mean/Max 제공

  - 4.) 모델링 기반 과거학습데이터의 품질 라벨링을 토대로 추후 생산계획 참고 제안 


    <img width="1599" height="455" alt="그림4" src="https://github.com/user-attachments/assets/18c411e5-9e66-4d15-b073-d22f9d167a46" />

### 주요 성과 

   - 1.) AI시스템 완성 이후 실제 불량생산 정확도 약 20% 향상
     
   - 2.) 예측기반 최적 생산 참고용 데이터 가이던스 활용 시 불량 생산 약 5% 감소 



