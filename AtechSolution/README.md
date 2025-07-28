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
     
---

### 주요 전처리 
  - 1.) 설비/품목/1일 Lot생산단위 불량률 계산
    
    <img width="1000" height="500" alt="그림1" src="https://github.com/user-attachments/assets/137fb13a-177d-4add-8443-98f7abe4f377" />

  - 2.) Setting 데이터 구분

      → 다변량 Setting데이터의 Unique조합 계산 (Drop Duplicates)
    
      → K-Means Clustering을 통한 고유 Setting 조합에 대한 Numbering 부여

      → 동일 설비/품목은 여러 날짜(Lot)별로 1개 혹은 다수의 Setting에 의해 생산 진행

    <img width="296" height="326" alt="화면 캡처 2025-07-28 131912" src="https://github.com/user-attachments/assets/bf60ecc7-8386-4bb4-aa06-07dd19738e8e" />

  - 3.) Lot단위 불량률 & 고유 Setting 정보 정량화 기반 학습데이터 완성 

     <img width="1918" height="475" alt="그림3" src="https://github.com/user-attachments/assets/b9ad7c9d-e64c-4d7f-ba4e-d368b5caf8a0" />


---

### 학습 프로세스  

   - 1.) 총불량률 <1.0%인 [Facility-Item-Setting] 데이터 학습 / 총불량률 >=1.0 검증

       → Lot 불량률 0.0% --> 1.0% 기준 완화로 정상품질에 대한 과적합 사전 방지 & 학습 유연성 부여 

   - 2.) 학습 데이터 개수에 따른 학습모델 선택

       → IsolationForest, AutoEncoder, Memory Augmented AutoEncoder

       → Ex.) 특정 (설비/품목/Setting Cluster)의 분리학습 

       <img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/52b96386-97fd-439a-9f30-c39a8ad1529c" />

---

### 실시간 추론 프로세스  

  - 1.) 실시간 MongoDB 수집 데이터 품질 검토

       → 결측 정보(Ex. 품목 , Production, Setting) Forward/Backward/Mean/Mode Fill

       → 정보 결함으로 인한 예외처리 방지 


  - 2.) 기본 추론 ( 학습이력이 있는 설비/품목의 정보 )

       → 학습모델 업로드

       → Trained Reconstruction MAE Loss 분포의 변동계수(Coefficient of Variance)로 임계값에 가중치 & Margin Limit 부여 ( 과적합 방지 & 추론 유연성 부여 )

    <img width="343" height="150" alt="화면 캡처 2025-07-28 135612" src="https://github.com/user-attachments/assets/2d0c0c99-271e-4665-8bee-0b73e0742804" />


  - 3.) 적응형 추론 ( 학습이력이 없는 설비/품목정보 )

       → 학습했던 다변량 Setting 조합 Dictionary중 현재 수집데이터의 Setting조합과 최근접 정보 탐색 ( Euclidean Distance)

       → 최근접 정보 기반 대체 추론용 학습모델 선택

       → Trained Reconstruction MAE Loss 분포의 변동계수(Coefficient of Variance) & Setting Euclidean Distance로 임계값에 가중치 & Margin Limit 부여 ( 과적합 방지 & 추론 유연성 부여 )

     <img width="343" height="150" alt="화면 캡처 2025-07-28 135636" src="https://github.com/user-attachments/assets/d5b4d90f-fbd6-41b6-a3e9-3effbe15f910" />

---

### 최적 생산 Setting조건 제공용 데이터 구축   

  - 1.) 기존 학습모델의 임계값 추론 기반으로 학습데이터 임시라벨링 부여 (예측 기반 라벨링)

  - 2.) 각 설비 & 품목 생산정보 중 최다 생산 이력 / 최소 예측불량률 이력을 가진 Cluster Setting정보 정리

  - 3.) Unique Setting 정보 & 해당 Setting으로 생산한 Production Data들의 Min/Mean/Max 제공


    <img width="343" height="269" alt="그림4" src="https://github.com/user-attachments/assets/5b042c74-0732-41f2-8c70-b8fc7dd56a1f" />





