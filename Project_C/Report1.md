## Project C-1 – 사출 설비 품질 예측 시스템
(N사 자동차 부품 제조기업 / K-등대공장 구축사업 / 2023.05 ~ 2025.03)

--- 

### 디렉토리 구조
#### Injection_Prediction(Old)
- `1.) 사출데이터_전처리(Old).ipynb` : 데이터 수집 & 전처리
- `2.) 예측모델_생성(Old).ipynb` : 수집데이터 학습
- `3.) 추론코드(Old).ipynb` : 추론 점검

#### Injection_Prediction(New)
- `1.) 사출데이터 전처리_비지도학습(New).ipynb` : 데이터수집/전처리/학습
- `2.) 사출_작업형_추론코드작성(New).ipynb` : 추론코드 점검용 
- `main.py : FastAPI 기반 추론 서버
- `prediction.py : 추론 API 모듈
- 'neonentPredictionAPI : 실시간 추론 서버 실행 스크립트
---

### 사용 환경
- Python 3.9.13 / pandas 1.5.3 / numpy 1.23.1 / scipy 1.9.1 / tensorflow 2.7.0 / scikit-learn 1.2.2 / lightgbm 3.3.5 / xgboost 1.7.4 / pymongo 4.8.0 / fastapi 0.115.12 / uvicorn 0.34.3


---

### 문제 정의
- 1.) **다수 사출기에 대한 실시간 품질 분석 필요**
  - 다수 설비에서 동시에 수집되는 시계열 데이터를 기반으로 정상/불량 판정 수행.
- 2.) **Label 부족 및 수집/측정 방식 문제**
  - 라벨링 적합도 부족, 불량 데이터 희소성으로 지도학습 기반 추론 불가.
  - 기존: 사출+PLC 결합 데이터 → 지도학습
  - 개선: 사출 데이터 단독 기반 → **비지도 AutoEncoder 학습**

 
---

### 주요 전처리 

- 1.) **기존 입력 처리**
  - 각 사출기에서 수집된 4개 온도 변수 + 1초 단위 PLC 시계열 데이터 통계량 결합

    <img width="500" height="300" alt="그림2" src="https://github.com/user-attachments/assets/c34ebfb3-1f15-4e84-8b87-1cabee5bbc70" />

    <img width="500" height="300" alt="그림3" src="https://github.com/user-attachments/assets/c458974a-80d5-49dd-a161-9a2889a3cc2a" />


- 2.) **변경 입력 처리**
  - 기존 1:1 라벨링 대신, **N Cavity = 1Shot** 단위 집계
  - Cavity 집약 데이터 중 하나라도 불량 유형 발생 시 전체를 불량으로 정의

    <img width="500" height="300" alt="그림1" src="https://github.com/user-attachments/assets/d176d45f-7be9-4712-8ab0-6e59add305f3" />


---
### 학습 프로세스

- 1.) **기존 지도학습 방식**
  - 사출 + PLC 결합데이터 → RF, XGB, LGBM 등 Tree ML 기반 분류.
  - Isolation Forest로 소수 이상치 제거.
  - **Under/OverSampling**으로 클래스 불균형 보정.
  - Tree ML 기반 단일/Stacking Classifier 구축.

     <img width="170" height="85" alt="화면 캡처 2025-08-01 160424" src="https://github.com/user-attachments/assets/e7e630f5-374c-4f26-86a6-787131d5f2b4" />

- 2.) **비지도 방식(변경 후)**
  - 불량 정의의 모호성과 데이터 희소성 문제 → 지도학습 한계.
  - 각 사출기별 정상 데이터(Cavity 불량률 = 0.0%)만으로 AutoEncoder 학습.
  - Early Stopping + Best Weight 저장으로 학습 안정성 확보.

    <img width="500" height="300" alt="다운로드" src="https://github.com/user-attachments/assets/b087fafc-9255-48d8-a30b-c35993634435" />


---
### 실시간 추론 프로세스  

- 1.) **데이터 수집**
  - MongoDB에서 실시간 N개 데이터 수집 → **N Cavity = 1Shot 구조 변경 없이 그대로 입력.**
- 2.) **Fixed Prediction**
  - AE의 고정 임계값 기반 실시간 단일 샘플 예측.
- 3.) **Distribution Adaptable Prediction**
  - 실시간 N개 데이터에 대한 AE 예측 MAE Loss 집합 생성.
  - KDE 기반 분포화 → Quantile 기준 임계값 계산.
  - MAE Loss·예측 라벨·Threshold 등 통합 정보 출력.
- 4.) **Hybrid Anomaly Detection**
  - 분포 안정성(CV, Skew, Kurtosis) 평가.
  - 안정적이면 Fixed + Adaptable 혼합, 불안정하면 Fixed 단독.
  - Fixed=Normal & Adaptable=Fault 충돌 시 → 최근 20개 MAE Loss의 Normal 비율로 최종 판정.
- 5.) **Recheck Process**
  - CP/CPK, Robust Z-Score, Threshold Gap 기반 Fault 예측 데이터 재검토.
  - 임계값을 소폭 초과한 케이스에 대해 불필요한 **False Alarm 최소화.**
- 6.) **작업시간 예외 처리**
  - 작업자 직접 제어 시간대 데이터는 Default=Normal로 취급.
  
---
### 전체 프로세스 예시 

- **실시간 N개 데이터 수집 → 단일/분포 기반 예측 → Hybrid 판정 → Recheck → 최종 품질 판정**

   <img width="578" height="100" alt="화면 캡처 2025-08-01 160940" src="https://github.com/user-attachments/assets/65d47c72-a8ec-417a-bf1b-d1cff7e9587b" />
   

   <img width="263" height="194" alt="화면 캡처 2025-08-01 160842" src="https://github.com/user-attachments/assets/6f3915d6-f646-4916-8f03-87b5cfa3e2e4" />
   

   <img width="647" height="355" alt="화면 캡처 2025-08-01 160909" src="https://github.com/user-attachments/assets/61cc01c9-c673-46bf-a7b5-6674f5b94556" />
   

   <img width="578" height="100" alt="화면 캡처 2025-08-01 161349" src="https://github.com/user-attachments/assets/c399504f-4b9b-489b-a52b-179654b35485" />
   

   <img width="263" height="194" alt="화면 캡처 2025-08-01 161419" src="https://github.com/user-attachments/assets/348deb92-6ef0-42b0-8f58-32c6722d2391" />
   

   <img width="647" height="355" alt="화면 캡처 2025-08-01 161456" src="https://github.com/user-attachments/assets/ac291c8a-995e-47db-a6b2-fe55ec307e99" />

---

---
## C-2. 공정조건 추천 시스템 (평균 사출/냉각 시간)

### 디렉토리 구조
- `1.) K-Means학습.ipynb` : 데이터 수집 & 전처리 & 모델학습
- `2.) Optimizer_Pred.ipynb` : 추론API 점검용
  
---

### 문제 정의
- 1.) 2개 사출기의 Set_InjectionTime_Mean & Set_CoolingTime_Mean 실시간 추천 

---

### 주요 전처리 

  - 1.) 데이터 매칭 

     → Production & Environment 수집

     → TimeStamp 1대1 매칭 안되는 관계로 최근접 시간으로 매칭 시도 

---
### 학습 프로세스

  - 1.) 매칭데이터 군집학습  

     → Set_포함된 Production + Environment 수치형 입력변수 대상으로 Unique 집합 선별

     → K-Means Clustering 모델학습 이후 기존 학습데이터 전체에 Cluster Numbering 정보 부여

     → 학습모델 & Numbering 완료된 학습데이터 저장 

---
### 실시간 추론 프로세스  

   - 1.) 실시간 MongoDB 수집

       → Production + Environment N개 실시간 수집 & 최근접 시간 매칭 완료

     
   - 2.) K-Means 실시간 데이터 예측 

       → 각 사출기별 최근 Working_No 대상으로 Set_포함된 Production + Environment 수치형 정보의 군집번호 예측

       → 학습데이터에서 해당 군집번호에 속하는 부분집합 선택

       → 실시간 마지막 데이터는 기존 학습데이터(Recipe Data)에 지속 업데이트 적용 


   - 3.) 실시간 대비 Set Injection/CoolingTime_mean 추천값 제공 

       → 부분집합 개수 다수일 경우 현재 실시간 데이터의 Set_Injection/CoolingTime_mean과 최근사값 선택 

       → Set_Injection/CoolingTime_Std 변수 기반으로 Set_Injection/CoolingTime_Mean ± Std 범위 계산

       → Set_Injection/CoolingTime_Mean 범위 중 무작위값 선택 최종 완료


   - 4.) 추천값 예외처리 적용 1 

       → 군집기반 부분집합 개수 1개일 경우 해당 Set_Injection/CoolingTime +Mean값 그대로 출력

   - 5.) 추천값 예외처리 적용 2

       → 최종 추천값 Set_Injection/CoolingTime_Mean값이 현재 실시간 데이터와 같거나 크기 비교 기준 미달 발생 경우

     : Set_Injection/CoolingTime_Mean값에 Gaussian Noise 추가 보정
     

   - 6.) 추천값 예외처리 적용 3

       → 실시간 MongoDB수집 문제로 인한 Production & Environment 매칭 오류 / Working_No조회 불가  

       → 실시간 30개 Production데이터의 Set_Injection/CoolingTime Mean값들의 평균으로 추천값 대체 
        
    
---

### 전체 프로세스 예시 

   - 1.) 실시간 N개 데이터 수집 중 최근 Working_No에 대한 과거조건 기반 CoolingTime/InjectionTime 추천

   - 2.) 현재 Cooling/Injection_Mean Setting값들과 K-Means기반 과거공정조건과 차이 검토 & 생산 참고 

      <img width="400" height="300" alt="화면 캡처 2025-08-01 170838" src="https://github.com/user-attachments/assets/ee32ee33-6405-47e2-8462-fb5f58266b1a" />

---



    
