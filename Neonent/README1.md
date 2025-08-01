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
- 1.) 사출기 다수수에 대한 품질 분석 

- 2.) 사출기 점검/ 수집 / Labeling 측정 방식 문제 발생으로 기존 학습/추론방식 활용 불가 이슈

   → 기존 사출+PLC데이터 활용한 지도학습에서 사출데이터만 활용한 비지도학습 변경
 
---

### 주요 전처리 

  - 1.) 기존 입력값 처리

     → 각 사출기에서 수집된 4개 온도 + 생산과정에서 수집되는 약 1초단위 PLC시계열 데이터 통계량 결합

    <img width="500" height="300" alt="그림2" src="https://github.com/user-attachments/assets/c34ebfb3-1f15-4e84-8b87-1cabee5bbc70" />

    <img width="500" height="300" alt="그림3" src="https://github.com/user-attachments/assets/c458974a-80d5-49dd-a161-9a2889a3cc2a" />


  - 2.) 변경 입력값 처리

     → 기존 개별 1대1 라벨링기반 데이터에서 N Cavity = 1Shot(품목) 형식으로 데이터 집계

     → 각각의 N Cavity집약 기반 데이터에서 1개라도 불량유형 입력 시 불량으로 처리 

    <img width="500" height="300" alt="그림1" src="https://github.com/user-attachments/assets/d176d45f-7be9-4712-8ab0-6e59add305f3" />


---
### 학습 프로세스

  - 1.) 기존 

     → 사출 + PLC 결합데이터로 각 사출기 대상 지도학습 수행

     → Trained Normal Data 대상으로 Isolation Forest 적용 & 소수 이상 정상값 제거

     → Train/Test 대상 Undersampling & Train에 추가 OverSampling 적용으로 정상/불량 데이터 가중치 조정

     → Tree ML기반 학습모델 다수 & Stacking Classifier 구축 (그림)

     <img width="170" height="85" alt="화면 캡처 2025-08-01 160424" src="https://github.com/user-attachments/assets/e7e630f5-374c-4f26-86a6-787131d5f2b4" />


  - 2.) 변경 

     → 불량데이터 정의 모호 & 수집개수 지나치게 부족 현상 발생 (Ex.100,000개중 100개미만 or X)

     → 각 사출기 대상 정상데이터(Cavity불량률=0.0%)로 기본 AutoEncoder모델 학습 (Early Stopping 적용& Best Weight 저장)

    <img width="500" height="300" alt="다운로드" src="https://github.com/user-attachments/assets/b087fafc-9255-48d8-a30b-c35993634435" />


---
### 실시간 추론 프로세스  

   - 1.) 실시간 MongoDB 수집

       → 각 사출기당 N개 데이터 실시간 수집 & N Cavity = 1Shot형식으로 변경 X상태로 그대로 개별 예측 방식 적용 
     
   - 2.) 기본 예측 (Fixed Prediction)

       → 실시간 특정 사출기의 1개 Unique_Num_Cavity(Working_No)에 해당하는 데이터는 AE의 고정임계값 기반 예측 

   - 3.) Distribution Adaptable Prediction

       → 실시간 특정 사출기에서 N개 데이터 수집 & AE모델 예측기반 test_mae_loss 집합 계산

       → Test_Mae_loss집합의 KDE분포 생성 & KDE기반 Quantile기준으로 불량예측용 임계값 계산

       → Test_Mae_loss집합의 mae_loss/anomaly여부/pred결과/threshold 통합 정보 출력

       → Test_Mae_Loss 중 마지막 20개 & Unique_Num_Cavity(Working_No)에 해당하는 예측 결과 출력 
       
   - 4.) Hybrid Anomaly Detection System 구축

       → N개 Test_mae_loss에 대한 KDE분포의 안정성 Custom 계산 (CV,Skew,Kurtosis 종합)

       → Stable=False시 Fixed Prediction 단독 / Stable=True시 Fixed & Adaptable 혼용 기반 품질 예측 

       → Fixed=Normal / Adaptable=False 결과 발생 시 마지막 20개 데이터 예측 결과의 정상비율 기반으로 최종 품질 예측
        
   - 5.) Recheck Anomalies Process
     
       → 각 사출기당 실시간 N개 수지된 데이터의 Test_MAE_Loss 분포의 공정능력지수(CP,CPK) 계산

       → Pred Normals의 mae_loss값들을 기반으로 Pred Faults mae_loss값들의 Robust Z Score 계산

       → Pred Faults의 mae loss값들과 Dist Adaptable Pred과정에서의 KDE-Quantile임계값 차이 계산

       → 총 계산 통계량들로 불량예측이 발생한 데이터들에 대한 예측 결과 수정 검토 (Ex. 임계값을 약간 초과한 불량결과인지)

   - 6.) 작업시간 예외처리 적용 
    
       → 작업자가 직접 사출기 가동하는 시간은 분석 제외 (이때 모든 데이터는 Default=Normal로 취급)
        
---

### 전체 프로세스 예시 

   - 1.) 실시간 N개 데이터 수집 & 최근 UniqueNum Key값에 대해서 예측 지속

   - 2.) Prediction Case 예시 제공 

   <img width="578" height="74" alt="화면 캡처 2025-08-01 160940" src="https://github.com/user-attachments/assets/65d47c72-a8ec-417a-bf1b-d1cff7e9587b" />

   <img width="263" height="194" alt="화면 캡처 2025-08-01 160842" src="https://github.com/user-attachments/assets/6f3915d6-f646-4916-8f03-87b5cfa3e2e4" />

   <img width="647" height="355" alt="화면 캡처 2025-08-01 160909" src="https://github.com/user-attachments/assets/61cc01c9-c673-46bf-a7b5-6674f5b94556" />

   <img width="578" height="79" alt="화면 캡처 2025-08-01 161349" src="https://github.com/user-attachments/assets/c399504f-4b9b-489b-a52b-179654b35485" />

   <img width="263" height="201" alt="화면 캡처 2025-08-01 161419" src="https://github.com/user-attachments/assets/348deb92-6ef0-42b0-8f58-32c6722d2391" />

   <img width="542" height="351" alt="화면 캡처 2025-08-01 161456" src="https://github.com/user-attachments/assets/ac291c8a-995e-47db-a6b2-fe55ec307e99" />






 
    
