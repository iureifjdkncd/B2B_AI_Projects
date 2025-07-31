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
   - 1측

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




 
    
