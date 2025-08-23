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

     <img width="250" height="150" alt="화면 캡처 2025-08-01 160424" src="https://github.com/user-attachments/assets/e7e630f5-374c-4f26-86a6-787131d5f2b4" />

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

 <img width="517" height="395" alt="화면 캡처 2025-08-23 183009" src="https://github.com/user-attachments/assets/0a02c2b2-09ed-4e3b-aca1-88d27d87cbef" />

 <img width="470" height="310" alt="화면 캡처 2025-08-23 183059" src="https://github.com/user-attachments/assets/48a4b9ef-7e46-4224-9dc6-8c569e99375a" />



---

---
## Project C-2 – 공정조건 추천 시스템 (평균 사출/냉각 시간)

### 디렉토리 구조
- `1.) K-Means학습.ipynb` : 데이터 수집 & 전처리 & 모델학습
- `2.) Optimizer_Pred.ipynb` : 추론API 점검용
  
---

### 문제 정의
- 2개 사출기의 Set_InjectionTime_Mean & Set_CoolingTime_Mean에 대해 **실시간 추천값 제공** 요.
- Production & Environment 데이터 간 시점 불일치 발생 가능 → **최근접 시간 매**칭 필요.
- 군집 기반 추천 + **예외 대응**이 가능한 모듈 필요.


---

### 주요 전처리 

- Production & Environment 데이터 매칭.
- Timestamp 직접 일치 불가 → 최근접 시점 매칭 방식 적용.
 

---
### 학습 프로세스

- Set_변수가 포함된 Production + Environment 데이터 기반 Unique 집합 추출.
- K-Means Clustering 학습 → Cluster 번호 부여.
- 학습 모델 및 Numbering 완료 데이터(**Recipe Data**) 저장


---
### 실시간 추론 프로세스  

- 1.) MongoDB에서 Production + Environment N개 실시간 수집 및 매칭.
- 2.) 각 사출기별 최근 Working_No 기반 군집 번호 예측.
- 3.) 군집 내 데이터 중 현재 Set_Injection/CoolingTime_Mean과 최근사값 선택.
- 4.) Std 기반 범위(Mean ± Std)에서 무작위 추천값 제공.
- 5.) **예외처리**
  - 군집 내 데이터 1개 → 해당 Mean Set_Injection/CoolingTime 그대로 사용.
  - 추천값이 현재값보다 낮을 경우 Gaussian Noise 보정.
  - MongoDB 수집 문제 발생 시 최근 30개 데이터 평균으로 대체.
 - 6.) 실시간 데이터를 Recipe Data에 업데이트하여 운영 지속성 확보.

  
---

### 전체 프로세스 예시 

- **실시간 데이터 수집 → 군집 기반 추천값 도출 → 범위 유연화 → 예외처리 적용 → 최종 추천값 제공.**

  <img width="350" height="300" alt="화면 캡처 2025-08-01 170838" src="https://github.com/user-attachments/assets/ee32ee33-6405-47e2-8462-fb5f58266b1a" />

---



    
