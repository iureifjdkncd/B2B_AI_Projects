## 프로젝트 정리 B

### 디렉토리 구조
- `1.) K-Means학습.ipynb` : 데이터 수집 & 전처리 & 모델학습
- `2.) Optimizer_Pred.ipynb` : 추론API 점검용 

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
- 1.) 2개 사출기의 Set_InjectionTime & CoolingTime 실시간 추천 

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

       → 실시간 마지막 데이터는 기존 학습데이터에 지속 업데이트 적용 


   - 3.) Set Injection/CoolingTime_mean 추천값 제공 

       → 부분집합 개수 다수일 경우 현재 실시간 데이터의 Injection/CoolingTime_mean과 최근사값 선택 

       → Set_Injection/CoolingTime_Std값을 기반으로 Injection/CoolingTime_Mean ± Std 범위 계산

       → Injection/CoolingTime_Mean 범위 중 무작위값 선택 최종 완료


   - 4.) 추천값 예외처리 적용 1 

       → 군집기반 부분집합 개수 1개일 경우 해당 Injection/CoolingTime +Mean값 출력 & std기반 범위 적용 X 

   - 5.) 추천값 예외처리 적용 2

       → 최종 추천값 Injection/CoolingTime_Mean값이 현재 실시간 데이터와 같거나 크기 비교 기준 미달 발생 경우

       → Injection/CoolingTime_Mean값에 Gaussian Noise 추가 적용

   - 6.) 추천값 예외처리 적용 3

       → 실시간 MongoDB수집 문제로 인한 Production & Environment 매칭 오류 / Working_No조회 불가  

       → 실시간 30개 Production데이터의 Set_Injection/CoolingTime Mean값들의 평균으로 추천값 대체 
        
    
---

### 전체 프로세스 예시 

   - 1.) 실시간 N개 데이터 수집 중 최근 Working_No에 대한 과거조건 기반 CoolingTime/InjectionTime 추천

      <img width="400" height="300" alt="화면 캡처 2025-08-01 170838" src="https://github.com/user-attachments/assets/ee32ee33-6405-47e2-8462-fb5f58266b1a" />

---
