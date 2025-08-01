## 프로젝트 정리

### 디렉토리 구조
- `1.) 통합 예측분석.ipynb` : 데이터 수집 / 전처리/ 모델링 구축
- `2.) Prediction_API.ipynb` : 추론용 API 작업형 
---

### 사용 환경
- Python 3.9.13
- pandas 2.0.3
- numpy 1.23.1
- scipy 1.9.1
- scikit-learn 1.2.2
- pymongo 4.3.3
- xgboost version: 1.7.4
- lightgbm version: 3.3.5
- fastapi 0.115.12
- uvicorn 0.34.3

---

### 문제 정의
- 1.) 특정 사출기 품질 예측
---

### 주요 전처리 
  - 1.) 불필요 변수 제거 & 입력변수 정의 

  - 2.) 학습데이터 기반 변수선택법 적용

     → Welch's T-Test 기준 변수 선택 

---

### 학습 프로세스  
   - 1.) 정상/불량 라벨링된 Train_Data 기반 변수선택 

       → Welch's T-Test기반 다변량 입력변수 필터링 
     
   - 2.) 교차검증용 데이터 구축
     
       → Train/Test Split 매번 다르게 5회 적용

       → 매번 다른 Train Data 기반 변수선택 개별 적용 

   - 3.) Tree기반 ML Classifier 모델 정의


   - 5.) 특정 학습모델 기반 교차검증 실험 진행

       → Test 1~5까지 매번 다른 입력변수 기반 학습모델로 예측 수행 

     <img width="273" height="119" alt="화면 캡처 2025-08-01 181019" src="https://github.com/user-attachments/assets/0a4b7d0f-f7ef-4016-9e25-1f75f0ed8878" />

   - 6.) 추론용 API 작성 (시스템 가동용 X)
     

---



    
