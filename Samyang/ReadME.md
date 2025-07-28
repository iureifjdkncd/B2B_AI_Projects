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
  
   → 원당-용해-포충-여과-정제-이중효용관-MVR 공정 중에서 Target은 포충1,2탑 ML공급유량 & MVR공정의 FL_BRIX 농도로 정의


- 2.) 총 3개의 사용자 정의 목표값 or 예측 품질 대비 같거나 큰 품질에 대응하는 다변량 공정 입력조건 추천

     
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

    예측 결과 예시 추가 
---

### 실시간 추론 프로세스  

  - 1.) 실시간 MongoDB 데이터 수집

       → 공정 일련 프로세스에 대해서 100개씩 조회 

  - 2.) 각 품질 예측값 KDE Ensemble & 품질 범위 정량화  

  - 3.) 목표 품질 (예측 or 사용자 정의 기반) 1차 KDE 예측값 정의

  - 4.) Gamma-KDE 기반 예측값 2차 보정

      - 분포 그림 1,2,3 첨부 

---

### 목표값 기반 다변량 공정조건 탐색 프로세스  

  - 1.) 실시간 100개 중 마지막 데이터를 기준으로 선택 

  - 2.) 공정조건 탐색용 데이터 업로드 

  - 1.) 원산지 + 원당조건 필터링 

  - 1.) 경로탐색 & 예외처리 

---



  

