## 📌 B2B Projects

### Project A. 비지도 학습 기반 사출 품질 예측 및 최적 세팅 추천 시스템
(자율형 공장 구축 사업 / 2023.08 ~ 2025.07)
- 핵심 알고리즘: K-Means, IsolationForest, AutoEncoder, Memory-Augmented AutoEncoder
- 주요 결과
  - 동일 설비·품목 내 Setting별 데이터 분포를 K-Means로 분리 → 독립 학습군 구성
  - 비지도 학습 기반 불량 탐지 정확도: **실험 92% → 현장 73%**
  - 학습 이력 유무 판별 → 유사 학습 모델 기반 대체 추론 + 임계값 가중치 적용 → **불량탐지 정확도 평균 16% 향상**
  - 예측 기반 라벨링으로 품질 통계·최적 조건 도출 → **생산 불량률 약 5% 감소**

--- 
### Project B. 확률적 시계열 예측 기반 제당 공정 품질 추론 및 공정조건 최적화 시스템
(AI 바우처 사업 / 2024.05 ~ 2025.03)
- 핵심 알고리즘: Tree ML Quantile Regression(XGBoost, LightGBM), Monte Carlo Dropout, BiLSTM/GRU
- 주요 결과
  - 확률적 예측 적용 → 실험 정확도 95% 이상, **현장 적용 87~91% 유지**
  - Ensemble & KDE 기반 분포 정량화 → 예측력 집중 향상 기여
  - Gamma-KDE + Custom 경로탐색 통합 → 목표값 기반 **공정조건 추천 일치율 85~90%** (vs 작업자 Rule-Based)
  - **설탕 품질(FL_BRIX 농도) 약 1% 개선**, 실시간 데이터 누적을 통한 **지속 운영 가능성 확보**


---

### Project C. K-등대공장 구축 사업
(2023.05 ~ 2025.03)

#### [1] 사출 설비 품질 예측 시스템
- 핵심 알고리즘: Tree ML Classifier(XGBoost, LightGBM), AutoEncoder
- 주요 결과
  - 지도학습 기반 불량 탐지 정확도: 실험 85~91% → **현장 65~68%**
  - 라벨링 한계 극복 → 데이터 재정의 + 비지도 학습 대체 적용
  - **AutoEncoder**기반 실시간 단일·다중 입력 대응형 **혼합 체계** 구축 → **불량탐지 정확도 14% 향상**

#### [2] 평균 냉각/사출시간 공정조건 추천 모듈
- 핵심 알고리즘: K-Means Clustering
- 주요 결과
  - 실시간 공정 입력 → **K-Means** 군집 기반 CoolingTime/InjectionTime 추천
  - 조건별 표준편차 + Gaussian Noise 적용 → **추천값 유연성 확보**
  - 실제 vs 추천 공정조건 **일치도 차이 ±3~5% 이내 유지**
  - 실시간 데이터 누적 → **지속 가능성 강화**


---

#### Project D: 사출설비 불량 예측 및 TTA 인증 / 수행기간: 2023.08 ~ 2023.09
- 핵심 알고리즘: Tree ML Classifier (e.g., XGBoost, LightGBM)
- 주요 결과
  
 - 학습데이터 기반 변수 선택 및 교차검증을 통한 최적 Tree ML모델 구성 → F1 점수 95% 이상 달성
 
 - 특정 사출설비 대상 불량 예측 코드 개발 및 모델 성능 검증 완료
 
 - TTA 인증을 위한 품질 예측 기능 개발 및 적용 준비 완료
 

---

#### Project E: Cogging Motor 기반 조립·병렬 공정 불량 탐지 POC / 수행기간: 2024.07 ~ 2024.08
- 핵심 알고리즘: Tree 기반 ML Classifier (e.g., XGBoost, LightGBM)
- 주요 결과
  
 - Tree ML기반 분류 모델을 활용해 조립·병렬 공정의 불량 탐지 AI 모델 개발
 
 - 공정별 AI 적용 가능성 검토: 일부 공정은 F1 점수 약 75~91% 수준으로 적용 및 개선 가능성 확인
 
 - 적용 불가 공정은 특성 분석 수행 및 원인 진단
 
 - 향후 AI 적용 전략 수립 및 운영 방향성 제시
 

---

#### Project F: 고속 사출기 불량 탐지 POC / 수행기간: 2023.11 ~ 2023.12
- 핵심 알고리즘: Tree 기반 ML Classifier (e.g., XGBoost, LightGBM)
- 주요 결과
  
 - Tree ML기반 분류 모델을 활용한 고속 사출기 불량 탐지 AI 모델 개발
 
 - Cavity 및 CycleTime 단위로 Raw 공정 데이터를 표준화하여 예측 정확도 향상

 - 검사(Target) 시점 재정의 및 Optuna 기반 Tree ML모델 고도화 적용 → F1 점수 약 12%(70% → 82%) 향상

 ---
 



