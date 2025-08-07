## B2B Projects

#### Project A: 비지도 학습 기반 사출 품질 예측 시스템
- 핵심 알고리즘: K-Means, AutoEncoder, Memory-Augmented AutoEncoder
- 주요 결과
  
 - K-Means를 활용해 동일 설비·품목 내에서도 Setting별 데이터 분포를 분리하여 독립된 학습군 구성
  
 - 비지도학습 기반 실험 불량 탐지 정확도 약 86% → 실제 현장 적용 시 약 67% 
 
 - 학습 이력 유무 판단 → 유사 학습 모델 기반 대체 추론 + 임계값 가중치 적용 → 불량 탐지 정확도 평균 약 16% 향상
 
 - 예측 기반 라벨링을 통해 품질 통계 및 최적 조건 도출 → 생산 불량률 약 5% 감소
 

--- 

#### Project B: 확률적 예측 기반 제당 공정 품질 추론 및 조건 최적화 시스템
- 핵심 알고리즘: Tree ML Quantile Regression(e.g., XGBoost, LightGBM), Monte Carlo Dropout, Bidirectional LSTM/GRU
- 주요 결과
  
 - 확률적 예측 접근 → 실험 정확도 95% 이상, 실제 현장 적용 시 약 85~87% 수렴
 
 - Ensemble & KDE기반 예측분포 정량화 → 품질 정확도 평균 약 5% 향상
 
 - 예측기반 최적공정조건 추론 (Gamma-KDE & Custom경로탐색 통합) → 목표값 기반 실제 vs 추천 공정조건 일치율 약 85~90% 수준 유지 

 - 실시간 데이터 누적 기반 시스템 지속가능성 강화
 

---

#### Project C: 
#### [1] 사출 설비 품질 예측 시스템
- 핵심 알고리즘: Tree ML Classifier (e.g., XGBoost, LightGBM), AutoEncoder
- 주요 결과
  
 - 지도학습 기반 실험 불량 탐지 정확도 약 85 ~ 91% → 실제 적용 시 약 65 ~ 68%

 - 기존 라벨링 한계를 반영한 데이터 재정의 및 비지도 학습 대체 

 - 실시간 단일·다수 입력 대응형 혼합 예측 체계 구축 → 불량 탐지 정확도 약 14% 향상(→ 79% ~ 82% 수준)


#### [2] 냉각/사출 조건 추천 모듈
- 핵심 알고리즘: K-Means Clustering
- 주요 결과
  
 - 실시간 공정 입력에 대해 K-Means 군집 기반 CoolingTime / InjectionTime 추천
 
 - 조건별 표준편차 및 Gaussian Noise 적용으로 추천값 유연성 확보
   
 - 실제 vs 추천 조건 일치도 차이 약 3~5% 이내 유지

 - 실시간 데이터 누적 기반 시스템 지속가능성 강화

---

#### Project D: 사출설비 불량 예측 및 TTA 인증
- 핵심 알고리즘: Tree ML Classifier (e.g., XGBoost, LightGBM)
- 주요 결과
  
 - 학습데이터 기반 변수 선택 및 교차검증을 통한 최적 모델 구성 → F1 점수 약 95% 달성
 
 - 특정 사출설비 대상 불량 예측 코드 개발 및 모델 성능 검증 완료
 
 - TTA 인증을 위한 품질 예측 기능 개발 및 적용 준비 완료
 

---

#### Project E: Cogging Motor 기반 조립·병렬 공정 불량 탐지 POC
- 핵심 알고리즘: Tree 기반 ML Classifier (e.g., XGBoost, LightGBM)
- 주요 결과
  
 - Tree 기반 분류 모델을 활용해 조립·병렬 공정의 불량 탐지 AI 모델 개발
 
 - 공정별 AI 적용 가능성 검토: 일부 공정은 F1 점수 약 75~91% 수준으로 적용 및 개선 가능성 확인
 
 - 적용 불가 공정은 특성 분석 수행 및 원인 진단
 
 - 향후 AI 적용 전략 수립 및 운영 방향성 제시
 

---

#### Project F: 고속 사출기 불량 탐지 POC
- 핵심 알고리즘: Tree 기반 ML Classifier (e.g., XGBoost, LightGBM)
- 주요 결과
  
 - Tree 기반 분류 모델을 활용한 고속 사출기 불량 탐지 AI 모델 개발
 
 - Cavity 및 CycleTime 단위로 Raw 공정 데이터를 표준화하여 예측 정확도 향상

 - 검사(Target) 시점 재정의 및 Optuna 기반 모델 고도화 적용 → F1 점수 약 70% → 82%로 향상

 ---
 

## 기술 스텍



