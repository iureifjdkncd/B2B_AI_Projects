## Project C: 사출 설비 품질 예측 시스템 및 공정조건 추천 모듈
(K-등대공장 구축사업 / 수행기간: 2023.05 ~ 2025.03)

---

#### 프로젝트 개요

- 사출 공정에서 실시간 품질 예측 위한 비지도 학습 기반 불량탐지 시스템 구축 (A)

---

#### A. 품질 예측 시스템 

1.) 문제 정의
- 다수 사출기의 실시간 데이터 기반 이진 품질 예측(정상/불량)

- Label 적합도 부족 , 데이터 혼재 등 문제 발생으로 인해 지도학습 기반 추론 방식의 한계

- 기존 사출+PLC 데이터에서 → 사출 단독 기반 AutoEncoder 중심 비지도 학습 전환

---

2.) 핵심 기술 및 알고리즘

- Tree 기반 분류기 (RandomForest, XGBoost, LightGBM, GBM, Stacking)

- AutoEncoder 기반 비지도 이상 탐지

- AE기반 Hybrid Prediction 구조 (Fixed + Distribution Adaptable)

- KDE 기반 Fault Threshold 동적 계산 및 Robust Z-Score 기반 Anomaly Predicted value Recheck 프로세스

---

#### 실시간 추론 프로세스

1.) Fixed Prediction: AE 기반 고정 임계값으로 실시간 단일 샘플 품질 예측

2.) Adaptable Prediction: 다수 샘플의 test_mae_loss 분포를 KDE로 정량화한 임계값으로 실시간 샘플 품질 예측

3.) Hybrid Prediction: 예측 안정성(CV, Skew, Kurtosis) 기반 Fixed 단일/Fixed & Adaptable 혼합 판단

4.) Recheck Process: (3)에서 Fault로 예측된 데이터에 대해 CP/CPK, Robust Z, Threshold Gap 등 통계 기반 재검토

5.) 작업시간 예외 처리: 작업자 직접 제어 시간대 → 예측 제외 처리

--- 

#### 주요 성과

- 지도학습 기준 초기 실험 데이터 기준 불량 탐지 정확도 약 85 ~ 91%, 실제 현장 적용 시 약 65 ~ 68%로 하락 

- 비지도학습 기반 추론 변경 이후, Hybrid AE추론 구조(Fixed + Adaptable) 적용을 통해 작업자 검사 기반 정확도 약 14% 향상 (→ 79% ~ 82% 수준)

---

#### 기여도

- 품질 예측 전처리, 학습 구조, AE 모델 학습/추론 설계 및 구현

- Hybrid Prediction 시스템 구성 및 Recheck 알고리즘 구현

---

---
## Project C: 공정조건 추천 시스템 (사출/냉각 시간)

---

#### 프로젝트 개요

- 특정 공정조건을 제공하기 위한 군집 기반 추천 모듈 구축 (B)

---

#### 문제 정의

- 현재 Set_InjectionTime / CoolingTime Mean에 대한 실시간 추천값 제공 기반 모니터링 시스템 필요

- 수집되는 Production과 Environment 데이터 간 시점 불일치 문제 발생 가능성 존재

- 통계 기반 유연한 추천값 + 예외 대응이 가능한 K-Means 기반 클러스터링 추천 시스템 필요

---

#### 핵심 기술 및 알고리즘

- K-Means 기반 Set_변수 대상 클러스터링 (Production + Environment 기준)

- 실시간 군집 예측 + 군집 내 부분집합 중 현재값 대비 최근사 InjectionTime / CoolingTime Mean 선택

- 추천값의 표준편차 변수, Gaussian Noise 기반 범위 산정 및 출력값 유연화

- 예외 케이스: 군집 미탐색, 매칭 불가 등 → 실시간 30개 수집 데이터 대상 평균 대체 추천

---

#### 실시간 추천 프로세스

1.) 실시간 데이터 수집 → Production + Environment 매칭

2.) 군집 예측 → 현재 입력값에 대해 K-Means Cluster 예측 & 해당 Cluster를 가진 학습데이터 선택

3.) 추천값 산정

- 해당 군집의 부분집합 중 현재 Set_Injection/CoolingTime_mean과의 최근사값 선택 

- 표준편차 변수 기반 Set_Injection/CoolingTime_mean ± Std 계산된 범위 중 무작위 추천값 선택

4.) 예외처리 1: 군집 데이터 1개 → 현재 군집기반 추천값 사용

5.) 예외처리 2: 추천값이 현재값보다 낮은 경우 → Gaussian Noise 보정

6.) 예외처리 3: 실시간 데이터 조회 실패 → 전체 평균으로 대체 추천

---

#### 주요 성과

- Set_Injection/CoolingTime_mean 실시간 추천값 vs 실제 세팅값 오차 평균 약 3 ~ 5% 유지

---

#### 기여도

- 군집기반 사출조건 추천 간편 알고리즘 설계 및 전처리 흐름 구축

- 유연화 로직 구현으로 실시간 매칭/예측/추천 전체 흐름 최적화 및 예외 상황 대응 제공

- 실시간 군집 예측 대상 데이터 → 학습데이터에 업데이트로 추천값 제공 운영 지속가능성 강화

---
