## 프로젝트 정리

### 디렉토리 구조
- `1.) 기초 분석.ipynb` : 데이터 수집 / 전처리/ 모델링 구축
- `POC_요약정리.pdf` : POC분석 결과 정리 
---

### 사용 환경
- Python 3.9.13
- pandas 2.0.3
- numpy 1.23.1
- scipy 1.9.1
- scikit-learn 1.2.2
- xgboost version: 1.7.4
- lightgbm version: 3.3.5
---

### 문제 정의

- 1.) Cogging Motor 공정데이터에 대한 조립/병렬공정 처리방식에 대한 지도학습 모델 적
---

### 주요 전처리 
  - 1.) Target의 수치형분포에 대한 이상치 기준 설정  

  - 2.) 이상치 기준 Target 이진분류 학습위한 변형 [0(Normal),1(Fault)]

  - 3.) 품질 라벨링 불균형 가중치 조정 (Over/UnderSampling)

---
### 학습 프로세스

  - 1.) 동일 입력값 대비 다수 Target에 대한 Tree기반 ML Classifier 구축

  - 2.) 성능평가에 따라 현재 Classifier 적용/불가능인 조립/병렬공정 조합 파악 



