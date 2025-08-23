## Project A - 비지도 학습 기반 사출 품질 예측 및 최적 세팅 추천 시스템
(A사 자동차 부품 제조기업 / 자율형 공장 구축 사업 / 수행기간: 2023.08 ~ 2025.07)

--- 

### 디렉토리 구조
- `1.) 데이터수집_postgre.ipynb` : PostgreSQL 연동을 통한 데이터 수집
- `2.) 학습데이터_구축_postgre.ipynb` : 학습용 데이터 정제 및 가공
- `3.) 학습모델구축.ipynb` : 모델링 및 학습 실행
- `4.) 임시라벨링_데이터점검.ipynb` : 학습모델 기반 데이터 라벨링 및 점검
- `5.) RecipeAI.ipynb` : 최적 생산 조건 도출용 데이터 생성
- `prediction_점검용.ipynb` : 추론 검증 및 점검
- `prediction.py` : 추론 API 로직
- `memae.py` : Memory-Augmented AutoEncoder구현 (Custom 함수)
- `main.py` : FastAPI 기반 추론 서버
- `predictionAPI.bat` : FastAPI 실시간 추론 실행 스크립트

---

### 사용 환경
- Python 3.9.21
- pandas(2.2.3), numpy(1.23.1), scipy(1.10.1), tensorflow(2.7.0), scikit-learn(1.2.2), pymongo(4.10.1), psycopg2(2.9.10), fastapi(0.115.12), uvicorn(0.34.3)

---

### 문제 정의
- 1.) **PostgreSQL Raw 사출 데이터의 라벨링 부재**
  - Lot(일 단위) 생산 데이터에는 정상/불량 총합만 기록되어 있음.
  - 개별 샘플 단위의 Good/Bad 여부 라벨링이 없어, **샘플 단위 품질 학습 불가**.
- 2.) **Setting 조합별 데이터 분포 차이**
  - 동일 Maker, 동일 Facility-Item이라도 Lot별로 서로 다른 Setting 조합으로 운영.
  - Setting 조합에 따라 Sensing값들의 분포가 상이 → **독립적인 학습군 구성 필요.**

---

### 주요 전처리 
- 1.) **Lot 단위 불량률 계산**
  - Lot별 총 생산량 대비 불량률(defect_rate) 집계.
  - defect_rate < 1.0% → 학습용, ≥ 1.0% → 검증용으로 분리.
  - 정상(0.0%)만 학습되는 과적합을 방지하기 위해 불량률 1% 미만까지 학습에 포함.
  - 샘플 수 500 미만인 데이터는 학습에서 제외.
  <img width="700" height="250" alt="그림1" src="https://github.com/user-attachments/assets/137fb13a-177d-4add-8443-98f7abe4f377" />
- 2.)	**Setting기반 데이터 구분**
  - 다변량 Setting 데이터의 중복 제거 후 Unique 조합 추출.
  - K-Means 기반으로 Cluster 번호 부여.
  - Maker/Facility/Item/Setting 단위로 세분화하여 학습 데이터셋 구성.
  <img width="296" height="326" alt="화면 캡처 2025-07-28 131912" src="https://github.com/user-attachments/assets/bf60ecc7-8386-4bb4-aa06-07dd19738e8e" />
---

### 학습 프로세스  
- 1.)	**학습 데이터 구축**
  - Lot 단위 불량률과 고유 Setting 정보를 정량화하여 최종 학습데이터 완성
  <img width="1918" height="475" alt="그림3" src="https://github.com/user-attachments/assets/b9ad7c9d-e64c-4d7f-ba4e-d368b5caf8a0" />
- 2.)	**모델 선택 로직 (데이터 규모 기반)**
  - 샘플 < 1000: **IsolationForest**
  - 1000 ~ 5000: **AutoEncoder**
  - 5000 이상: **Memory-Augmented AutoEncoder (MemAE)**
  - AE 계열은 Early Stopping 및 Best Weight 저장으로 학습 안정성 확보.
  <img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/52b96386-97fd-439a-9f30-c39a8ad1529c" />

---

### 실시간 추론 프로세스  
- 1.)	**실시간 데이터 전처리 (MongoDB)**
  - 품목명, 생산량, Setting 등 결측치는 Forward/Backward/Mean/Mode Fill 적용.
  - 예측 불가능한 조건은 사전 예외 처리하여 시스템 안정성 확보.
- 2.)	**기본 추론 (학습 이력 존재)**
  - 현재 Setting → Cluster 예측 → 해당 Cluster 학습 모델 로드.
  - MinMaxScaler(clip=True)로 스케일링 → 발산 방지.
  - Trained Reconstruction MAE Loss분포의 변동계수(CV) 기반으로 **Threshold 가중치 및 Margin 부여**.
- 3.) **적응형 추론 (학습 이력 없음 or 기존 학습불가 Setting정보)**
  - Euclidean Distance 기반으로 현재 Setting의 **최근접 Trained Setting 탐색**.
  - 최근접 Cluster의 학습모델을 활용하여 대체 추론 수행.
  - Threshold는 Trained mae loss CV + 거리 기반 가중치를 적용하여 유연성 확보.
- 4.) **임계값 유연화**
  - 학습 당시 고정 Threshold와 Trained Reconstruction MAE Loss Max값 중 큰 값을 선택.
  - Trained mae loss CV(필수 적용) 및 Distance 기반 가중치(**적응형 추론 시**)로 임계값 조정
  - 동적 Margin을 설정하여 실시간 test mae loss와의 허용 오차 정량화
- 5.) **적응형 추론 예시 (학습이력 존재O / 존재X)**
<img width="500" height="350" alt="화면 캡처 2025-07-28 135612" src="https://github.com/user-attachments/assets/2d0c0c99-271e-4665-8bee-0b73e0742804" />
<img width="500" height="350" alt="화면 캡처 2025-07-31 192107" src="https://github.com/user-attachments/assets/788e8d9c-435c-4cf7-a0c6-433cbdcebc5d" />
---

### 최적 생산 Setting조건 제공용 데이터 구축   
- 1.) AE 모델 기반 Threshold로 기존 학습데이터에 **임시 라벨링** 부여.
- 2.)	설비·품목별로 **최다 생산 + 최소 예측 불량률**을 만족하는 Setting 도출.
- 3.)	각 Setting 조합별 Min/Mean/Max 생산 데이터 및 품질 통계 정량화.
- 4.) 이를 기반으로 생산 계획에 참고 가능한 **최적 조건 가이던스** 제공.
<img width="1599" height="455" alt="그림4" src="https://github.com/user-attachments/assets/18c411e5-9e66-4d15-b073-d22f9d167a46" />





