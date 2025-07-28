# 프로젝트 정리

## 디렉토리 구조
- `1.) 데이터수집_postgre.ipynb` : 데이터 수집용 쥬피터 노트북
- `2.) 학습데이터_구축_postgre.ipynb` : 학습용 데이터 정제
- `3.) 학습모델구축.ipynb` : 모델링 및 학습 
- `4.) 임시라벨링_데이터점검.ipynb` : 라벨링 및 검토
- `5.) RecipeAI.ipynb` : 최적 생성조건 데이터 생성

---

### 사용 환경
- Python 3.9.21
- pandas 2.2.3
- numpy 1.23.1
- scipy 1.10.1
- tensorflow 2.7.0
- scikit-learn 1.2.2
- pymongo 4.10.1
- psycopg2 2.9.10
- fastapi 0.115.12
- uvicorn 0.34.3

---

### 문제 정의
- 1.) 사출 생산데이터의 1대1 라벨링 부재
  
      → 데이터의 1일 Lot단위 생산 특징
      → 1일 단위 총 생산량 (정상 + 불량품질 총합 개수만 파악 가능)
      → 개별 데이터의 Good/Bad Qty의 수치는 총생산량 기반으로 기록 (개별 데이터의 품질 의미 X)

- 2.) Setting 기반 데이터 분포 차이
  
      → 동일 설비 & 품목일지라도 다변량 Production(Sensing)은 공정의 Unique Setting 집합에 따라서 분포의 차이를 나타냄
     
---

### 주요 전처리 
  - 1.) 설비/품목/1일 Lot생산단위 불량률 계산
    
    <img width="1273" height="531" alt="그림1" src="https://github.com/user-attachments/assets/137fb13a-177d-4add-8443-98f7abe4f377" />

  - 2.) Setting 데이터 구분

      → 다변량 Setting데이터의 Unique조합 계산 (Drop Duplicates)
    
      → K-Means Clustering을 통한 고유 Setting 조합에 대한 Numbering 부여

      → 동일 설비/품목은 여러 날짜(Lot)별로 1개 혹은 다수의 Setting에 의해 생산 진행

    <img width="296" height="326" alt="화면 캡처 2025-07-28 131912" src="https://github.com/user-attachments/assets/bf60ecc7-8386-4bb4-aa06-07dd19738e8e" />

  - 3.) Lot단위 불량률 & 고유 Setting 정보 정량화 기반 학습데이터 완성 

     <img width="1918" height="475" alt="그림3" src="https://github.com/user-attachments/assets/b9ad7c9d-e64c-4d7f-ba4e-d368b5caf8a0" />


---

### 주요 전처리 

  - 1.) 총불량률 <1.0%인 [Facility-Item-Setting] 데이터 학습 / 총불량률 >=1.0 검증

  - 2.) 학습 데이터 개수에 따른 학습모델 선택

       → IsolationForest, AutoEncoder, Memory Augmented AutoEncoder

       → Ex.) 특정 (설비/품목/Setting Cluster)의 분리학습 

       <img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/52b96386-97fd-439a-9f30-c39a8ad1529c" />










