# 프로젝트 개요

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
- 
      → 데이터의 1일 Lot단위 생산 특징
      → 1일 단위 총 생산량 (정상 + 불량품질 총합 개수만 파악 가능)
      → 개별 데이터의 Good/Bad Qty의 수치는 총생산량 기반으로 기록 (개별 데이터의 품질 의미 X)

- 2.) Setting기반 데이터 분포 차이
- 
      → 동일 설비 & 품목일지라도 다변량 Production(Sensing)은 공정의 Unique Setting 집합에 따라서 분포의 차이를 나타냄
      
  
  
  
      



