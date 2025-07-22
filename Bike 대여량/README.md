# 자전거 수요 예측 프로젝트 (Bike Sharing Demand)

## 프로젝트 개요

- 목표: 시간대별 기상정보 및 날짜 데이터를 바탕으로 자전거 대여 수요(count)를 예측하는 회귀 모델 구축
- 데이터 출처: Kaggle - Bike Sharing Demand
- 모델 평가 지표: RMSE (Root Mean Squared Error)


## 1. 데이터 전처리 및 EDA

### 주요 컬럼 분류

| 변수 유형 | 컬럼                                                      |
| ----- | ------------------------------------------------------- |
| 범주형   | `season`, `holiday`, `workingday`, `weather`            |
| 수치형   | `temp`, `atemp`, `humidity`, `windspeed`, `count`       |
| 파생 | `datetime` -> `year`, `month`, `hour`, `weekday`, `date` |

### 상관관계 분석
- temp와 atemp는 매우 높은 상관관계(> 0.99)
- humidity는 대여 수(count)와 음의 상관관계
- windspeed는 상관관계가 낮아 제거함

### 타겟 분포 확인 및 변환

- count는 오른쪽 꼬리가 긴 분포 → 로그 변환(log1p) 수행
- 이유: 이상값 영향 완화 + 정규성 증가 + 비율 학습 가능


## 2. 특징 생성 및 데이터 정제

- datetime 컬럼에서 year, hour, weekday 등 파생
- casual, registered, count, datetime, windspeed, month, date 등 모델 입력에서 제거
- count는 타겟
- casual + registered = count → 누설 방지
- 범주형 변수 -> pd.get_dummies()로 원-핫 인코딩 처리 (drop_first=True)
- temp, atemp, humidity → StandardScaler로 정규화


## 3. 모델 학습 및 비교

| Model                | Test RMSE | Test R2 | CV RMSE | CV R2  |
| -------------------- | --------- | ------- | ------- | ------ |
| **LGBM**             | 44.7378   | 0.9394  | 45.2415 | 0.9374 |
| **Linear\_Poly**     | 47.0057   | 0.9331  | 48.6829 | 0.9273 |
| **XGBoost**          | 51.1132   | 0.9208  | 55.2360 | 0.9067 |
| **RandomForest**     | 60.3718   | 0.8896  | 66.5023 | 0.8646 |
| **GradientBoosting** | 92.4181   | 0.7412  | 92.9320 | 0.7359 |


## 최종 선택 모델

- 모델명: LGBMRegressor
- 튜닝 파라미터:   
n_estimators=800   
max_depth=12   
learning_rate=0.1   
num_leaves=20   
subsample=0.8,   
colsample_bytree=0.8   

- LGBM은 뛰어난 성능 발휘

## 4. 모델 평가 방식
- 평가 지표는 로그 예측값을 expm1()으로 역변환 후 RMSE, R2 계산
- 교차검증은 cross_val_score() + make_scorer() 사용
- log 예측값은 np.clip(pred_log, 0, 10) 처리로 음수 예측 방지


## 5. 예측 및 제출
- test 데이터에 동일한 전처리 적용 (scaling, get_dummies, 컬럼 정렬 등)
- 예측된 log 스케일 결과는 np.expm1()로 복원
- submission.csv로 저장 (datetime, count)

## 주요 인사이트 요약
- season, weather, hour, workingday는 대여 수에 큰 영향 -> 중요 피처
- casual, registered는 타겟 누설로 제거
- 로그 변환은 대여 수의 이상치 영향 완화에 효과적
- LGBM이 모든 모델 중 가장 좋은 성능 확보 -> 최종 모델 선정

## 향후 개선 방향
- Feature engineering 고도화
- GridSearchCV로 LGBM 하이퍼파라미터 최적화
- 변수 중요도 해석 강화

### 캐글 제출 결과

0.38906 으로 약 155등
