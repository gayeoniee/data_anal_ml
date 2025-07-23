# 중고차 가격 예측 프로젝트 – Dubizzle 데이터 기반

## 프로젝트 개요
- 목표: 중고차 거래 플랫폼의 데이터를 활용해 차량의 가격을 예측할 수 있는 머신러닝 회귀 모델을 구축
- 데이터 출처: Dubizzle 중고차 판매 데이터셋 (Dubizzle_used_car_sales.csv)
- 예측 대상: price_in_aed (AED 단위 차량 판매가)

## 데이터 전처리 및 피처 엔지니어링

### 결측치 처리 및 정제
- no_of_cylinders: Unknown -> NaN -> float -> 최빈값(6) -> 정수형으로 변환
- motors_trim: NaN -> Unknown
- year: 중위값으로
- horsepower: 문자열을 수치형 범주로 인코딩, Unknown은 최빈값으로 대체
- mechanical_condition, body_condition: 'Perfect inside and out' 여부로 이진화 후 원컬럼 삭제

### 새로운 피처 생성
- car_age: 2025 - year로 차량 나이 계산
- is_perfect, is_mechanically_perfect: 차량 외관/기계 상태 이진화
- is_automatic: 변속기 자동 여부 이진화
- is_gasoline: 연료가 가솔린인지 여부 이진화
- day_posted, month_posted, year_posted: date_posted에서 파생

### 범주형 변수 인코딩
- seller_type, body_type, regional_specs, color, emirate: One-Hot Encoding
- color, body_type, model: 희소 항목은 'Other'로 통합
- company, model, motors_trim: OrdinalEncoder로 수치 인코딩

### 이상치 처리
- price_in_aed: 상위 1% 클리핑하여 price_in_aed_clipped 생성


## 모델 비교 및 선택

초기 전체 피처 기반 성능 비교
| 모델                   | Test RMSE (AED) | Test R²    | CV RMSE (AED)  | CV R²      |
| -------------------- | --------------- | ---------- | -------------- | ---------- |
| **RandomForest**     | 113,785.22      | 0.9143     | 118,026.90     | 0.9067     |
| **GradientBoosting** | 128,209.31      | 0.8912     | 124,888.78     | 0.8952     |
| **XGBoost**          | **106,275.02**  | **0.9253** | **108,258.07** | **0.9217** |

- GradientBoosting은 성능이 낮아 제외하고 XGBoost와 RandomForest만 다음 단계로 선정

Permutation Importance 기반 Top 30 피처 적용 성능

| 모델                       | Test RMSE (AED) | Test R²    | CV RMSE (AED)  | CV R²      |
| ------------------------ | --------------- | ---------- | -------------- | ---------- |
| **XGBoost (PermTop30)**  | **102,037.45**  | **0.9311** | **106,443.98** | **0.9243** |
| RandomForest (PermTop30) | 113,544.83      | 0.9147     | 117,041.54     | 0.9083     |

- RandomForest도 제외하고 XGBoost PermTop30 모델에 대해 하이퍼파라미터 튜닝을 실시

하이퍼파라미터 튜닝 (GridSearchCV)

- 모델: XGBoostRegressor
- 피처: Permutation Top 30개
- 탐색 파라미터 공간:

`xgb_params = {
    'n_estimators': [300, 500],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 6],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}`

최적 파라미터

`{
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 500,
    'subsample': 1.0
}`

## 최종 모델 성능 (XGBoost + PermTop30 + 튜닝 완료)
| 지표        | 성능                 |
| --------- | ------------------ |
| Test RMSE | **101,722.26 AED** |
| Test R²   | **0.9315**         |
| CV RMSE   | **101,324.74 AED** |
| CV R²     | **0.9316**         |


## 결론 및 인사이트
- XGBoost가 전 모델 중 가장 낮은 RMSE와 높은 R²를 달성하였고, 과적합 없이 CV 점수도 매우 안정적임
- Permutation Importance 기반 피처 선택으로 예측 성능이 향상됨 (RMSE 약 5,000 이상 개선)
