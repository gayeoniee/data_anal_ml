# 신용카드 사기 탐지 모델 성능 비교 및 오버샘플링 적용 보고서

## 1. 데이터 개요
  - 사용 데이터: creditcard.csv
  - 총 샘플 수: 284,807건
  - 타겟 클래스: Class (정상: 0, 사기: 1)
  - 심각한 클래스 불균형 존재: 사기 거래 비율 약 0.17%

## 2. 데이터 전처리
(1) 이상치 제거
  - Amount 변수에 대해 IQR 기반 이상치 제거 수행
  - 정상 거래(0)만을 대상으로 IQR 적용 후 사기 거래와 병합

(2) 변수 스케일링
  - Amount, Time 변수는 StandardScaler를 이용하여 scaled_amount, scaled_time으로 변환
  - 원래 변수는 제거하고 스케일링된 변수만 사용

(3) 상관관계 분석
  - Class와 상관이 높은 상위 변수 확인: V17, V14, V12, V10 등

## 3. 모델 성능 비교 (기본 모델, 오버샘플링 X)

테스트셋 분할

- train_test_split(stratify=y)로 클래스 불균형 유지
- test_size=0.2, random_state=42

### 모델별 성능 요약 (양성 예측 기준)
| 모델                  | Recall (재현율) | Precision (정밀도) | F1-score | ROC-AUC |
| ------------------- | ------------ | --------------- | -------- | ------- |
| Logistic Regression | 낮음 (0.67)    | 높음    0.90          | 보통    0.77   |   |
| Random Forest       | 0.82        | 우수     0.96         | 우수    0.88   |    |
| **XGBoost**         | **0.81**     | **0.95**        | **0.94** | **최고** 0.978 |
| LightGBM            | 0.63      | 0.28             | 0.38      | 0.779      |


XGBoost가 전반적으로 가장 뛰어난 성능을 보여줌

## 4. Threshold 튜닝 (XGBoost 기준)
기본 threshold = 0.5   

- F1-score = 0.8889
- ROC-AUC = 약 0.98

최적 threshold 탐색 (Precision-Recall Curve 기반)

- precision_recall_curve()로 threshold별 F1-score 계산
- 최적 threshold ≈ 0.3360
- 이때 F1-score = 0.8913, Recall 상승, Precision 약간 감소

| Threshold  | Precision | Recall | F1-score |
| ---------- | --------- | ------ | -------- |
| 0.5        | 0.98      | 0.81   | 0.88   |
| **0.3360** | 0.95      | 0.84   | 0.89     |

threshold를 낮추면 더 많은 사기를 잡을 수 있음 (Recall 증가) → 실제 서비스 적용시 유용


## 5. 오버샘플링 (SMOTE)
적용 대상
- train 데이터에 대해 SMOTE로 증가시킴
- XGBClassifier, LGBMClassifier에 적용

### 오버샘플링 후 성능 (XGBoost 기준)

| 모델               | Precision | Recall   | F1-score | ROC-AUC |
| ---------------- | --------- | -------- | -------- | ------- |
| XGBoost (SMOTE)  | 0.85      | **0.87** | 0.86     | 0.974    |
| LightGBM (SMOTE) | 0.70      | **0.85** | 0.90     | 0.975    |
| Logistic Regression |  0.09   | 0.87        |   0.17   |  |
| Random Forest       |  0.92      |    0.82       | 0.86   |    |

SMOTE 적용 시 Recall(재현율)이 눈에 띄게 상승, 사기 탐지에 더 민감해짐   

## 6. 결론 및 제언
- 기본 모델 중에서는 XGBoost가 가장 우수
- threshold 조절을 통해 Precision vs Recall 트레이드오프 조정 가능
- SMOTE 오버샘플링은 사기 탐지 향상에 효과적
