# 고객 만족 예측 모델링 프로젝트

## 프로젝트 개요
- 목표: 고객의 만족 여부(TARGET)를 예측하여 만족하지 않은 고객을 사전에 파악하고 맞춤형 대응 전략을 수립할 수 있도록 함

## 데이터 개요 및 전처리

### 데이터 특성
- 총 370개 이상의 피처를 가진 고차원 구조
- TARGET 값은 이진 분류 (0: 만족, 1: 불만족)
- 불만족 고객의 비율이 약 4% 수준 -> 심각한 클래스 불균형


### 전처리 과정
| 전처리 항목        | 설명                             |      |                              |
| ------------- | ------------------------------ | ---- | ---------------------------- |
| 이상치 처리        | `var3`에서 `-999999` → 최빈값 2로 대체 |      |                              |
| 결측치           | 없음                             |      |                              |
| 상관관계 기반 컬럼 제거 | \`                             | corr | > 0.95\` 이상 피처 제거 (다중공선성 완화) |

## 타겟 분포 및 분할 전략
- 전체에서 불만족 고객(1) 비율은 약 3.7%
- Stratified Split 적용하여 학습/테스트 데이터 분할 시 비율 유지

## 모델 비교 및 성능 분석

**평가 지표: ROC AUC**
| 모델                  | ROC AUC (Test) | Precision (Class 1) | Recall (Class 1) |
| ------------------- | -------------- | ------------------- | ---------------- |
| Logistic Regression | \~0.61         | 낮음                  | 낮음               |
| Random Forest       | \~0.80         | 낮음                  | 낮음               |
| Gradient Boosting   | \~0.83         | 개선                  | 낮음            |
| **XGBoost**         | **\~0.83+**    | 가장 우수               |낮음         |


- Gradient Boosting 및 XGBoost가 가장 우수
- 다만 불만족 고객(1)에 대한 정밀도와 재현율은 개선 여지 있음

## 하이퍼파라미터 튜닝 결과

**최종 채택 모델: XGBoost**

`XGBClassifier(   
    colsample_bytree = 0.8,   
    learning_rate = 0.08,   
    max_depth = 4,   
    min_child_weight = 2,   
    n_estimators = 300,   
    subsample = 0.8,   
    eval_metric = 'auc',   
    random_state = 42   
)`   
- Early Stopping 적용
- 과적합 방지를 위해 colsample_bytree, subsample, min_child_weight 등 조정

## 최종 제출 결과 (캐글 기준)
| 모델      | Private Score | Public Score |
| ------- | ------------- | ------------ |
| GBM     | 0.82112       | 0.83472      |
| XGBoost | **0.82627**   | **0.83796**  |

## 향후 개선 방향
- 클래스 불균형 대응: SMOTE, ADASYN 등 오버샘플링 기법 도입
- scale_pos_weight 조절: 퍼블릭 점수는 향상되었으나 프라이빗 점수 저하 → 신중한 적용 필요
- Feature Engineering: 중요한 피처 추출 후 재학습 (Permutation Importance, SHAP 등)
- 앙상블 및 스태킹 기법 적용
