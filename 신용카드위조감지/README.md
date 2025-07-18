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

1. 로지스틱 회귀
2. XGBoost
3. RandomForest
4. lightGBM
