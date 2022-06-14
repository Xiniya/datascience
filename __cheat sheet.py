#!/usr/bin/env python
# coding: utf-8

# # 머신러닝 문제 풀이 순서

# In[1]:


# 필요 패키지 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


# ## EDA 및 전처리
# 
# (1) 결측치 존재 여부 : 평균/중앙값 대치 or KNN 활용
# 
#     - 10% 미만: 삭제 or 대치
#     - 10 ~ 20%: Hot deck(매년 자료->해당 년 자료 추정) or regression or model based imputation
#     - 20 ~ 50% 이상: regression or model based imputation
#     - 50% 이상: 해당 컬럼(변수) 자체 제거
# 
#         - 단순 대치법 : 수치형 변수라면, 각 컬럼의 평균이나 중앙값을 사용하여 결측치를 보간할 수 있으며, 명목형/범주형 변수라면 최빈값을 사용하여 대치할 수 있음
#         - KNN을 이용한 결측치 대체 : 보간법 중 결측치가 없는 컬럼들의 최근접 이웃 알고리즘을 통해 결측치가 있는 변수 대체를 할 수 있다.
#         * 단, KNN을 이용할 때에는 거리 계산이 가능한 수치형 변수만 사용 가능하다.
#         * 많은 데이터가 전부 결측치인 경우가 존재한다면 제거, 한 컬럼씩만 결측이라면 삭제보다는 대체를 활용.
# 
# (2) 데이터 타입 설명
# 
#     - 연속형/이산형/범주형 변수 확인. 이산형/범주형 변수는 인코딩을 통하여 변환 가능성 검토.
#     
#         - 이산형 변수는 boolean 타입으로 변경하여 KNN을 이용한 결측치 처리와 머신러닝에서 변수를 사용가능하도록 변환.
#         - 수치나 순위형 변수는 원핫인코딩 하지 않는다.
#         
# (3) 종속변수 분포 설명
# 
#     - 종속변수 정규 분포를 띄는 지, 불균형인지 확인.
#     
# (4) 종속변수와 독립변수의 상관관계 설명
# 
#     - 수치형 변수와 종속변수 상관성 확인. 큰 상관성이 확인 안된다면, 파생변수 활용 고려.
#     - 종속변수가 이산형이고, 독립변수가 모두 연속형 수치형 변수일 때, 종속변수에 따른 관계를 확인하기 위해 groupby 활용 평균내고, 시각화.
#     
# (5) 독립변수 상관관계 설명
# 
#     - 독립변수 사이 다중공선성 있는 지 체크 (0.9 이상).
#     
# (6) 이상치 식별
# 
#     - 이상치를 판단하기 위해서는 mean, min, max값을 확인하는 것이 좋다. 평균과 min, max 값이 std에 비해 한참 차이가 난다면, 이상치가 있을 가능성이 높다.
#     - 이상치를 정확하게 판단하기 위해서는 boxplot으로 시각화하여 보는 것이 정확하다. boxplot을 한 번에 그려주기 위해서는 melt를 이용해 데이터를 재구조화해주어야 한다.
#     - 최소값과 최대값이 차이가 많이 나는 컬럼이 존재하는 경우, 선형 모델 사용 시 scale을 적용할 필요가 있음.
#     
# (7) 데이터 분할
# 
#     1. 랜덤 분할 : train test 데이터세트를 나누어서 학습된 데이터를 검증할 수 있으며, 분할 시에 무작위로 사용자가 지정한 비율로 분할한다. 
#                  전체 분석 데이터 중 머신러닝 모델을 학습시키기 위한 학습용 데이터와 테스트용 데이터를 나누어서 적용시키는 이유는 모델 결과가 다른 데이터에도 적용 가능한 지,
#                  일반화가 가능한 지를 검증하기 위함이다.
#     
#     2. 층화 추출 기법 : 종속변수가 범주형인 경우에는 종속변수의 클래스의 비율을 기준으로 학습용 데이터와 테스트용 데이터의 비율이 동일하게 분할한다. 
#                      즉, 클래스의 편항을 막을 수 있다.
#     
#     * 종속변수가 연속형인 경우, 회귀분석을 사용한다. 이 경우에는 층화추출기법이 아닌 랜덤 샘플링을 통한 분할을 사용하여, 7:3 비율로 분할한다.
#     
# (8) 클래스 불균형 처리
# 
#     오버샘플링 / 언더샘플링
#     
#     1. 오버 샘플링 : 오버 샘플링 기법은 비중이 적은 데이터를 추가로 생성해 수를 늘려 데이터 불균형을 극복하는 방식이다. 
#                   소수 레이블을 가진 데이터 세트를 다수 레이블을 가진 데이터 세트의 수만큼 증식시켜 학습에 충분한 데이터를 확보하는 기법이다. 
#                   언더 샘플링은 데이터 손실의 문제로 인해 예측 성능이 저하되는 단점이 있으므로, 일반적으로는 불균형한 데이터를 처리하는 방식으로 오버 샘플링을 사용한다.
#     
#         1. Random Oversampling
#             - 소수 클래스에 속하는 데이터의 관측치를 복사하는 방식으로 데이터를 증식한다.
#             - 데이터를 단순 복사하는 방식이르모 기존의 데이터와 동일한 복제 데이터를 생성한다.
#             - Random Oversampling은 소수 클래스에 과적합이 발생할 가능성이 있다는 단점이 있지만, 사용방법이 간단하다는 장점이 있다.
#             
#         2. SMOTE
#             - SMOTE는 적은 데이터 세트에 있는 개별 데이터들의 k-최근접 이웃을 찾아, 해당 데이터와 k개 이웃들의 차이를 일정한 값으로 만들어 
#               기존 데이터와 약간의 차이를 지닌 새로운 데이터를 생성하는 방식이다.
#             - SMOTE는 Resampling 방식보다 속도가 느리다는 단점이 있지만, 데이터를 단순히 동일하게 증식시키는 방식이 아니기 때문에, 
#               과적합 문제를 예방할 수 있다는 장점이 있다.
#             
#     2. 언더 샘플링 : Under sampling은 다수 클래스를 감소시켜 소수 클래스 개수에 맞추는 방식으로, 
#                   대표적으로 random으로 다수의 클래스의 데이터를 선택하여 삭제하는 RandomUnderSampler, 
#                   서로 다른 클래스가 있을 때, 서로 다른 클래스에 가장 가까운 데이터들이 토멕 링크로 묶여서 토멕 링크 중 다수 클래스의 데이터를 제거하는 Tomek Link 방식이 있다.
#     
#     * 보통의 경우 답안 : 둘 중 해당 데이터에서는 Oversampling이 적합하다. 데이터가 총 768개로 당뇨병 화자를 대표하기에는 너무 적은 data이다. 
#                      심지어 환자의 수는 768명 중 268명 뿐이다. Undersampling을 선택하게 되면 전체 데이터가 더 적어지기 때문에 오버피팅이 일어날 위험이 더 크므로 
#                      oversampling을 선택하겠다.

# # 모델링
# 
# svm, xgboost, randomforest 3개의 알고리즘 공통점을 쓰고 학생 성적 예측 분석에 적합한 알고리즘인 지 설명하시오.
#     
#     3개 알고리즘의 공통점
# 
#     1. 회귀분석과 분류분석을 모두 할 수 있는 분석알고리즘이다.
#     2. 모두 범주형 변수를 독립변수로 사용할 수 없어 변환을 해주어야 한다.
#     3. 과대적합 과소적합을 피하기 위한 매개변수의 설정이 필요하다.
#     4. 회귀분석에서 다중공선성의 문제를 해결할 수 있다.
#     
#     해당 데이터에서는 종속변수의 값이 연속형 변수이므로 회귀분석이 적합하다. 회귀분석에서 다중공선성의 문제를 해결하는 것이 중요한데 svm은 커널트릭을 통해, 
#     xgboost와 randomforest는 트리 모델을 통해 다중공선성을 해결할 수 있다. 그러므로 회귀분석을 지원하는 위 3가지의 알고리즘은 연속형 변수를 예측하기에 적합하다.
# 
# 
# 당뇨병 환자를 예측하는 최소 3개 이상의 알고리즘을 제시하고 정확도 측면의 모델 1개와 속도 측면의 모델 1개를 제시하시오.
# 
#     속도 측면에서 Logistic Regression, 정확도 측면에서 svm, 기타로 xgboost를 제시한다.
#     
# 속도 개선을 위한 차원축소 방법을 설명하고 수행하시오.
# 
#     속도 개선을 위한 차원축소방법인 PCA를 사용할 수 있다. 예측의 성능은 원본의 데이터를 그대로 사용하는 것보다 떨어질 수 있지만, 
#     차원을 축소함으로써 예측의 속도는 훨씬 상승시킬 수 있다. 
#     *PCA를 사용할 때, 데이터의 스케일에 따라 각 주성분이 설명가능한 분산량이 달라질 수 있기 때문에 데이터 스케일링을 꼭 해주어야 한다.

# In[ ]:




