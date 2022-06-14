#!/usr/bin/env python
# coding: utf-8

# # 데이터 가공 및 시각화
# - 1. 전처리
#     - 결측값 처리: 단순대치, 평균 대치, 단순확률 대치 (Hot-deck, nearest neighbor), 다중 대치, knnImputation, centralimputation
#     - 클래스불균형: 업샘플링 (SMOTE, Boaderline SMOTE, Adasyn), 다운샘플링
#     - 이상값 처리: 극단값 절단, 조정
#     - **변수 변환, 스케일링: 수치형 변수 변환(로그변환, 제곱근변환, 지수변환, 제곱변환, Box-cox 변환, 표준화, 정규화), 범주형 변수 변환(범주형 변수 인코딩, 대규모 범주형 변수처리), 날짜 및 변수 변환,  피쳐스케일링**
#     - 원핫인코딩(더미변수), 컬럼 트랜스퍼, 구간분할, 이산화, 피쳐선택
# - 2. 표본 추출: 단순랜덤 추출법, 계통추출법, 집락추출법, 층화추출법
# - 3. 데이터 분할: 구축/검정/시험용, 홀드아웃방법, 교차확인방법 (10 fold 교차분석), 부트스트랩
# - 4. 그래프 그리기:
#     - 산점도, 막대그래프, 선그래프, 히트맵, 서브플롯, 트리맵, 도넛차트, 버블차트, 히스토그램, 체르노프 페이스, 스타차트, 다차원척도법, 평행좌표계
#     - 도식화와 시각화

# ## 범주형 변수 변환 (Categorical feature)
# 특정 애플리케이션에 가장 적합한 데이터 표현을 찾는 것을 특성 공학 (feature engineering)이라고 한다. 올바른 데이터 표현은 지도 학습 모델에서 적절한 매개변수를 선택하는 것보다 성능에 더 큰 영향을 미친다. 여기서는 범주형 특성 (categorical feature) 혹은 이산형 특성 (discrete feature)를 변환하는 방법들을 살펴보려고 한다.
# 
# GBDT와 같이 결정트리에 기반을 두는 모델에서는 레이블 인코딩으로 범주형 변수를 변환하는 게 가장 편리하지만, 타겟 인코딩이 더 효과적일 때도 많다. 다만 타겟 인코딩은 데이터 정보 누출의 위험이 있다. 원핫인코딩이 가장 전통적인 방식이고, 신경망의 경우에는 임베딩 계층을 변수별로 구성하는게 조금 번거롭지만 유효하다.
# 
# - 범주형 변수 변환
#      - 원핫인코딩(One-hot-encoding), 더미코딩(dummy coding), 이펙트코딩(Effect coding), 숫자로 표현된 범주형 특성, 레이블인코딩(Label encoding), 특징 해싱(Feature Hashing), 빈도인코딩(Frequency encoding)
# 

# ### 범주형 변수 변환 - 1) 원핫인코딩 (One-hot-encoding) with get_dummies, OneHotEncoder, ColumnTransformer
#  One-out-of-N encoding, 가변수(dummy variable)라고도 한다. 범주형 변수를 0 또는 1 값을 가진 하나 이상의 새로운 특성으로 바꾼 것이다. 0과 1로 표현된 변수는 선형 이진 분류 공식에 적용할 수 있어서 개수에 상관없이 범주마다 하나의 특성으로 표현한다.
# 
# 원핫인코딩은 통계학의 dummy coding과 비슷하지만 완전히 같지는 않다. 간편하게 하려고 각 범주를 각기 다른 이진 특성으로 바꾸었기 때문이다. 이는 분석의 편리성 (데이터 행렬의 랭크 부족 현상을 피하기 위함) 때문이다.
# 
# 훈련데이터와 테스트데이터 모두를 포함하는 df를 사용해서 get_dummies 함수를 호출하든지 또는 각각 get_dummies를 호출한 후에 훈련 세트와 테스트 세트의 열이름을 비교해서 같은 속성인지를 확인해야 한다.
# 
# 특징의 개수가 범주형 변수의 레벨 개수에 따라 증가하기 때문에 정보가 적은 특징이 대량 생성돼서 학습에 필요한 계산 시간이나 메모리가 급증한다. 따라서 범주형 변수의 레벨이 너무 많을 때는 다른 인코딩 방법을 검토하거나 범주형 변수의 레벨 개수를 줄이거나 빈도가 낮은 범주를 기타 범주로 모아 정리하는 방법을 써야 한다.
# 
# 구현이 쉽고 가장 정확하며 온라인 학습이 가능한 반면, 계산 측면에서 비효율적이고 범주 수가 증가하는 경우에 적합하지 않고, 선형 모델 외에는 적합하지 않으며, 대규모 데이터셋일 경우 대규모 분산 최적화가 필요하다.
# 
# pandas의 get_dummies(데이터) 함수를 사용하거나 scikit learn의 OneHotEncoder 혹은 ColumnTransformer를 사용할 수 있다.
# 
# - OneHotEncoder는 모든 특성을 범주형이라고 가정하여 수치형 열을 포함한 모든 열에 인코딩을 수행한다. 문자열 특성과 정수 특성이 모두 변환되는 것이다.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/benny/Desktop/datascience/heart.csv', na_values=['','NA',-1,9999])
df.info()


# In[4]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) # sparse=True면 DataFrame 반환
print(ohe.fit_transform(df))
print(ohe.get_feature_names())


# In[6]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
ct = ColumnTransformer([
    ('scaling', StandardScaler(), ['Age', 'RestingBP', 'Cholesterol', 'FastingBS',
                                 'MaxHR','Oldpeak']),
    ('onehot', OneHotEncoder(sparse=False), ['Sex', 'ChestPainType', 'RestingECG',
                                            'ExerciseAngina', 'ST_Slope', 'HeartDisease'])
    ])
OC = ct.fit_transform(df)
OC


# ### 범주형 변수 변환 - 2) 더미코딩 (Dummy coding) with get_dummies(drop_first=True)
# 더미코딩은 pandas의 get_dummies 함수에서 파라미터 drop_first=True를 설정함으로써 구현할 수 있다.
# 
# 범주형 변수의 레벨이 n개일때 해당 레벨 개수만큼 가변수를 만들면 다중공선성이 생기기 때문에 이를 방지하기 위해 n-1개의 가변수를 만드는 방법을 쓰는 것이 더미코딩이다.

# In[7]:


dummies = pd.get_dummies(df)
print(list(df.columns), df.shape)
print(list(dummies.columns), dummies.shape) # 각 범주형 특성의 값마다 새로운 특성이 됨


# In[8]:


pd.get_dummies(df, drop_first=True).shape


# ### 범주형 변수 변환 - 3) 이펙트코딩 (Effect coding)
# 통계학에서 나온 범주형 변수에 대한 또 다른 변형이다. 더미코딩과 유사하지만 기준 범주가 모두 -1의 벡터로 표현된다는 것이 차이점이다. 선형 회귀 모델의 결과를 해석하기가 더 쉽다. 이펙트 코딩에서는 기준 범주를 나타내는 단일 feature가 없기 때문에 기준 범주의 효과는 다른 모든 범주의 계수의 음수 합계로서 별도로 계산해야 한다.
# 
# 여러 개의 범주형 변수를 모델에서 다룬다면 이펙트 코딩이든 더미 코딩이든 큰 차이가 없지만, 두 개의 범주형 변수가 상호작용이 있는 경우에는 이펙트 코딩이 더 이점을 가진다. 이펙트 코딩으로 합리적인 주효과와 상호작용의 추정치를 얻을 수 있다. 더미코딩의 경우, 상호작용 추정치는 괜찮지만 주효과는 진짜 주효과가 아니라 simple effect에 더 가깝다.

# ### 범주형 변수 변환 - 4) 숫자로 표현된 범주형 특성 with get_dummies
# 데이터 취합 방식에 따라 범주형 변수인데 숫자로 인코딩된 경우가 많다. 예를 들어 문자열이 아닌 답안 순서대로 0~8까지의 숫자로 채워지는 설문응답 데이터가 있다. 이 값은 이산적이기 때문에 연속형 변수로 다루면 안된다.
# 
# 숫자 특성도 가변수로 만들고 싶다면 get_dummies를 사용하여 아래와 같이 적용하면 된다.
# 
#     - get_dummies(columns=[숫자 특성도 포함하여 인코딩하려는 열을 나열])
#     - 데이터프레임 단에서 숫자 특성을 str 속성으로 변경해준 뒤 get_dummies 진행

# In[9]:


df_ = df.copy()
df_['HeartDisease'] = df_['HeartDisease'].astype(str)
dummies2 = pd.get_dummies(df_)
print(list(dummies2.columns), dummies.shape)


# ### 범주형 변수 변환 - 5) 레이블 인코딩 (Label encoding) with LabelEncoder
# 각 레벨을 단순히 정수로 변환하는 방법이다. Ordinal encoding이라고도 한다. 5개의 레벨이 있는 범주형 변수는 각 레벨이 0~4까지의 수치로 바뀐다.
# 
# 사전 순으로 나열했을 때의 인덱스 수치는 대부분 본질적인 의미가 없다. 따라서 결정 트리 모델에 기반을 둔 방법이 아닐 경우 레이블 인코딩으로 변환한 특징을 학습에 직접 이용하는 건 그다지 적절하지 않다. 결정트리에서는 범주형 변수의 특정 레벨만 목적 변수에 영향을 줄 때도 분기를 반복함으로써 예측값에 반영할 수 있으므로 학습에 활용할 수 있다.
# 
# GBDT모델에서 레이블 인코딩은 범주형 변수를 변환하는 기본적인 방법이다.

# In[11]:


from sklearn.preprocessing import LabelEncoder
LEdf = pd.DataFrame()
for col in df.columns:
    le = LabelEncoder()
    le.fit(df[col])
    LEdf[col]=le.transform(df[col])
LEdf


# In[12]:


df


# ### 범주형 변수 변환 - 6) 특징 해싱 (Feature Hashing) with FeatureHasher
# 원핫인코딩으로 변환한 뒤 특징의 수는 범주의 레벨 수와 같아지는데 특징 해싱은 그 수를 줄이는 변환방법이다. 변환 후의 특징 수를 먼저 정해두고(파라미터 n_features) 해시 함수를 이용하여 레벨별로 플래그를 표시할 위치를 경정한다.
# 
# 원핫인코딩에서는 레벨마다 서로 다른 위치에 플래그를 표시하지만 특징 해싱에서는 변환 후에 정해진 특징 수가 범주의 레벨 수보다 적으므로 해시 함수에 따른 계산에 의해 다른 레벨에서도 같은 위치에 플래그를 표시할 수 있다.
# 
# 구현하기 쉽고, 모델 학습에 비용이 적게 들며, 새로운 범주 추가가 쉽고, 희귀 범주 처리가 쉽고, 온라인 학습이 가능한 장점을 가지고 있다. 반면, 선형 또는 커널 모델에만 적합하고 해시된 feature는 해석이 불가하며 정확도에 대해 엇갈린 보고가 있다.
# 
# Scikit Learn의 FeatureHasher 함수로 각 열을 대상으로 

# In[17]:


from sklearn.feature_extraction import FeatureHasher
FHdf = pd.DataFrame(None)
for col in df.columns:
    fh = FeatureHasher(n_features=3, input_type='string')
    hash_df = fh.fit_transform(df[[col]].astype(str).values)
    hash_df = pd.DataFrame(hash_df.todense(), columns=[f'{col}_{i}' for i in range(3)])
    FHdf = pd.concat([FHdf, hash_df], axis=1)
FHdf


# ### 범주형 변수 변환 - 7) 빈도 인코딩 (Frequency Encoding) with value_counts, map
# 각 레벨의 출현 횟수 혹은 출현 빈도로 범주형 변수를 대체하는 방법이다. 각 레벨의 출현 빈도와 목적변수 간에 관련성이 있을 때 유효하다.
# 
# 레이블 인코딩의 변형으로서 사전순으로 나열한 순서 인덱스가 아닌 출현 빈도순으로 나열하는 인덱스를 만들기 위해 사용할 수도 있다. 동률의 값이 발생할 수 있으니 주의해야 한다. 또한, 수치형 변수 스케일링과 마찬가지로 학습데이터와 테스트 데이터를 따로따로 정의하여 변환해버리면 다른 의미의 변수가 되므로 조심해야 한다.

# In[18]:


FEdf = df.copy()
for col in FEdf.columns:
    freg = FEdf[col].value_counts()
    FEdf[col] = FEdf[col].map(freg)
FEdf


# In[ ]:




