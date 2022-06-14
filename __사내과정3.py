#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
os.chdir(r"D:\강의 자료\2011_신한카드_Data Scientist 심화 과정 3기\프로젝트 데이터")

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("자전거 대여량 예측 데이터_학습.csv", engine = "python")
df.set_index('ID', inplace = True)


# In[3]:


X = df.drop(['Rented Bike Count'], axis = 1)
Y = df['Rented Bike Count']

from sklearn.model_selection import train_test_split
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y)


# In[4]:


Train_X.shape


# In[5]:


Train_X.describe()


# In[6]:


# 결측이 없음을 확인
Train_X.isnull().sum(axis = 0)


# In[7]:


# Date 변수에서 시간, 월 변수 등을 추출할 수 있지만, 이미 기상 정보가 있어서 반영되어 있다고 판단
Train_X.drop('Date', axis = 1, inplace = True)


# In[8]:


Test_X.drop('Date', axis = 1, inplace = True)


# In[9]:


# 범주형과 연속형 변수 구분
cate_cols = []
cont_cols = []
not_sure_cols = []

for col in Train_X.columns:
    size_of_state_space = len(Train_X[col].unique()) # 상태공간 크기 정의
    col_type = Train_X[col].dtype # 컬럼의 데이터 타입 확인
    
    if size_of_state_space > 10 and col_type in [int, float, 'int64', 'int32', 'int8', 'float64', 'float16']: # 상태공간의 크기가 10 초과이고, 데이터 타입이 int나 float
        cont_cols.append(col)
    elif size_of_state_space < 10 and col_type == object: # 상태공간의 크기가 10 미만이고, 데이터타입이 object
        cate_cols.append(col)        
    else: # 위의 두 조건을 만족못하면 더 확인해보기!
        not_sure_cols.append(col)    
        
print(cate_cols)
print(cont_cols)
print(not_sure_cols) # 확실하지 않은 변수가 없음


# In[10]:


# 주의: 시간 변수는 연속형으로 절대 간주할 수 없음
# 더미화를 하기 위해서는 범주형으로 바꿔줘야 함
Train_X['Hour'] = Train_X['Hour'].astype(str)
cate_cols.append('Hour')


# In[11]:


# 더미화
from feature_engine.categorical_encoders import OneHotCategoricalEncoder as OHE
dummy_model = OHE(variables = cate_cols, drop_last = True).fit(Train_X)
Train_X = dummy_model.transform(Train_X)
Test_X = dummy_model.transform(Test_X)


# In[12]:


# 스케일링 수행
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(Train_X)
Train_X = pd.DataFrame(scaler.transform(Train_X),
                      columns = Train_X.columns,
                      index = Train_X.index) 

Test_X = pd.DataFrame(scaler.transform(Test_X),
                      columns = Test_X.columns,
                      index = Test_X.index) 


# In[13]:


Train_X.info() # 더미화로 인해 이진형 변수가 크게 늘어남


# #### 모델 선택
# - 이진형 변수와 연속형 변수가 다수 섞임 --> 의사결정나무
# - 미래를 예측하는 상황 --> 신경망

# In[22]:


from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.neural_network import MLPRegressor as NN
from sklearn.feature_selection import *


# In[19]:


# 의사결정나무의 파라미터 그리드 정의
tree_grid = ParameterGrid({"max_depth": [3, 4, 5, 6, 7],
                           "min_samples_leaf": [1, 2, 3, 4, 5]})

NN_grid = ParameterGrid({"max_iter": [100000],
                         "hidden_layer_sizes":[(5,), (10, 10), (20, 20), (10, 10, 10), (30, 30)],
                        "random_state":[1, 2, 3]})

num_feature_grid = list(range(5, 36, 5))


# In[20]:


model_grid = {DT:tree_grid, NN:NN_grid}


# In[ ]:


# 지금까지 찾은 최고 점수 (best_score) 초기화
best_score = 999999999999999

for num_feature in num_feature_grid:
    print(num_feature)
    # 특징 개수가 num_feature인 selector 학습
    selector = SelectKBest(f_classif, k = num_feature).fit(Train_X, Train_Y)
    # 선택된 특징 정의: selected_features
    selected_features = Train_X.columns[selector.get_support()]

    for model_func in model_grid.keys():
        for parameter in model_grid[model_func]:
            model = model_func(**parameter).fit(Train_X.loc[:, selected_features], Train_Y)
            pred_Y = model.predict(Test_X.loc[:, selected_features])
            score = MAE(Test_Y, pred_Y)

            if score < best_score:
                best_score = score
                best_model = model
                best_feature = selected_features


# In[ ]:


print(best_score) # 최고 점수 확인
print(best_model) # 최고 모델 확인
print(len(best_feature)) # 최고 특징 집합의 길이 확인


# In[ ]:


def pipeline(X_cols, model, dummy_model, scaler, features, new_X):
    new_X.set_index('CompanyID', inplace = True)    
    new_X = new_X.loc[:, X_cols]    
    new_X = dummy_model.transform(new_X)   
    new_X = pd.DataFrame(scaler.transform(new_X),
                         columns = X_cols,
                         index = new_X.index)    
    
    new_X = new_X.loc[:, features]
    return pd.Series(model.predict(new_X), index = new_X.index)


# In[ ]:


new_X = pd.read_csv("자전거 대여량 예측 데이터_평가.csv", engine = "python")


# In[ ]:


output = pipeline(X.columns, best_model, dummy_model, scaler, best_feature, new_X)


# In[ ]:


output.to_csv("자전거 대여량 예측 데이터_예측.csv", index = True)

