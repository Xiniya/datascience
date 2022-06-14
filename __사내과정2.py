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


df = pd.read_csv("부도 예측 데이터_학습.csv")
df.set_index('CompanyID', inplace = True)


# In[3]:


X = df.drop(['bankruptcy'], axis = 1)
Y = df['bankruptcy']

from sklearn.model_selection import train_test_split
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y)


# In[4]:


Train_X.shape


# In[5]:


Train_X.describe()


# In[6]:


# 전부 float64임을 확인
pd.Series(Train_X.dtypes).value_counts()


# In[7]:


# 결측 존재
Train_X.isnull().sum(axis = 0)


# In[8]:


# 결측이 가장 많은 열은 1155개의 결측을 갖고 있음
Train_X.isnull().sum(axis = 0).sort_values()


# In[9]:


# 결측치 처리
from sklearn.impute import SimpleImputer as SI
imputer = SI().fit(Train_X)

Train_X = pd.DataFrame(imputer.transform(Train_X),
                      columns = Train_X.columns,
                      index = Train_X.index) 

Test_X = pd.DataFrame(imputer.transform(Test_X),
                      columns = Test_X.columns,
                      index = Test_X.index) 


# In[10]:


# 스케일링 수행
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(Train_X)
Train_X = pd.DataFrame(scaler.transform(Train_X),
                      columns = Train_X.columns,
                      index = Train_X.index) 

Test_X = pd.DataFrame(scaler.transform(Test_X),
                      columns = Test_X.columns,
                      index = Test_X.index) 


# #### 모델 선택
# - 전부 다 연속형에 특징이 적당히 많음
# - 신경망과 k-NN이 적절할 것이라 예상

# In[11]:


from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier as NN
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.feature_selection import *


# In[12]:


NN_grid = ParameterGrid({"max_iter": [100000],
                         "hidden_layer_sizes":[(5,), (10, 10), (20, 20), (10, 10, 10), (30, 30)],
                        "random_state":[1, 2, 3]})

KNN_grid = ParameterGrid({"metric":["euclidean", "manhattan"],
                         "n_neighbors":[1, 3, 5, 7, 9]})

num_feature_grid = list(range(5, 64, 5))


# In[13]:


model_grid = {NN:NN_grid, KNN:KNN_grid}


# In[14]:


# 지금까지 찾은 최고 점수 (best_score) 초기화
best_score = -1

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
            score = f1_score(Test_Y, pred_Y)

            if score > best_score:
                best_score = score
                best_model = model
                best_feature = selected_features


# In[15]:


print(best_score) # 최고 점수 확인
print(best_model) # 최고 모델 확인
print(len(best_feature)) # 최고 특징 집합의 길이 확인


# In[16]:


def pipeline(X_cols, model, imputer, scaler, features, new_X):
    new_X.set_index('CompanyID', inplace = True)    
    new_X = new_X.loc[:, X_cols]    
    new_X = pd.DataFrame(imputer.transform(new_X),
                         columns = X_cols,
                         index = new_X.index)
    
    new_X = pd.DataFrame(scaler.transform(new_X),
                         columns = X_cols,
                         index = new_X.index)    
    
    new_X = new_X.loc[:, features]
    return pd.Series(model.predict(new_X), index = new_X.index)


# In[17]:


new_X = pd.read_csv("부도 예측 데이터_평가.csv")


# In[18]:


output = pipeline(X.columns, best_model, imputer, scaler, best_feature, new_X)


# In[19]:


output.to_csv("부도 예측 데이터_예측.csv", index = True)

