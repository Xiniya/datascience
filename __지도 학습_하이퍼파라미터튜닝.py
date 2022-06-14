#!/usr/bin/env python
# coding: utf-8

# # 데이터마이닝
# ## 1. 지도학습
# ### 1. 의사결정나무: CART, C5.0, C4.5, CHAID, 분리기준 (카이제곱통계량 p값, 지니 지수, 엔트로피 지수, F통계량, 분산의 감소량), 가지치기
# ### 2. 앙상블분석: 배깅, 부스팅(Adaboost), 랜덤포레스트, 스태킹, 엑스트라트리, 에이다부스트
# ### 3. 인공신경망
#     - 활성화함수(계단함수, 부호함수, 시그모이드 함수, relu함수, softmax 함수)
#     - 다층퍼셉트론
#     - ANN, DNN, CNN, RNN, GAN(InfoGAN, CycleGAN), RBM, DBN
#     - MLP-CNN-RNN 구현 및 비교
#     - ResNet, DenseNet
#     - AutoEncoder, VAE, DQN
#     - 진화 학습 (유전 알고리즘)
#     - 강화학습 (마르코프 결정과정)
#     - 대칭가중치와 심층신뢰 네트워크
# ### 4. 회귀분석
#     - 가정검토(선형성, 등분산성-잔차도, 정규성-히스토그램/QQplot/Shapiro-wilk, 오차항의 독립성-더빈왓슨검정)
#     - 단순선형회귀분석(회귀계수 검정, 결정계수 계산-SST/SSR/SSE, 회귀직선의 적합도 검토)
#     - 다중선형회귀분석(회귀계수 검정, 회귀식, 결정계수 계산, 모형의 통계적 유의성, 교호작용, 다중공선성-PCA회귀, VIF 상위변수 제거)
#     - 다항회귀분석
#     - 스플라인 회귀
#     - 로지스틱 회귀
#     - 최적회귀방정식(전진선택법, 후진제거법, 단계적선택법 - AIC/BIC)
#     - 정규화 선형회귀 Regularized Linear Regression (Ridge회귀, Lasso회귀, Elastic Net 회귀)
#     - 일반화 선형회귀 Generalized Linear Regression
#     - 회귀분석의 기울기에 영향을 주는 영향점 진단: Cook's Distance, DFBETAS, DFFITS, Leverage H
#     - 변수 선택의 기준: 결정계수, Mallow's Cp, AIC/BIC
# ### 5. 최근접 이웃법 (KNN), 가우시안 혼합모델
# ### 6. 베이지안 분류
# ### 7. SVM
# ### 8. 판별분석
# ### 9. 사례기반 추론 (Case based reasoning)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('/Users/benny/Desktop/datascience/heart.csv')
df_dummy = pd.get_dummies(df)
X = df_dummy.drop('HeartDisease', axis=1)
y = df_dummy['HeartDisease']
from sklearn.model_selection import train_test_split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1993)
print(X_train_clf.shape, X_test_clf.shape)


# In[ ]:


from sklearn.datasets import load_boston
df = load_boston()
X = pd.DataFrame(df['data'], columns=df['feature_names'])
y = pd.Series(df['target'], name='MEDV')

from sklearn.model_selection import train_test_split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.2, random_state=1993)
print(X_train_reg.shape, X_test_reg.shape)


# ### k-최근접 이웃 (k-NN, k-Nearest Neighbors) - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


estimator = KNeighborsClassifier()

param_grid = {'n_neighbors':[1,2,3,4,5], 'metric':['minkowski','manhattan','euclidean'], 'weights':['uniform','distance']}

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='accuracy') #디폴트로 cv=3, 분류에서 디폴트로 scoring='accuracy'

grid.fit(X_train_clf, y_train_clf)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_n_neighbors'))
#print(df.sort_values(by='param_n_neighbors', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_

# y_pred = estimator.decision_function(X_train_clf)[:2]
y_prob = estimator.predict_proba(X_train_clf[:2]) #predict_proba is not available when  probability=False
y_predict = estimator.predict(X_train_clf[:2])
print(y_prob, '\n', y_predict)

score = grid.score(X_test_clf, y_test_clf)
print(score)


# ### k-최근접 이웃 (k-NN, k-Nearest Neighbors) - 회귀 - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

estimator = KNeighborsRegressor()

param_grid = {'n_neighbors':[1,2,3,4,5], 'metric':['minkowski','manhattan','euclidean'], 'weights':['uniform','distance']}

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='r2') #디폴트로 cv=3, 회귀에서 디폴트로 scoring='r2'

grid.fit(X_train_reg, y_train_reg)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_n_neighbors'))
#print(df.sort_values(by='param_n_neighbors', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
'''
#estimator = KNeighborsRegressor(**grid.best_params_)
estimator = KNeighborsRegressor()
estimator.set_params(**grid.best_params_)

estimator.fit(x_data, y_data)
'''

y_pred = estimator.predict(X_train_reg[:2])
print(y_pred)

score = grid.score(X_test_reg, y_test_reg)
print(score)


# ### 서포트 벡터 머신 (Support Vector Machine) - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

estimator = SVC(probability=True)

param_grid = {'kernel':['rbf'], 'C':[1,100,10,0.1,0.01,0.001]}
#param_grid = [ 
#  {'kernel':['linear'], 'C':[1,100,10,0.1,0.01,0.001]}, #특정 하이퍼 파라메타 조합 피하기
#  {'kernel':['poly','rbf'], 'C':[1,100,10,0.1,0.01,0.001], 'gamma':['auto','scale',1000,100,10,1,0.1,0.01,0.001,0.0001]}
#]

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='accuracy') #디폴트로 cv=3, 분류에서 디폴트로 scoring='accuracy'

grid.fit(X_train_clf, y_train_clf)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_C'))
#print(df.sort_values(by='param_C', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
'''
#estimator = SVC(**grid.best_params_)
estimator = SVC()
estimator.set_params(**grid.best_params_)

estimator.fit(x_data, y_data)
'''

y_pred = estimator.decision_function(X_train_clf)[:2]
y_prob = estimator.predict_proba(X_train_clf[:2]) #predict_proba is not available when  probability=False
y_predict = estimator.predict(X_train_clf[:2])
print(y_prob, '\n', y_pred, '\n', y_predict)

score = grid.score(X_test_clf, y_test_clf)
print(score)


# ### 서포트 벡터 머신 (Support Vector Machine) - 회귀 - 하이퍼 파라미터 찾기
# 

# In[ ]:


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

estimator = SVR()

param_grid = {'kernel':['rbf'], 'C':[1,100,10,0.1,0.01,0.001]}
'''
param_grid = [ 
  {'kernel':['linear'], 'C':[1,100,10,0.1,0.01,0.001]}, #특정 하이퍼 파라메타 조합 피하기
  {'kernel':['poly','rbf'], 'C':[1,100,10,0.1,0.01,0.001], 'gamma':['auto','scale',1000,100,10,1,0.1,0.01,0.001,0.0001]}
]
'''

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='r2') #디폴트로 cv=3, 회귀에서 디폴트로 scoring='r2'

grid.fit(X_train_reg, y_train_reg)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_n_neighbors'))
#print(df.sort_values(by='param_n_neighbors', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
'''
#estimator = SVR(**grid.best_params_)
estimator = SVR()
estimator.set_params(**grid.best_params_)

estimator.fit(x_data, y_data)
'''

y_pred = estimator.predict(X_train_reg[:2])
print(y_pred)

score = grid.score(X_test_reg, y_test_reg)
print(score)


# ### 의사 결정 나무 (Decision Tree) - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

estimator = DecisionTreeClassifier()

#param_grid = {'criterion':['gini'], 'max_depth':[None,2,3,4,5,6]}
param_grid = {'criterion':['gini','entropy'], 'max_depth':[None,2,3,4,5,6], 'max_leaf_nodes':[None,2,3,4,5,6,7], 'min_samples_split':[2,3,4,5,6], 'min_samples_leaf':[1,2,3], 'max_features':[None,'sqrt','log2',3,4,5]}

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='accuracy') #디폴트로 cv=3, 분류에서 디폴트로 scoring='accuracy'

grid.fit(X_train_clf, y_train_clf)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_max_depth'))
#print(df.sort_values(by='param_max_depth', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
'''
#estimator = DecisionTreeClassifier(**grid.best_params_)
estimator = DecisionTreeClassifier()
estimator.set_params(**grid.best_params_)

estimator.fit(x_data, y_data)
'''

# y_pred = estimator.decision_function(X_train_clf)[:2]
y_prob = estimator.predict_proba(X_train_clf[:2]) #predict_proba is not available when  probability=False
y_predict = estimator.predict(X_train_clf[:2])
print(y_prob, '\n', y_predict)

score = grid.score(X_test_clf, y_test_clf)
print(score)


# ### 의사 결정 나무 (Decision Tree) - 회귀 - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

estimator = DecisionTreeRegressor()

param_grid = {'criterion':['mse'], 'max_depth':[None,2,3,4,5,6]}
#param_grid = {'criterion':['mse','friedman_mse','mae'], 'max_depth':[None,2,3,4,5,6], 'max_leaf_nodes':[None,2,3,4,5,6,7], 'min_samples_split':[2,3,4,5,6], 'min_samples_leaf':[1,2,3], max_features:[None,'sqrt','log2',3,4,5]}

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='r2') #디폴트로 cv=3, 회귀에서 디폴트로 scoring='r2'

grid.fit(X_train_reg, y_train_reg)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_max_depth'))
#print(df.sort_values(by='param_max_depth', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
'''
#estimator = KNeighborsRegressor(**grid.best_params_)
estimator = KNeighborsRegressor()
estimator.set_params(**grid.best_params_)

estimator.fit(x_train, y_train)
'''

y_pred = estimator.predict(X_train_reg[:2])
print(y_pred)

score = grid.score(X_test_reg, y_test_reg)
print(score)


# ### 앙상블 배깅 - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV

base_estimator = DecisionTreeClassifier()
estimator = BaggingClassifier(base_estimator=base_estimator)

#'''
param_grid = {'n_estimators':[10,1,2,3,4,5,6,7,8,9]}
#'''
'''
param_grid = {
    'n_estimators':[10,1,2,3,4,5,6,7,8,9], 
    'bootstrap':[True,False],
    'base_estimator__criterion':['gini','entropy'], 'base_estimator__max_depth':[None,2,3,4,5,6], 'base_estimator__max_leaf_nodes':[None,2,3,4,5,6,7], 'base_estimator__min_samples_split':[2,3,4,5,6], 'base_estimator__min_samples_leaf':[1,2,3], 'base_estimator__max_features':[None,'sqrt','log2',5,6,7,8,9,10]
}
'''

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='accuracy') #디폴트로 cv=3, 분류에서 디폴트로 scoring='accuracy'

grid.fit(X_train_clf, y_train_clf)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_n_estimators'))
#print(df.sort_values(by='param_n_estimators', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
'''
#estimator = BaggingClassifier(base_estimator=base_estimator, **grid.best_params_)
estimator = BaggingClassifier(base_estimator=base_estimator)
estimator.set_params(**grid.best_params_)

estimator.fit(x_data, y_data)
'''

# y_pred = estimator.decision_function(X_train_clf)[:2]
y_prob = estimator.predict_proba(X_train_clf[:2]) #predict_proba is not available when  probability=False
y_predict = estimator.predict(X_train_clf[:2])
print(y_prob, '\n', y_predict)

score = grid.score(X_test_clf, y_test_clf)
print(score)


# ### 앙상블 부스팅 - 그레디언트 부스팅 - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import GridSearchCV

estimator = GradientBoostingClassifier()

param_grid = {'n_estimators':[100,90,91,92,93,94,95,96,97,98,99]}
#param_grid = {'n_estimators':[100,90,91,92,93,94,95,96,97,98,99], 'criterion':['friedman_mse','mse','mae'], 'max_depth':[3,4,5,6], 'max_leaf_nodes':[None,2,3,4,5,6,7], 'min_samples_split':[2,3,4,5,6], 'min_samples_leaf':[1,2,3], 'max_features':[None,'sqrt','log2',5,6,7,8,9,10]}

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='accuracy') #디폴트로 cv=3, 분류에서 디폴트로 scoring='accuracy'

grid.fit(X_train_clf, y_train_clf)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_alpha'))
#print(df.sort_values(by='param_alpha', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
'''
#estimator = GradientBoostingClassifier(**grid.best_params_)
estimator = GradientBoostingClassifier()
estimator.set_params(**grid.best_params_)

estimator.fit(x_data, y_data)
'''

# y_pred = estimator.decision_function(X_train_clf)[:2]
y_prob = estimator.predict_proba(X_train_clf[:2]) #predict_proba is not available when  probability=False
y_predict = estimator.predict(X_train_clf[:2])
print(y_prob, '\n', y_predict)

score = grid.score(X_test_clf, y_test_clf)
print(score)


# ### 앙상블 부스팅 - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

base_estimator = DecisionTreeClassifier()
estimator = AdaBoostClassifier(base_estimator=base_estimator)

param_grid = {'n_estimators':[10,1,2,3,4,5,6,7,8,9]}
'''
param_grid = {
    'n_estimators':[10,1,2,3,4,5,6,7,8,9], 
    'base_estimator__criterion':['gini','entropy'], 'base_estimator__max_depth':[None,2,3,4,5,6], 'base_estimator__max_leaf_nodes':[None,2,3,4,5,6,7], 'base_estimator__max_features':[None,1], 'base_estimator__min_samples_split':[2,3,4,5,6], 'base_estimator__min_samples_leaf':[1,2,3], 'base_estimator__max_features':[None,'sqrt','log2',5,6,7,8,9,10]
}
'''

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='accuracy') #디폴트로 cv=3, 분류에서 디폴트로 scoring='accuracy'

grid.fit(X_train_clf, y_train_clf)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_n_estimators'))
#print(df.sort_values(by='param_n_estimators', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
'''
#estimator = AdaBoostClassifier(base_estimator=base_estimator, **grid.best_params_)
estimator = AdaBoostClassifier(base_estimator=base_estimator)
estimator.set_params(**grid.best_params_)

estimator.fit(x_data, y_data)
'''

# y_pred = estimator.decision_function(X_train_clf)[:2]
y_prob = estimator.predict_proba(X_train_clf[:2]) #predict_proba is not available when  probability=False
y_predict = estimator.predict(X_train_clf[:2])
print(y_prob, '\n', y_predict)

score = grid.score(X_test_clf, y_test_clf)
print(score)


# ### 앙상블 배깅 - 랜덤 포레스트 - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

estimator = RandomForestClassifier()

param_grid = {'n_estimators':[10,1,2,3,4,5,6,7,8,9]}
#param_grid = {'n_estimators':[10,1,2,3,4,5,6,7,8,9], 'bootstrap':[True,False], 'criterion':['gini','entropy'], 'max_depth':[None,2,3,4,5,6], 'max_leaf_nodes':[None,2,3,4,5,6,7], 'min_samples_split':[2,3,4,5,6], 'min_samples_leaf':[1,2,3], 'max_features':[None,'sqrt','log2',5,6,7,8,9,10]}

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='accuracy') #디폴트로 cv=3, 분류에서 디폴트로 scoring='accuracy'

grid.fit(X_train_clf, y_train_clf)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_n_estimators'))
#print(df.sort_values(by='param_n_estimators', ascending=0))
#print(df.sort_values(by='rank_test_score'))

#'''
estimator = grid.best_estimator_
#'''
'''
#estimator = RandomForestClassifier(**grid.best_params_)
estimator = RandomForestClassifier()
estimator.set_params(**grid.best_params_)

estimator.fit(x_data, y_data)
'''

# y_pred = estimator.decision_function(X_train_clf)[:2]
y_prob = estimator.predict_proba(X_train_clf[:2]) #predict_proba is not available when  probability=False
y_predict = estimator.predict(X_train_clf[:2])
print(y_prob, '\n', y_predict)

score = grid.score(X_test_clf, y_test_clf)
print(score)


# ### 앙상블 다수결 투표 - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

estimator = VotingClassifier(estimators=[('lr', LogisticRegression()),
                                         ('kn', KNeighborsClassifier()),
                                         ('dt', DecisionTreeClassifier())])

param_grid = {'voting':['hard','soft']}
'''
param_grid = {
  'voting':['hard','soft'], 
  'weights':[[2,1,1],[1,2,1]],
    'kn__n_neighbors':[5,1,2,3,4], 'kn__metric':['minkowski','manhattan','euclidean'], 'kn__weights':['uniform','distance'],
    'dt__criterion':['gini','entropy'], 'dt__max_depth':[None,2,3,4,5,6], 'dt__max_leaf_nodes':[None,2,3,4,5,6,7], 'dt__max_features':[None,1], 'dt__min_samples_split':[2,3,4,5,6], 'dt__min_samples_leaf':[1,2,3]
}
'''

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='accuracy') #디폴트로 cv=3, 분류에서 디폴트로 scoring='accuracy'

grid.fit(X_train_clf, y_train_clf)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_voting'))
#print(df.sort_values(by='param_voting', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
'''
estimator = VotingClassifier(estimators=[('lr', LogisticRegression()),
                                         ('kn', KNeighborsClassifier()),
                                         ('dt', DecisionTreeClassifier())])
estimator.set_params(**grid.best_params_)

estimator.fit(x_data, y_data)
'''

# y_pred = estimator.decision_function(X_train_clf)[:2]
# y_prob = estimator.predict_proba(X_train_clf[:2]) #predict_proba is not available when  probability=False
y_predict = estimator.predict(X_train_clf[:2])
print(y_predict)

score = grid.score(X_test_clf, y_test_clf)
print(score)


# ### 앙상블 배깅 - 회귀 - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV

base_estimator = DecisionTreeRegressor()
estimator = BaggingRegressor(base_estimator=base_estimator)

param_grid = {'n_estimators':[10,1,2,3,4,5,6,7,8,9]}
'''
param_grid = {
    'n_estimators':[10,1,2,3,4,5,6,7,8,9],
    'bootstrap':[True,False],
    'base_estimator__criterion':['gini','entropy'], 'base_estimator__max_depth':[None,2,3,4,5,6], 'base_estimator__max_leaf_nodes':[None,2,3,4,5,6,7], 'base_estimator__min_samples_split':[2,3,4,5,6], 'base_estimator__min_samples_leaf':[1,2,3], 'base_estimator__max_features':[None,'sqrt','log2',5,6,7,8,9,10]
}
'''

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='accuracy') #디폴트로 cv=3, 분류에서 디폴트로 scoring='accuracy'

grid.fit(X_train_reg, y_train_reg)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_n_estimators'))
#print(df.sort_values(by='param_n_estimators', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
'''
#estimator = BaggingRegressor(base_estimator=base_estimator, **grid.best_params_)
estimator = BaggingRegressor(base_estimator=base_estimator)
estimator.set_params(**grid.best_params_)

estimator.fit(x_data, y_data)
'''

y_pred = estimator.predict(X_train_reg[:2])
print(y_pred)

score = grid.score(X_test_reg, y_test_reg)
print(score)


# ### 앙상블 배깅 - 랜덤 포레스트 - 회귀 - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import GridSearchCV

estimator = RandomForestRegressor()

param_grid = {'n_estimators':[10,1,2,3,4,5,6,7,8,9]}
#param_grid = {'n_estimators':[10,1,2,3,4,5,6,7,8,9], 'bootstrap':[True,False], 'criterion':['mse','friedman_mse','mae'], 'max_depth':[None,2,3,4,5,6], 'max_leaf_nodes':[None,2,3,4,5,6,7], 'min_samples_split':[2,3,4,5,6], 'min_samples_leaf':[1,2,3], 'max_features':[None,'sqrt','log2',5,6,7,8,9,10]}

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='accuracy') #디폴트로 cv=3, 분류에서 디폴트로 scoring='accuracy'

grid.fit(X_train_reg, y_train_reg)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_n_estimators'))
#print(df.sort_values(by='param_n_estimators', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
'''
#estimator = RandomForestRegressor(**grid.best_params_)
estimator = RandomForestRegressor()
estimator.set_params(**grid.best_params_)

estimator.fit(x_data, y_data)
'''

y_pred = estimator.predict(X_train_reg[:2])
print(y_pred)

score = grid.score(X_test_reg, y_test_reg)
print(score)


# ### 앙상블 부스팅 - 그레디언트 부스팅 - 회귀 - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import GridSearchCV

estimator = GradientBoostingRegressor()

param_grid = {'n_estimators':[100,90,91,92,93,94,95,96,97,98,99]}
#param_grid = {'n_estimators':[100,90,91,92,93,94,95,96,97,98,99], 'criterion':['friedman_mse','mse','mae'], 'max_depth':[3,4,5,6], 'max_leaf_nodes':[None,2,3,4,5,6,7], 'min_samples_split':[2,3,4,5,6], 'min_samples_leaf':[1,2,3], max_features:[None,'sqrt','log2',5,6,7,8,9,10]}

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='accuracy') #디폴트로 cv=3, 분류에서 디폴트로 scoring='accuracy'

grid.fit(X_train_reg, y_train_reg)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_n_estimators'))
#print(df.sort_values(by='param_n_estimators', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
'''
#estimator = GradientBoostingRegressor(**grid.best_params_)
estimator = GradientBoostingRegressor()
estimator.set_params(**grid.best_params_)

estimator.fit(x_data, y_data)
'''

y_pred = estimator.predict(X_train_reg[:2])
print(y_pred)

score = grid.score(X_test_reg, y_test_reg)
print(score)


# ### 앙상블 부스팅 - 회귀 - 하이퍼 파라미터 찾기

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

base_estimator = DecisionTreeRegressor()
estimator = AdaBoostRegressor(base_estimator=base_estimator)

param_grid = {'n_estimators':[10,1,2,3,4,5,6,7,8,9]}
'''
param_grid = {
    'n_estimators':[10,1,2,3,4,5,6,7,8,9], 
    'base_estimator__criterion':['gini','entropy'], 'base_estimator__max_depth':[None,2,3,4,5,6], 'base_estimator__max_leaf_nodes':[None,2,3,4,5,6,7], 'base_estimator__min_samples_split':[2,3,4,5,6], 'base_estimator__min_samples_leaf':[1,2,3], 'base_estimator__max_features':[None,'sqrt','log2',5,6,7,8,9,10]
}
'''

grid = GridSearchCV(estimator, param_grid=param_grid) 
#grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='accuracy') #디폴트로 cv=3, 분류에서 디폴트로 scoring='accuracy'

grid.fit(X_train_reg, y_train_reg)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_n_estimators'))
#print(df.sort_values(by='param_n_estimators', ascending=0))
#print(df.sort_values(by='rank_test_score'))

estimator = grid.best_estimator_
'''
#estimator = AdaBoostRegressor(base_estimator=base_estimator, **grid.best_params_)
estimator = AdaBoostRegressor()
estimator.set_params(**grid.best_params_)

estimator.fit(x_data, y_data)
'''

y_pred = estimator.predict(X_train_reg[:2])
print(y_pred)

score = grid.score(X_test_reg, y_test_reg)
print(score)


# In[ ]:




