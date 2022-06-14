#!/usr/bin/env python
# coding: utf-8

# ## 1. 현재 날짜 데이터 추출하기

# In[1]:


from datetime import datetime
datetime.today()


# In[3]:


datetime.today().year


# ## 2. 날짜 형식으로 변환하기

# In[4]:


datetime.strptime('2022-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')


# In[5]:


time = datetime.today()
time.strftime('%Y-%m-%d %H:%M:%S')


# ## 3. 날짜 데이터의 연산

# In[6]:


from datetime import timedelta
time = datetime.today()
time + timedelta(days=100)


# # 시계열분석
# ## 1. 시계열 분해
# 시계열 분해는 시계열 자료를 추세, 계절성, 잔차로 분해하는 기법
# 
# 시간의 요인은 추세, 계절성이며 외부요인은 잔차
# 
# (1) 모형 판단
# 
# 시계열 데이터를 보고 시계열의 주기적 반복/계절성이 있는 지에 따라 Additive 모형과 Multiplicative 모형 중 어떤 모형이 더 적합한 지 판단
# 
# 추세와 계절성이 별개로 존재한다면 Additive 모형을 선택하고, 추세에 따라 계절성이 있다면 Multiplicative 모형을 적용

# In[9]:


import pandas as pd
import warnings

data = pd.read_csv('/Users/benny/Desktop/datascience/print/arima_data.csv', names = ['day', 'price'])
data.head()


# In[10]:


data.info()


# In[11]:


data['day'] = pd.to_datetime(data['day'], format='%Y-%m-%d')
data.set_index('day', inplace=True)
data.head(3)


# In[15]:


import matplotlib.pyplot as plt
plt.plot(data.index, data['price'])
plt.show()

# 추세에 따라 계절성이 존재하는 것을 확인
# 시간이 지날수록 변동이 커지므로 Multiplicative 모델을 적용


# In[18]:


from statsmodels.tsa.seasonal import seasonal_decompose

ts = data
result = seasonal_decompose(ts, model = 'multiplicative')
plt.rcParams['figure.figsize'] = [12, 8]
result.plot()
plt.show()

# 해당 데이터는 Trend와 Seasonal이 명확히 존재하며, 불규칙 요인은 거의 없음을 확인


# ## 2. 정상성 변환
# 여러 가지 방법이 있지만, ARIMA 모델을 ADP 시험에서 사용하는 것을 추천
# 
# ARIMA는 AR 모형과 MA 모형을 합한 모형
# 
# ### 정상성
# (1) 개념
#     
#     정상성이란 평균, 분산이 시간에 따라 일정한 성질을 가지고 있다는 것
#     
#     시계열의 특성이 시간의 흐름에 따라 변하지 않는 상태를 의미
#     
#     비정상 시계열의 경우 ARIMA 모형을 적용시킬 수 없으므로 비정상 시계열을 정상 시계열로 변환해주어야 함
#     
#     변환의 방법은 대표적으로 로그 변환과 차분이 존재
#     
# (2) 로그 변환
# 
# (3) 차분
# 
# (4) 파이썬을 활용한 데이터 전처리
# 
#     정상성을 검정하기 위해서는 Augmented Dickey-Fuller Test를 진행
#     
#     귀무가설 : 데이터가 정상성을 갖지 않는다.
#     
#     대립가설 : 데이터가 정상성을 갖는다.
#    

# In[19]:


from statsmodels.tsa.stattools import adfuller

# Train, Test 데이터 구분
training = data[:'2016-12-01']
test = data.drop(training.index)

adf = adfuller(training, regression='ct')
print('ADF Statistic: {}'.format(adf[0]))
print('p-value: {}'.format(adf[1]))

# 우상향 트렌드를 보인 데이터이므로 ct 값을 적용하여 regression 검정 결과 p-value가 설정한 유의수준 0.05보다 높음
# 해당 데이터는 정상성을 갖지 않다고 할 수 있음
# 비정상시계열을 정상시계열로 변환시키기 위해서는 1차 차분 혹은 로그변환을 해야한다.


# In[21]:


diff_data = training.diff(1)
diff_data = diff_data.dropna()
diff_data.plot()
plt.show()


# In[23]:


adf = adfuller(diff_data)
print('ADF Statistic: {}'.format(adf[0]))
print('p-value: {}'.format(adf[1]))

# adfuller()의 regressiong default 값은 'c'
# 1차 차분한 그래프가 트렌드를 보이지 않기에 매개변수는 'c' 값을 적용하여 검정해야 한다.
# 검정 결과 p-value가 0.05보다 작으므로 귀무가설을 기각할 수 있다.


# ## 3. AR 모형과 MA 모형
# (1) AR
# 
#     1. 개념
# 
#         AR 모형은 자기회귀과정이란 뜻으로 현 시점의 데이터를 이전의 데이터들의 상관성으로 나타내는 모형
#         
#         과거의 값이 현재의 값에 얼마나 영향을 미쳤는 지를 파악하는 것
#         
#         최적의 성능을 가지는 모델을 만들 수 있는 과거의 값을 찾게 되는데, 이 값을 p라고 하며, AR(p) 모형이라 한다.
#         
#     2. ACF
#     
#         ACF는 자기상관 함수로 이 값은 시차에 따른 자기상관성을 의미
#         
#         ACF 값을 시차에 따른 그래프로 시각화를 해보면, 최적의 p 값을 찾을 수 있다.
#         
#         비정상 시계열일 경우에는 ACF 값은 느리게 0에 접근하며, 양수의 값을 가질 수도 있다.
#         
#         정상 시계열일 경우에는 ACF 값이 빠르게 0으로 수렴하며, 0으로 수렴할 때에 시차를 p 값으로 설정

# In[27]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

plot_acf(diff_data)
plt.show()

# ACF 값을 확인해 보았을 때, 시차 2 이후에 0에 수렴하는 것을 알 수 있다.
# 그러므로 AR 모형에서 최적의 p 값은 2로 설정할 수 있다.


# (2) MA
# 
#     1. 개념
#     
#         MA 모형은 과거의 예측 오차들의 가중이동평균으로 현재 시점의 데이터를 표현하는 모형
#         
#         과거의 예측 오차를 이용하여 미래를 예측하는 모형
#         
#         과거의 예측 오차들에 따라서 가중이동평균이 달라지기 때문에 MA 모형은 최적 모형이 되는 구간을 구하는 것이 중요함
#         
#         MA 모형이 최적이 되게끔 하는 변수 값이 q이며 이 모형을 MA(q) 모형이라 함
#         
#     2. PACF
#     
#         PACF는 편자기상관 함수이다. PACF는 ACF와는 다르게 시차가 다른 두 시계열 데이터 간의 순수한 상호 연관성을 나타냄
#         
#         PACF 값이 0에 수렴할 때의 q 값을 MA 모형의 q 값으로 설정

# In[28]:


plot_pacf(diff_data)
plt.show()

# PACF 값을 확인해 보았을 때, 시차 2 이후에 0에 수렴하는 것을 알 수 있다.
# 그러므로 MA 모형에서 최적의 q 값은 2로 설정할 수 있음


# ## 4. ARIMA
# (1) 개념
# 
# ARIMA는 우선 비정상적 시계열 자료에 대해 분석하는 모형이다
# 
# 변환 중 차분을 사용하여 비정상 시계열을 정상 시계열로 만든다. 그리고 정상 시계열의 경우 AR 모형과 MA 모형이 상호변환이 가능하기에 이 두 모형을 결합하여 과거의 시점의 데이터로 현재 혹은 미래의 시점의 데이터를 예측하는 모형이다.
# 
# ARIMA 모형의 파라미터로는 (p, d, q)를 사용한다. ARIMA 모형은 시계열 자료 외에 다른 자료가 없을 때, 그 변동 상태를 확인할 수 있다는 장점을 갖고 이씅며 어떠한 시계열에도 적용이 가능한 모델이라는 장점이 있다.

# In[37]:


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(training, order=(2,1,2))
res = model.fit()
res.summary()

# AIC와 AR, MA 모델의 p-value가 중요
# AIC는 다른 모델과 비교할 때 사용할 수 있으며 작을수록 성능이 좋다.
# coef에서 p-value가 0.05 이하라면, AR 모형과 MA 모형을 사용할 수 있다.
# L1, L2는 사용하느 ㄴ시차의 개념이다.
# 만약 p가 5라면 AR.L1~L5까지 변수를 모델에서 사용한다.


# In[39]:


plt.plot(res.predict())
plt.plot(training)

# 해당 모델의 학습 정도를 보기 위해, 학습된 모델인 res에서 학습시킨 데이터를 예측
# training 데이터를 학습시키고 확인하였을 때, 그래프의 모양이 거의 일치하므로 과소적합은 의심되지 않는다.


# In[40]:


forecast_data = res.forecast(steps=len(test), alpha=0.05)
# 학습 데이터 세트로부터 test 데이터 길이만큼 예측

pred_y = forecast_data
pred_y


# In[41]:


test_y = test # 실제 데이터
test_y


# In[42]:


plt.plot(pred_y, color='gold', label='Predict') # 모델이 예상한 가격 그래프
plt.plot(test_y, color='green', label='test') # 실제 가격 그래프
plt.legend()
plt.show()

# 그래프를 보면 예측하지 못했음. R^2값과 RMSE 값 확인


# In[44]:


from sklearn.metrics import mean_squared_error, r2_score

print('r2_score : ', r2_score(test_y, pred_y))
RMSE = mean_squared_error(test_y, pred_y)**0.5
print('RMSE : ', RMSE)

# R^2 값이 음수가 나온다는 것은 해당 모델의 정확도가 매우 낮다는 것이다.
# ARIMA의 경우 긴 값을 예측할 때, 표본평균으로 회귀하려는 경향 때문에 R^2이 작게 나오는 거승로 판단된다.
# 예제에 사용한 데이터와 같이 계절성이 있는 경우, 계절성 지수가 추가된 SARIMA 모형을 사용하는 것이 좋다.


# ## 5. SARIMA
# (1) 개념
# 
#     SARIMA는 데이터가 지닌 계절성(주기에 따라 유사한 양상으로 변화하는 모양)까지 고려한 ARIMA 모델이다.
#     
# 모형 학습 과정
# 
#     1. 계절성이 몇 개의 데이터 단위로 나타나는 지를 확인
# 
#     2. seasonal_order에는 4개의 매개변수가 주어져야 하는데 s 값을 먼저 찾아주는 것이 좋다. 이는 시각화로 판단하는 것이 빠르며, 예제 데이터에서는 1년 단위로 계절성이 존재하는 것으로 보인다. 그러므로 s 값은 12로 설정해야 한다.
#     
#     3. 그 후 order * seasonal_order의 파라미터를 최적화 시켜주어야 하는데 seasonal_order의 (P, D, Q)는 머신러닝에서 배웠던 AIC값을 비교하며 최적의 값을 갖는 GridSearch로 찾을 수 있다. 이렇게 시계열 Grid_search를 지원해주는 패키지인 auto_arima가 있으나 ADP 시험장에서는 설치가 되어있지 않다. 하지만 21회부터 패키지 설치 가능

# In[45]:


# !pip install pmdarima


# In[47]:


from pmdarima import auto_arima

auto_model = auto_arima(training, start_p=0, d=1, start_q=0,
                       max_p=3, max_q=3,
                       start_P=0, start_Q=0,
                       max_P=3, max_Q=3, m=12,
                       seasonal=True, information_criterion='aic',
                       trace=True)

# auto_arima 결과, p=1, d=0, q=0, P=0, D=1, Q=0, m=12인 모델이 최적의 모델로 나왔다.
# arima에서는 2, 1, 2의 모델과는 완전히 다른 p, d, q 값을 가짐을 알 수 있다.
# 이는 계절성이 추가되면서 완전히 다른 학습이 되었다는 의미


# In[49]:


auto_model.summary()

# AIC가 480인 모델이 최적의 모델로 선택되었음을 알 수 있으며 p-value가 0.05보다 작은 MA(1) 변수와 m=12, D=1, Q=1이 적용된 모델임을 확인


# In[50]:


# 학습 데이터 세트로부터 test 데이터 길이만큼 예측
auto_pred_y = pd.DataFrame(auto_model.predict(n_periods=len(test)),
                          index=test.index)
auto_pred_y.columns = ['predicted_price']
auto_pred_y


# In[52]:


plt.figure(figsize=(10,6))
plt.plot(training, label='Train')
plt.plot(auto_pred_y, label='Prediction')
plt.plot(test, label='Test')
plt.legend(loc='upper left')
plt.show()

# 계절성이 존재하는 경우, ARIMA 모델보다 SARIMA 모델이 훨씬 정확도가 높음


# In[54]:


from sklearn.metrics import mean_squared_error, r2_score

print('r2_score : ', r2_score(test_y, auto_pred_y))
RMSE = mean_squared_error(test_y, auto_pred_y)**0.5
print('RMSE : ', RMSE)

# SARIMA 모델은 93% 정확도를 가지고 예측하였으며, 평균오차는 373원으로 매우 높은 정확도를 보였다.
# 이 데이터는 시간의 추세와 계절성만을 가지고 예측할 수 있다는 의미이다. 이 데이터는 시간의 영향을 많이 받는다.


# 시계열 분석에서 추세를 판단하는 데에는 ARIMA 모델이 정확도가 높을 수 있으나, 계절성이 존재하는 경우에는 SARIMA 모델을 사용하는 것이 좋다.
# 
# SARIMA 모델에서는 어떠한 매개변수를 데이터분석가가 적용시키느냐에 따라 정확도가 달라진다.

# In[ ]:





# In[ ]:





# In[ ]:




