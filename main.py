import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = 'LLz3seEtf9djqdmdMZAL'

df =quandl.get('WIKI/AMZN')
# print(df.head())
df = df[['Adj. Close']]
# print(df.head())

df['Adj. Close'].plot(figsize=(15, 6), color='g')
# plt.legend(loc='upper left')
plt.title('Amazon.com, Inc. Stock Price')
plt.ylabel('Adjusted closing price (USD)')
plt.xlim(pd.Timestamp(df.index[0]), pd.Timestamp(df.index[-1]))
plt.show()

forecast = 30
df['Prediction'] = df[['Adj. Close']].shift(-forecast)

x = np.array(df.drop(['Prediction'], 1))
x = preprocessing.scale(x)

x_forecast = x[-forecast:]
x =x[:-forecast]
y = np.array(df['Prediction'])
y = y[:-forecast]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# classifier
clf = LinearRegression()
clf.fit(x_train, y_train)

confidence = clf.score(x_test, y_test)

forecast_predicted = clf.predict(x_forecast)

dates = df.index[-30:]
plt.plot(dates, forecast_predicted, color='b')
df['Adj. Close'].plot(color='g')
plt.xlim(datetime.date(2017,4,26), df.index[-1])


