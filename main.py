import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

# acquire the financial data
quandl.ApiConfig.api_key = 'LLz3seEtf9djqdmdMZAL'

# get stock prices of Amazon
df =quandl.get('WIKI/AMZN')
# consider only the adjusted closing price
df = df[['Adj. Close']]

"""
# plot the history stock price of the company
df['Adj. Close'].plot(figsize=(15, 6), color='g')
plt.title('Amazon.com, Inc. Stock Price')
plt.ylabel('Adjusted closing price (USD)')
plt.xlim(pd.Timestamp(df.index[0]), pd.Timestamp(df.index[-1]))
plt.show()
"""

# the model starts here
forecast = 30
# shift the data by the length of forecast
df['Prediction'] = df[['Adj. Close']].shift(-forecast)
# 'Adj. Close' as feature, 'Prediction as label
x = np.array(df.drop(['Prediction'], 1))
# normalization, mean = 0, std = 1
x = preprocessing.scale(x)

x_forecast = x[-forecast:]
x =x[:-forecast]
y = np.array(df['Prediction'])
y = y[:-forecast]
# split the data into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# create a classifier
clf = LinearRegression()
# fit the data from the training set to the model
clf.fit(x_train, y_train)
# how good is the fitting? score on test set
confidence = clf.score(x_test, y_test)
# perform the prediction with the classifier from training
forecast_predicted = clf.predict(x_forecast)

"""
# predict the next 30 days from the last day of the data
dates_extend = pd.date_range(start='2018-03-28', end='2018-04-26')
plt.figure(figsize=(15, 6))
plt.plot(dates_extend, forecast_predicted, color='b')
df['Adj. Close'].plot(color='g')
plt.xlim(datetime.date(2017, 4, 26), dates_extend[-1])
plt.title('Amazon.com, Inc. Stock Price')
plt.ylabel('Adjusted closing price (USD)')
plt.legend(['prediction', 'history'], loc='upper left')
plt.show()
"""

# predict 30 days forward for the last 30 days in the data
# these last 30 days actually have a span of 43 days
dates = df.index[-30:]
dates_forecast = []
for date in dates:
    date = date.date() + datetime.timedelta(days=forecast)
    dates_forecast.append(date)
plt.figure(figsize=(15, 6))
plt.plot(dates_forecast, forecast_predicted, color='b')
df['Adj. Close'].plot(color='g')
plt.xlim(datetime.date(2017, 4, 26), dates_forecast[-1])
plt.title('Amazon.com, Inc. Stock Price')
plt.ylabel('Adjusted closing price (USD)')
plt.legend(['prediction', 'history'], loc='upper left')
plt.show()
