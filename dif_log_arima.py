# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 22:39:45 2023

@author: ibone
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as mt
#%%
#Import Data
data = pd.read_excel("usa_arima.xlsx")
# Converting year column to datetime format
data['date'] = pd.to_datetime(data['date'], format = '%Y-%m')
print(f'Time period start : {data.date.min()}\nTime period end : {data.date.max()}')
# Set as index the Date column and convert it into Datetime Object.
data.set_index('date',inplace=True)
#DO NOT USE IT
#tmp = data['brent']
#log_diff = 100*(np.log(tmp[1:])-np.log(tmp[:-1]))
#print("Length of tmp[1:]:", len(tmp[1:]))
#print("Length of tmp[:-1]:", len(tmp[:-1]))
#log_diff = 100 * (np.log(data['brent'].iloc[:-1]) - np.log(data['brent'].iloc[1:]))   #DATAFRAME NOT WORKING
# Check if the number of final values is equal to the number of initial values minus one
#if len(log_diff) == len(data['brent']) - 1:
#    print("The number of final values is equal to the number of initial values minus one.")
#else:
#    print("The number of final values is not equal to the number of initial values minus one.")

# Convert DataFrame to array   IT WORKS
array = data.values
print(array)
log_diff = 100 * (np.log(array[1:, 0]) - np.log(array[:-1, 0]))     #ARRAY
log_diff = log_diff[~np.isnan(log_diff)]
data['log_diff_brent']=np.nan
data['log_diff_brent'].iloc[1:] = log_diff

# Plot the array
plt.plot(data['log_diff_brent'])
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Log diff Brent Plot')
plt.show()

#%% check stationarity
data_cut = data.iloc[81:172,]
from statsmodels.tsa.stattools import adfuller           
test_brent_stationarity=adfuller(data_cut['log_diff_brent'])          

print('ADF Statistic: %f' % test_brent_stationarity[0])
print('p-value: %f' % test_brent_stationarity[1])
print('Critical Values:')
for key, value in test_brent_stationarity[4].items():
    print('\t%s: %.3f' % (key, value))

if test_brent_stationarity[0] < test_brent_stationarity[4]["5%"]:
    print ("Reject Ho - Time Series is Stationary")
else:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")
    
def test_stationarity(timeseries, windowroll = 12, cutoff = 0.05):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window = 12).mean()
    rolstd = timeseries.rolling(window = 12).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
# Stationarity test of log differend brent
test_stationarity(data_cut['log_diff_brent'])
#%%PACF & ACF plot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(data_cut['log_diff_brent']);
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# ACF & PACF Plots for log Differenced Brent
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14,6), sharex=False, sharey=False)
ax1 = plot_acf(data_cut['log_diff_brent'], lags=20, ax=ax1)
ax2 = plot_pacf(data_cut['log_diff_brent'], lags=20, ax=ax2)
plt.show()
#%% BOX JENKINS ARIMA
# Importing time series specific libraries
#!pip install pmdarima
#!pip install prophet

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import prophet
from prophet import Prophet

# Libaraies for evaluation of model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error
from statsmodels.tsa.arima.model import ARIMAResults
#%%
# Splitting TS data into train and test set for model training and testing
train_ts = data_cut['log_diff_brent'].iloc[0:80,]
test_ts = data_cut['log_diff_brent'].iloc[80: ,]
model = pm.auto_arima(train_ts, d=0, D=0,
                      seasonal=False, m=4, trend='c',start_p=0, 
                      start_q=0, max_order=7, test='adf', stepwise=True, trace=True)
results = model.fit(train_ts)
results.summary()
#In sample predictions for test
predictions = results.predict(n_periods = len(test_ts)+20)
#In sample predictions for train
train_predict = results.predict_in_sample(start = 1, end = 80)

plt.figure(figsize = (15,6))
plt.plot(data_cut['log_diff_brent'], color = 'green', label = 'Log Transformed Original data')
plt.plot(train_predict, color = 'blue', label = 'Predicted values for train dataset')
plt.plot(predictions, color = 'orange', label = 'Predicted values for test dataset')
plt.xlabel('date')
plt.ylabel('Brent')
plt.title('Auto-ARIMA model qualitative performance')
plt.legend(loc = 'best')
plt.show()

model.plot_diagnostics(figsize=(15, 12))
plt.show()
 #%%BUILD MODEL       
model = pm.auto_arima(data_cut['log_diff_brent'], d=0, D=0,
                      seasonal=False, m=4, trend='c',start_p=0, 
                      start_q=0, max_order=7, test='adf', stepwise=True, trace=True)
results = model.fit(data_cut['log_diff_brent'])
print(results.summary())

model.plot_diagnostics(figsize=(15, 12))
plt.show()
# fit model
# order (p, d, q)
model = ARIMA(train_ts, order=(0,0,1))
results2 = model.fit()
results2.summary()
predictions2 = results2.forecast(steps = len(test_ts)+20)
plt.figure(figsize = (15,6))
plt.plot(data_cut['log_diff_brent'], color = 'green', label = 'Log Transformed Original data')
plt.plot(results2.fittedvalues, color = 'blue', label = 'Predicted values for train dataset')
plt.plot(predictions2, color = 'orange', label = 'Predicted values for test dataset')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('ARIMA(3,2,2)')
plt.legend(loc = 'best')
plt.show()
#%%
forecast = results.get_forecast(steps=30)
ci = forecast.conf_int()
ax = data_cut['log_diff_brent'].plot(label='Brent', figsize=(8, 6))
forecast.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(ci.index,
                ci.iloc[:, 0],
                ci.iloc[:, 1], color='y', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Price')
plt.legend()
plt.show()
# Make predictions
predictions = model.predict(start=0, end=len(data_cut)-1)
r2_score(data_cut['log_diff_brent'],predictions)
#%%
# Evaluation metrics for model
MSE_train = mean_squared_error(train_ts, train_predict)
print('Mean Squared Error (MSE) of model Auto-ARIMA for trained data = ',MSE_train)
MAE_train = mean_absolute_error(train_ts, train_predict)
print('Mean Absolute Error (MAE) of model Auto-ARIMA for trained data = ',MAE_train)
RMSE_train = np.sqrt(MSE_train)
print('Root Mean Squared Error (RMSE) of model Auto-ARIMA for trained data = ',RMSE_train)
MSE_test = mean_squared_error(test_ts, predictions)
print('Mean Squared Error (MSE) of model Auto-ARIMA] for test data = ',MSE_test)
MAE_test= mean_absolute_error(test_ts, predictions)
print('Mean Absolute Error (MAE) of model Auto-ARIMA for test data = ',MAE_test)
RMSE_test = np.sqrt(MSE_test)
print('Root Mean Squared Error (RMSE) of model Auto-ARIMA for test data = ',RMSE_test)

#%%FORECAST FOR EACH QUANTILE
#Standard deviation of Brent
stds = np.std(data_cut['brent'])
# Define quantiles
quantiles = [90, 10, 50, 80, 20, 70, 30, 60, 40]
quantile_neg = [10, 20,30,40]
# Loop through quantiles and calculate forecasts
forecasts = []
for quantile in quantiles:
    model.fit(data_cut['log_diff_brent'])
    forecast = model.get_forecast(steps=30, alpha=quantile)
    forecasted_values = forecast.predicted_mean
    forecasts.append(forecasted_values)


    
# Calculate and set the quantile values
for quantile in quantiles:
    quantile_name = 'Q' + str(quantile)
    multiplier = 1.66 if quantile in [90, 10] else 1.29 if quantile in [80, 20] else 1.03 if quantile in [70, 30] else 0.85 if quantile in [60, 40] else 0.0
    if quantile in quantile_neg:
       data[quantile_name] = data['log_diff_brent'] - multiplier * stds

    else:
       data[quantile_name] = data['log_diff_brent'] + multiplier * stds
