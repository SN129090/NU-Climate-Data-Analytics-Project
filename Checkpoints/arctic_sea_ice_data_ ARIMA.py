# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import time
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import r2_score, smape, mape, marre
from darts.models import StatsForecastAutoARIMA
#%%
plt.style.use("fivethirtyeight")
#%%
def display_forecast(pred_series, ts_transformed, forecast_type, chartname, start_date=None, ):
    plt.figure(figsize=(15, 8))
    if start_date:
        ts_transformed = ts_transformed.drop_before(start_date)
    ts_transformed.univariate_component(0).plot(label="actual")
    pred_series.plot(label=("historic " + forecast_type + " forecasts"))
    plt.title(chartname+
        "\nR2: {}".format(r2_score(ts_transformed.univariate_component(0), pred_series))
        +"\nMAPE: {}".format(mape(ts_transformed.univariate_component(0), pred_series))
        +"\nsMAPE: {}".format(smape(ts_transformed.univariate_component(0), pred_series))
        +"\nMARRE: {}".format(marre(ts_transformed.univariate_component(0), pred_series))    
    )
    plt.legend()
#%%
df = pd.read_csv("https://raw.githubusercontent.com/SN129090/NU-Climate-Data-Analytics-Project/main/Baffin%20Sea%20Ice%20Extent.csv")
df_beau = pd.read_csv("https://raw.githubusercontent.com/SN129090/NU-Climate-Data-Analytics-Project/main/Beaufort%20Sea%20Ice%20Extent.csv")
df_canarc = pd.read_csv("https://raw.githubusercontent.com/SN129090/NU-Climate-Data-Analytics-Project/main/CanArch%20Sea%20Ice%20Extent.csv")
df_hudson = pd.read_csv("https://raw.githubusercontent.com/SN129090/NU-Climate-Data-Analytics-Project/main/Hudson%20Sea%20Ice%20Extent.csv")
#%%
### Data preparation - converting the matrix of values into a single column for each dataset, 
### Creating date column and assigning as index

df['day'] = (df['day']).astype(str)
df['Month-Day'] = pd.concat([df['month'] + "-" + df['day']])
df_beau['day'] = (df_beau['day']).astype(str)
df_beau['Month-Day'] = pd.concat([df_beau['month'] + "-" + df_beau['day']])
df_canarc['day'] = (df_canarc['day']).astype(str)
df_canarc['Month-Day'] = pd.concat([df_canarc['month'] + "-" + df_canarc['day']])
df_hudson['day'] = (df_hudson['day']).astype(str)
df_hudson['Month-Day'] = pd.concat([df_hudson['month'] + "-" + df_hudson['day']])


baffin_melt = df.melt(id_vars='Month-Day', var_name='Year', value_name="BAFFIN SEA ICE EXTENT")
baffin_melt['Year'] = baffin_melt['Year'].astype(str)
baffin_melt = baffin_melt[baffin_melt.Year != 'month']
baffin_melt = baffin_melt[baffin_melt.Year != 'day']
baffin_melt['Date'] = pd.concat([baffin_melt['Month-Day']+"-"+baffin_melt['Year']])
baffin_melt = baffin_melt.dropna(axis=0)
baffin_melt['BAFFIN SEA ICE EXTENT'] = baffin_melt['BAFFIN SEA ICE EXTENT'].astype(np.float64)

beau_melt = df_beau.melt(id_vars='Month-Day', var_name='Year', value_name="BEAUFORT SEA ICE EXTENT")
beau_melt['Year'] = beau_melt['Year'].astype(str)
beau_melt = beau_melt[beau_melt.Year != 'month']
beau_melt = beau_melt[beau_melt.Year != 'day']
beau_melt['Date'] = pd.concat([beau_melt['Month-Day']+"-"+beau_melt['Year']])
beau_melt = beau_melt.dropna(axis=0)

canarc_melt = df_canarc.melt(id_vars='Month-Day', var_name='Year', value_name="CAN. ARCH. SEA ICE EXTENT")
canarc_melt['Year'] = canarc_melt['Year'].astype(str)
canarc_melt = canarc_melt[canarc_melt.Year != 'month']
canarc_melt = canarc_melt[canarc_melt.Year != 'day']
canarc_melt['Date'] = pd.concat([canarc_melt['Month-Day']+"-"+canarc_melt['Year']])
canarc_melt = canarc_melt.dropna(axis=0)

hudson_melt = df_hudson.melt(id_vars='Month-Day', var_name='Year', value_name="HUDSON BAY SEA ICE EXTENT")
hudson_melt['Year'] = hudson_melt['Year'].astype(str)
hudson_melt = hudson_melt[hudson_melt.Year != 'month']
hudson_melt = hudson_melt[hudson_melt.Year != 'day']
hudson_melt['Date'] = pd.concat([hudson_melt['Month-Day']+"-"+hudson_melt['Year']])
hudson_melt = hudson_melt.dropna(axis=0)

baffin_melt['Date'] = pd.to_datetime(baffin_melt['Date'], format='%B-%d-%Y')
baffin_melt['Date']
baffin_melt.set_index('Date', inplace =True)

beau_melt['Date'] = pd.to_datetime(beau_melt['Date'], format='%B-%d-%Y')
beau_melt['Date']
beau_melt.set_index('Date', inplace =True)

canarc_melt['Date'] = pd.to_datetime(canarc_melt['Date'], format='%B-%d-%Y')
canarc_melt['Date']
canarc_melt.set_index('Date', inplace =True)

hudson_melt['Date'] = pd.to_datetime(hudson_melt['Date'], format='%B-%d-%Y')
hudson_melt['Date']
hudson_melt.set_index('Date', inplace =True)
#%%
# Restoring month from date
baffin_melt['month'] = [d.strftime('%b') for d in baffin_melt.index]
beau_melt['month'] = [d.strftime('%b') for d in beau_melt.index]
canarc_melt['month'] = [d.strftime('%b') for d in canarc_melt.index]
hudson_melt['month'] = [d.strftime('%b') for d in hudson_melt.index]
#%%
from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(baffin_melt["BAFFIN SEA ICE EXTENT"])
print("Baffin pvalue = ", pvalue, " if above 0.05, data is not stationary")
#%%
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(beau_melt["BEAUFORT SEA ICE EXTENT"])
print("Beaufort Sea pvalue = ", pvalue, " if above 0.05, data is not stationary")
#%%
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(canarc_melt["CAN. ARCH. SEA ICE EXTENT"])
print("Canadian Archipelago pvalue = ", pvalue, " if above 0.05, data is not stationary")
#%%
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(hudson_melt["HUDSON BAY SEA ICE EXTENT"])
print("Hudson Bay pvalue = ", pvalue, " if above 0.05, data is not stationary")
#%%
# Reduce to monthly
baff_m = baffin_melt['BAFFIN SEA ICE EXTENT'].groupby(pd.Grouper(freq="M")).mean()
beau_m = beau_melt['BEAUFORT SEA ICE EXTENT'].groupby(pd.Grouper(freq="M")).mean()
canarc_m = canarc_melt['CAN. ARCH. SEA ICE EXTENT'].groupby(pd.Grouper(freq="M")).mean()
hudson_m = hudson_melt['HUDSON BAY SEA ICE EXTENT'].groupby(pd.Grouper(freq="M")).mean()
baff_q = baffin_melt['BAFFIN SEA ICE EXTENT'].groupby(pd.Grouper(freq="q")).mean()
beau_q = beau_melt['BEAUFORT SEA ICE EXTENT'].groupby(pd.Grouper(freq="q")).mean()
canarc_q = canarc_melt['CAN. ARCH. SEA ICE EXTENT'].groupby(pd.Grouper(freq="q")).mean()
hudson_q = hudson_melt['HUDSON BAY SEA ICE EXTENT'].groupby(pd.Grouper(freq="q")).mean()
#%%
# ARIMA MODEL SELECTION - BAFFIN Monthly
baff_m = baff_m.reset_index()
scaler = Scaler()
baf_m_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            baff_m, time_col="Date", value_cols="BAFFIN SEA ICE EXTENT")).astype(np.float32)

baf_arima_model = StatsForecastAutoARIMA(period=12)
size = int(len(baf_m_series) * 0.60)
X_train, X_test = baf_m_series[0:size], baf_m_series[size:len(baf_m_series)]

baf_ar_fit_start = time.time()
baf_arima_fit = baf_arima_model.fit(X_train)
baf_ar_fit_stop = time.time()
baf_ar_pre_start = time.time()
pred_baf_arima_series = baf_arima_fit.historical_forecasts(
    baf_m_series,
    start=pd.Timestamp("20131031"),
    forecast_horizon=7,
    stride=2,
    verbose=True,
)
baf_ar_pre_stop = time.time()
ba_ar_m_prediction = baf_arima_model.predict(50)
pred_baf_arima_series = scaler.inverse_transform(pred_baf_arima_series)
X_test = scaler.inverse_transform(X_test)
X_train = scaler.inverse_transform(X_train)
baf_m_series = scaler.inverse_transform(baf_m_series)
ba_ar_m_prediction = scaler.inverse_transform(ba_ar_m_prediction)
display_forecast(pred_baf_arima_series, X_test, "monthly","Auto-ARIMA Model - Baffin", start_date=pd.Timestamp("20131130"))
ba_ar_fit_time = baf_ar_fit_stop - baf_ar_fit_start
ba_ar_pred_time = baf_ar_pre_stop - baf_ar_pre_start
print("Train Time {}".format(ba_ar_fit_time)+"s"+ "\nPrediction Time: {}".format(ba_ar_pred_time)+"s")
#%%
# ARIMA MODEL SELECTION - Baffin Quarterly
baff_q = baff_q.reset_index()
scaler = Scaler()
ba_q_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            baff_q, time_col="Date", value_cols="BAFFIN SEA ICE EXTENT")).astype(np.float32)

q_arima_model = StatsForecastAutoARIMA(period=4)
size = int(len(ba_q_series) * 0.60)
X_train_baq, X_test_baq = ba_q_series[0:size], ba_q_series[size:len(ba_q_series)]

baq_ar_fit_start = time.time()
baq_arima_fit = q_arima_model.fit(X_train_baq)
baq_ar_fit_stop = time.time()
pred_baq_arima_series = baq_arima_fit.historical_forecasts(
    ba_q_series,
    start=pd.Timestamp("20130930"),
    forecast_horizon=7,
    stride=1,
    verbose=True,
)
baq_ar_pred_stop = time.time()

pred_baq_arima_series = scaler.inverse_transform(pred_baq_arima_series)
X_test_baq = scaler.inverse_transform(X_test_baq)
display_forecast(pred_baq_arima_series, X_test_baq, "quarterly","Auto-ARIMA Model - Baffin", start_date=pd.Timestamp("20131231"))
baq_ar_fit_time = baq_ar_fit_stop - baq_ar_fit_start
baq_ar_pred_time = baq_ar_pred_stop - baq_ar_fit_stop
print("Train Time {}".format(baq_ar_fit_time)+"s"+ "\nPrediction Time: {}".format(baq_ar_pred_time)+"s")
#%%
# ARIMA MODEL SELECTION - Beaufort Sea
beau_m = beau_m.reset_index()
scaler = Scaler()
be_m_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            beau_m, time_col="Date", value_cols="BEAUFORT SEA ICE EXTENT")).astype(np.float32)

arima_model = StatsForecastAutoARIMA(period=12)
size = int(len(be_m_series) * 0.60)
X_train_be, X_test_be = be_m_series[0:size], be_m_series[size:len(be_m_series)]

be_ar_fit_start = time.time()
be_arima_fit = arima_model.fit(X_train_be)
be_ar_fit_stop = time.time()
pred_be_arima_series = be_arima_fit.historical_forecasts(
    be_m_series,
    start=pd.Timestamp("20131031"),
    forecast_horizon=7,
    stride=2,
    verbose=True,
)
be_ar_pred_stop = time.time()

pred_be_arima_series = scaler.inverse_transform(pred_be_arima_series)
X_test_be = scaler.inverse_transform(X_test_be)
X_train_be = scaler.inverse_transform(X_train_be)
be_m_series = scaler.inverse_transform(be_m_series)
display_forecast(pred_be_arima_series, X_test_be, "monthly","Auto-ARIMA Model - Beaufort Sea", start_date=pd.Timestamp("20131130"))
be_ar_fit_time = be_ar_fit_stop - be_ar_fit_start
be_ar_pred_time = be_ar_pred_stop - be_ar_fit_stop
print("Train Time {}".format(be_ar_fit_time)+"s"+ "\nPrediction Time: {}".format(be_ar_pred_time)+"s")

#%%
# ARIMA MODEL SELECTION - Beaufort Sea Quarterly
beau_q = beau_q.reset_index()
scaler = Scaler()
be_q_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            beau_q, time_col="Date", value_cols="BEAUFORT SEA ICE EXTENT")).astype(np.float32)

q_arima_model = StatsForecastAutoARIMA(period=4)
size = int(len(be_q_series) * 0.60)
X_train_beq, X_test_beq = be_q_series[0:size], be_q_series[size:len(be_q_series)]

beq_ar_fit_start = time.time()
beq_arima_fit = q_arima_model.fit(X_train_beq)
beq_ar_fit_stop = time.time()
pred_beq_arima_series = beq_arima_fit.historical_forecasts(
    be_q_series,
    start=pd.Timestamp("20130930"),
    forecast_horizon=7,
    stride=1,
    verbose=True,
)
beq_ar_pred_stop = time.time()

pred_beq_arima_series = scaler.inverse_transform(pred_beq_arima_series)
X_test_beq = scaler.inverse_transform(X_test_beq)
display_forecast(pred_beq_arima_series, X_test_beq, "quarterly","Auto-ARIMA Model - Beaufort Sea", start_date=pd.Timestamp("20131231"))
beq_ar_fit_time = beq_ar_fit_stop - beq_ar_fit_start
beq_ar_pred_time = beq_ar_pred_stop - beq_ar_fit_stop
print("Train Time {}".format(beq_ar_fit_time)+"s"+ "\nPrediction Time: {}".format(beq_ar_pred_time)+"s")
#%%
# ARIMA MODEL SELECTION - Canadian Archipelago
canarc_m = canarc_m.reset_index()
scaler = Scaler()
ca_m_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            canarc_m, time_col="Date", value_cols="CAN. ARCH. SEA ICE EXTENT")).astype(np.float32)

arima_model = StatsForecastAutoARIMA(period=12)
size = int(len(ca_m_series) * 0.60)
X_train_ca, X_test_ca = ca_m_series[0:size], ca_m_series[size:len(ca_m_series)]

ca_ar_fit_start = time.time()
ca_arima_fit = arima_model.fit(X_train_ca)
ca_ar_fit_stop = time.time()
pred_ca_arima_series = ca_arima_fit.historical_forecasts(
    ca_m_series,
    start=pd.Timestamp("20131031"),
    forecast_horizon=7,
    stride=2,
    verbose=True,
)
ca_ar_pred_stop = time.time()
pred_ca_arima_series = scaler.inverse_transform(pred_ca_arima_series)
X_test_ca = scaler.inverse_transform(X_test_ca)
display_forecast(pred_ca_arima_series, X_test_ca, "monthly","Auto-ARIMA Model - Canadian Archipelago", start_date=pd.Timestamp("20131130"))
ca_ar_fit_time = ca_ar_fit_stop - ca_ar_fit_start
ca_ar_pred_time = ca_ar_pred_stop - ca_ar_fit_stop
print("Train Time {}".format(ca_ar_fit_time)+"s"+ "\nPrediction Time: {}".format(ca_ar_pred_time)+"s")
#%%
# ARIMA MODEL SELECTION - Canadian Archipelago Quarterly
canarc_q = canarc_q.reset_index()
scaler = Scaler()
ca_q_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            canarc_q, time_col="Date", value_cols="CAN. ARCH. SEA ICE EXTENT")).astype(np.float32)

q_arima_model = StatsForecastAutoARIMA(period=4)
size = int(len(ca_q_series) * 0.60)
X_train_caq, X_test_caq = ca_q_series[0:size], ca_q_series[size:len(ca_q_series)]

caq_ar_fit_start = time.time()
caq_arima_fit = q_arima_model.fit(X_train_caq)
caq_ar_fit_stop = time.time()
pred_caq_arima_series = caq_arima_fit.historical_forecasts(
    ca_q_series,
    start=pd.Timestamp("20130930"),
    forecast_horizon=7,
    stride=1,
    verbose=True,
)
caq_ar_pred_stop = time.time()
pred_caq_arima_series = scaler.inverse_transform(pred_caq_arima_series)
X_test_caq = scaler.inverse_transform(X_test_caq)
display_forecast(pred_caq_arima_series, X_test_caq, "quarterly","Auto-ARIMA Model - Canadian Archipelago", start_date=pd.Timestamp("20131130"))
caq_ar_fit_time = caq_ar_fit_stop - caq_ar_fit_start
caq_ar_pred_time = caq_ar_pred_stop - caq_ar_fit_stop
print("Train Time {}".format(caq_ar_fit_time)+"s"+ "\nPrediction Time: {}".format(caq_ar_pred_time)+"s")
#%%
# ARIMA MODEL SELECTION - Hudson Bay
hudson_m = hudson_m.reset_index()
scaler = Scaler()
hb_m_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            hudson_m, time_col="Date", value_cols="HUDSON BAY SEA ICE EXTENT")).astype(np.float32)

hb_arima_model = StatsForecastAutoARIMA(period=12)
size = int(len(hb_m_series) * 0.60)
X_train_hb, X_test_hb = hb_m_series[0:size], hb_m_series[size:len(hb_m_series)]
hb_ar_fit_start = time.time()
hb_arima_fit = hb_arima_model.fit(X_train_hb)
hb_ar_fit_stop = time.time()
pred_hb_arima_series = hb_arima_fit.historical_forecasts(
    hb_m_series,
    start=pd.Timestamp("20131031"),
    forecast_horizon=7,
    stride=2,
    verbose=True,
)
hb_ar_pred_stop = time.time()
pred_hb_arima_series = scaler.inverse_transform(pred_hb_arima_series)
X_test_hb = scaler.inverse_transform(X_test_hb)
display_forecast(pred_hb_arima_series, X_test_hb, "monthly","Auto-ARIMA Model - Hudson Bay", start_date=pd.Timestamp("20131130"))
hb_ar_fit_time = hb_ar_fit_stop - hb_ar_fit_start
hb_ar_pred_time = hb_ar_pred_stop - hb_ar_fit_stop
print("Train Time {}".format(hb_ar_fit_time)+"s"+ "\nPrediction Time: {}".format(hb_ar_pred_time)+"s")
#%%
# ARIMA MODEL SELECTION - Hudson Bay Quarterly
hudson_q = hudson_q.reset_index()
scaler = Scaler()
hb_q_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            hudson_q, time_col="Date", value_cols="HUDSON BAY SEA ICE EXTENT")).astype(np.float32)

q_arima_model = StatsForecastAutoARIMA(period=4)
size = int(len(hb_q_series) * 0.60)
X_train_hbq, X_test_hbq = hb_q_series[0:size], hb_q_series[size:len(hb_q_series)]
hbq_ar_fit_start = time.time()
hbq_arima_fit = q_arima_model.fit(X_train_hbq)
hbq_ar_fit_stop = time.time()
pred_hbq_arima_series = hbq_arima_fit.historical_forecasts(
    hb_q_series,
    start=pd.Timestamp("20130930"),
    forecast_horizon=7,
    stride=1,
    verbose=True,
)
hbq_ar_pred_stop = time.time()
pred_hbq_arima_series = scaler.inverse_transform(pred_hbq_arima_series)
X_test_hbq = scaler.inverse_transform(X_test_hbq)
display_forecast(pred_hbq_arima_series, X_test_hbq, "quarterly","Auto-ARIMA Model - Hudson Bay", start_date=pd.Timestamp("20131231"))
hbq_ar_fit_time = hbq_ar_fit_stop - hbq_ar_fit_start
hbq_ar_pred_time = hbq_ar_pred_stop - hbq_ar_fit_stop
print("Train Time {}".format(hbq_ar_fit_time)+"s"+ "\nPrediction Time: {}".format(hbq_ar_pred_time)+"s")
