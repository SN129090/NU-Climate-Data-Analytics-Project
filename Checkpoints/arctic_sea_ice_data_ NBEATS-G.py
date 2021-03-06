# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import r2_score, mape, smape, marre
import numpy as np
import time
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
    plt.legend(loc='best')
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
print("pvalue = ", pvalue, " if above 0.05, data is not stationary")
#%%
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(beau_melt["BEAUFORT SEA ICE EXTENT"])
print("pvalue = ", pvalue, " if above 0.05, data is not stationary")
#%%
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(canarc_melt["CAN. ARCH. SEA ICE EXTENT"])
print("pvalue = ", pvalue, " if above 0.05, data is not stationary")
#%%
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(hudson_melt["HUDSON BAY SEA ICE EXTENT"])
print("pvalue = ", pvalue, " if above 0.05, data is not stationary")
#%%
# Reduce to monthly
baff_m = baffin_melt['BAFFIN SEA ICE EXTENT'].groupby(pd.Grouper(freq="M")).mean()
beau_m = beau_melt['BEAUFORT SEA ICE EXTENT'].groupby(pd.Grouper(freq="M")).mean()
canarc_m = canarc_melt['CAN. ARCH. SEA ICE EXTENT'].groupby(pd.Grouper(freq="M")).mean()
hudson_m = hudson_melt['HUDSON BAY SEA ICE EXTENT'].groupby(pd.Grouper(freq="M")).mean()
baff_y = baffin_melt['BAFFIN SEA ICE EXTENT'].groupby(pd.Grouper(freq="y")).mean()
beau_y = beau_melt['BEAUFORT SEA ICE EXTENT'].groupby(pd.Grouper(freq="y")).mean()
canarc_y = canarc_melt['CAN. ARCH. SEA ICE EXTENT'].groupby(pd.Grouper(freq="y")).mean()
hudson_y = hudson_melt['HUDSON BAY SEA ICE EXTENT'].groupby(pd.Grouper(freq="y")).mean()
#baff_m.drop(index=baff_m.index[0], axis=0, inplace=True)
#baff_m.drop(index=baff_m.index[0], axis=0, inplace=True)
#%%
#Preparing data for n-beats models and normalizing using min-max scaler
baff_d = baffin_melt.reset_index()
baff_m = baff_m.reset_index()
baff_y = baff_y.reset_index()
scaler = Scaler()
baf_d_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            baffin_melt, time_col="Date", value_cols="BAFFIN SEA ICE EXTENT")).astype(np.float32)

baf_m_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            baff_m, time_col="Date", value_cols="BAFFIN SEA ICE EXTENT")).astype(np.float32)
baf_y_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            baff_y, time_col="Date", value_cols="BAFFIN SEA ICE EXTENT")).astype(np.float32)
baf_d_series.plot()
baf_y_series.plot()
plt.title("Baffin Sea Ice")
#%% - Beaufort Sea
beau_d = beau_melt.reset_index()
beau_m = beau_m.reset_index()
beau_y = beau_y.reset_index()
beau_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            beau_d, time_col="Date", value_cols="BEAUFORT SEA ICE EXTENT")).astype(np.float32)
beau_m_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            beau_m, time_col="Date", value_cols="BEAUFORT SEA ICE EXTENT")).astype(np.float32)
beau_y_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            beau_y, time_col="Date", value_cols="BEAUFORT SEA ICE EXTENT")).astype(np.float32)
beau_series.plot()
beau_m_series.plot()
plt.title("Beaufort Sea Ice")
#%% - Canadian Archipelago
canarc_d = canarc_melt.reset_index()
canarc_m = canarc_m.reset_index()
canarc_y = canarc_y.reset_index()
canarc_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            canarc_d, time_col="Date", value_cols="CAN. ARCH. SEA ICE EXTENT")).astype(np.float32)
canarc_m_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            canarc_m, time_col="Date", value_cols="CAN. ARCH. SEA ICE EXTENT")).astype(np.float32)
canarc_y_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            canarc_y, time_col="Date", value_cols="CAN. ARCH. SEA ICE EXTENT")).astype(np.float32)
canarc_series.plot()
canarc_m_series.plot()
plt.title("Canadian Archipelago Sea Ice")
#%% - Hudson Bay
hud_d = hudson_melt.reset_index()
hudson_m = hudson_m.reset_index()
hudson_y = hudson_y.reset_index()
hud_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
           hud_d, time_col="Date", value_cols="HUDSON BAY SEA ICE EXTENT")).astype(np.float32)
hud_m_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
           hudson_m, time_col="Date", value_cols="HUDSON BAY SEA ICE EXTENT")).astype(np.float32)
hud_y_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
           hudson_y, time_col="Date", value_cols="HUDSON BAY SEA ICE EXTENT")).astype(np.float32)
hud_series.plot()
hud_m_series.plot()
plt.title("Hudson Bay Sea Ice")
#%% N-BEATS Generic Architecture Model
model_nbeats = NBEATSModel(
    input_chunk_length=30,
    output_chunk_length=7,
    generic_architecture=True,
    num_stacks=10,
    num_blocks=1,
    num_layers=4,
    layer_widths=512,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=1024,
    model_name="nbeats_run",
)
#%% N-BEATS Generic Architecture Model - Quarterly
q_model_nbeats = NBEATSModel(
    input_chunk_length=20,
    output_chunk_length=5,
    generic_architecture=True,
    num_stacks=10,
    num_blocks=1,
    num_layers=4,
    layer_widths=512,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=1024,
    model_name="nbeats_run",
)
#%% - Daily
baf_d_train, baf_d_val = baf_d_series.split_after(pd.Timestamp("20041231"))
beau_train, beau_val = beau_series.split_after(pd.Timestamp("20041231"))
canarc_train, canarc_val = canarc_series.split_after(pd.Timestamp("20041231"))
hud_train, hud_val = hud_series.split_after(pd.Timestamp("20041231"))
baf_d_val, baf_test = baf_d_val.split_after(pd.Timestamp("20131031"))
beau_val, beau_test = beau_val.split_after(pd.Timestamp("20131031"))
canarc_val, canarc_test = canarc_val.split_after(pd.Timestamp("20131031"))
hud_val, hud_test = hud_val.split_after(pd.Timestamp("20131031"))
#%% - Monthly
baf_m_train, baf_m_val = baf_m_series.split_after(pd.Timestamp("20041231"))
beau_m_train, beau_m_val = beau_m_series.split_after(pd.Timestamp("20041231"))
canarc_m_train, canarc_m_val = canarc_m_series.split_after(pd.Timestamp("20041231"))
hud_m_train, hud_m_val = hud_m_series.split_after(pd.Timestamp("20041231"))
baf_m_val, baf_m_test = baf_m_val.split_after(pd.Timestamp("20131031"))
beau_m_val, beau_m_test = beau_m_val.split_after(pd.Timestamp("20131031"))
canarc_m_val, canarc_m_test = canarc_m_val.split_after(pd.Timestamp("20131031"))
hud_m_val, hud_m_test = hud_m_val.split_after(pd.Timestamp("20131031"))
#%%
#NBEATS MODEL - BAFFIN
baf_d = baffin_melt.reset_index()
scaler = Scaler()
baf_d_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            baf_d, time_col="Date", value_cols="BAFFIN SEA ICE EXTENT")).astype(np.float32)
baf_d_train, baf_d_val = baf_d_series.split_after(pd.Timestamp("20041231"))
baf_d_val, baf_test = baf_d_val.split_after(pd.Timestamp("20131031"))

ba_mod_start = time.time()
baf_d_model = model_nbeats.fit(baf_d_train, val_series=baf_d_val, verbose=True)
ba_mod_stop = time.time()
ba_pred_start = time.time()
pred_d_series = baf_d_model.historical_forecasts(
    baf_d_series,
    start=pd.Timestamp("20131031"),
    forecast_horizon=7,
    stride=5,
    retrain=False,
    verbose=True,
)
ba_pred_stop = time.time()
pred_d_series = scaler.inverse_transform(pred_d_series)
baf_test = scaler.inverse_transform(baf_test)
display_forecast(pred_d_series, baf_test, "daily","N-BEATS Generic Model - Baffin", start_date=pd.Timestamp("20131101"))
ba_mod_time = ba_mod_stop - ba_mod_start
ba_pred_time = ba_pred_stop - ba_pred_start
print("Train Time {}".format(ba_mod_time)+"s"+ "\nPrediction Time: {}".format(ba_pred_time)+"s")
#%%
#NBEATS MODEL - BAFFIN Generic Monthly
baff_m = baff_m.reset_index()
scaler = Scaler()
baf_m_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            baff_m, time_col="Date", value_cols="BAFFIN SEA ICE EXTENT")).astype(np.float32)
baf_m_train, baf_m_val = baf_m_series.split_after(pd.Timestamp("20041231"))
baf_m_val, baf_m_test = baf_m_val.split_after(pd.Timestamp("20131031"))
ba_m_mod_start = time.time()
baf_m_model = model_nbeats.fit(baf_m_train, val_series=baf_m_val, verbose=True)
ba_m_mod_stop = time.time()
ba_m_pred_start = time.time()
pred_ba_m_series = baf_m_model.historical_forecasts(
    baf_m_series,
    start=pd.Timestamp("20131031"),
    forecast_horizon=7,
    stride=2,
    retrain=False,
    verbose=True,
)
ba_m_pred_stop = time.time()
pred_ba_m_series = scaler.inverse_transform(pred_ba_m_series)
baf_m_test = scaler.inverse_transform(baf_m_test)
display_forecast(pred_ba_m_series, baf_m_test, "monthly","N-BEATS Generic Model - Baffin", start_date=pd.Timestamp("20131130"))
ba_m_mod_time = ba_m_mod_stop - ba_m_mod_start
ba_m_pred_time = ba_m_pred_stop - ba_m_pred_start
print("Train Time {}".format(ba_m_mod_time)+"s"+ "\nPrediction Time: {}".format(ba_m_pred_time)+"s")
#%%
#NBEATS MODEL - Beaufort Sea - Generic Model
beau_d = beau_melt.reset_index()
beau_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            beau_d, time_col="Date", value_cols="BEAUFORT SEA ICE EXTENT")).astype(np.float32)
beau_train, beau_val = beau_series.split_after(pd.Timestamp("20041231"))
beau_val, beau_test = beau_val.split_after(pd.Timestamp("20131031"))
be_mod_start = time.time()
beau_model = model_nbeats.fit(beau_train, val_series=beau_val, verbose=True)
be_mod_stop = time.time()
be_pred_start = time.time()
pred_beau_series = beau_model.historical_forecasts(
    beau_series,
    start=pd.Timestamp("20131031"),
    forecast_horizon=7,
    stride=5,
    retrain=False,
    verbose=True,
)
be_pred_stop = time.time()
pred_beau_series = scaler.inverse_transform(pred_beau_series)
beau_test = scaler.inverse_transform(beau_test)
display_forecast(pred_beau_series, beau_test, "5-day","N-BEATS Generic Model - Beaufort Sea", start_date=pd.Timestamp("20131101"))
be_mod_time = be_mod_stop - be_mod_start
be_pred_time = be_pred_stop - be_pred_start
print("Train Time {}".format(be_mod_time)+"s"+ "\nPrediction Time: {}".format(be_pred_time)+"s")
#%%
#NBEATS MODEL - Beaufort Sea Generic Monthly
beau_m = beau_m.reset_index()
scaler = Scaler()
beau_m_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            beau_m, time_col="Date", value_cols="BEAUFORT SEA ICE EXTENT")).astype(np.float32)
beau_m_train, beau_m_val = beau_m_series.split_after(pd.Timestamp("20041231"))
beau_m_val, beau_m_test = beau_m_val.split_after(pd.Timestamp("20131031"))
beau_m_mod_start = time.time()
beau_m_model = model_nbeats.fit(beau_m_train, val_series=beau_m_val, verbose=True)
beau_m_mod_stop = time.time()
beau_m_pred_start = time.time()
pred_beau_m_series = beau_m_model.historical_forecasts(
    beau_m_series,
    start=pd.Timestamp("20131031"),
    forecast_horizon=7,
    stride=2,
    retrain=False,
    verbose=True,
)
beau_m_pred_stop = time.time()
pred_beau_m_series = scaler.inverse_transform(pred_beau_m_series)
beau_m_test = scaler.inverse_transform(beau_m_test)
display_forecast(pred_beau_m_series, beau_m_test, "monthly","N-BEATS Generic Model - Beaufort Sea", start_date=pd.Timestamp("20131130"))
beau_m_mod_time = beau_m_mod_stop - beau_m_mod_start
beau_m_pred_time = beau_m_pred_stop - beau_m_pred_start
print("Train Time {}".format(beau_m_mod_time)+"s"+ "\nPrediction Time: {}".format(beau_m_pred_time)+"s")
#%%
#NBEATS MODEL - Canadian Archipelago Sea
canarc_d = canarc_melt.reset_index()
canarc_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            canarc_d, time_col="Date", value_cols="CAN. ARCH. SEA ICE EXTENT")).astype(np.float32)
canarc_train, canarc_val = canarc_series.split_after(pd.Timestamp("20041231"))
canarc_val, canarc_test = canarc_val.split_after(pd.Timestamp("20131031"))

ca_mod_start = time.time()
canarc_model = model_nbeats.fit(canarc_train, val_series=canarc_val, verbose=True)
ca_mod_stop = time.time()
ca_pred_start = time.time()
pred_canarc_series = canarc_model.historical_forecasts(
    canarc_series,
    start=pd.Timestamp("20131031"),
    forecast_horizon=7,
    stride=5,
    retrain=False,
    verbose=True,
)
ca_pred_stop = time.time()
pred_canarc_series = scaler.inverse_transform(pred_canarc_series)
canarc_test = scaler.inverse_transform(canarc_test)
display_forecast(pred_canarc_series, canarc_test, "daily","N-BEATS Generic Model - Canadian Archipelago", start_date=pd.Timestamp("20131101"))
ca_mod_time = ca_mod_stop - ca_mod_start
ca_pred_time = ca_pred_stop - ca_pred_start
print("Train Time {}".format(ca_mod_time)+"s"+ "\nPrediction Time: {}".format(ca_pred_time)+"s")
#%%
#NBEATS MODEL - Canadian Archipelago Sea Monthly
canarc_m = canarc_m.reset_index()
scaler = Scaler()
canarc_m_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            canarc_m, time_col="Date", value_cols="CAN. ARCH. SEA ICE EXTENT")).astype(np.float32)
canarc_m_train, canarc_m_val = canarc_m_series.split_after(pd.Timestamp("20041231"))
canarc_m_val, canarc_m_test = canarc_m_val.split_after(pd.Timestamp("20131031"))
ca_m_mod_start = time.time()
canarc_m_model = model_nbeats.fit(canarc_m_train, val_series=canarc_m_val, verbose=True)
ca_m_mod_stop = time.time()
ca_m_pred_start = time.time()
pred_m_canarc_series = canarc_m_model.historical_forecasts(
    canarc_m_series,
    start=pd.Timestamp("20131031"),
    forecast_horizon=7,
    stride=2,
    retrain=False,
    verbose=True,
)
ca_m_pred_stop = time.time()
pred_m_canarc_series = scaler.inverse_transform(pred_m_canarc_series)
canarc_m_test = scaler.inverse_transform(canarc_m_test)
display_forecast(pred_m_canarc_series, canarc_m_test, "monthly","N-BEATS Generic Model - Canadian Archipelago", start_date=pd.Timestamp("20131130"))
ca_m_mod_time = ca_m_mod_stop - ca_m_mod_start
ca_m_pred_time = ca_m_pred_stop - ca_m_pred_start
print("Train Time {}".format(ca_m_mod_time)+"s"+ "\nPrediction Time: {}".format(ca_m_pred_time)+"s")
#%%
#NBEATS MODEL - Hudson Bay
hud_d = hudson_melt.reset_index()
hud_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
           hud_d, time_col="Date", value_cols="HUDSON BAY SEA ICE EXTENT")).astype(np.float32)
hud_train, hud_val = hud_series.split_after(pd.Timestamp("20041231"))
hud_val, hud_test = hud_val.split_after(pd.Timestamp("20131031"))

hb_mod_start = time.time()
hud_model = model_nbeats.fit(hud_train, val_series=hud_val, verbose=True)
hb_mod_stop = time.time()
hb_pred_start = time.time()
pred_hud_series = hud_model.historical_forecasts(
    hud_series,
    start=pd.Timestamp("20131031"),
    forecast_horizon=7,
    stride=5,
    retrain=False,
    verbose=True,
)
hb_pred_stop = time.time()
pred_hud_series = scaler.inverse_transform(pred_hud_series)
hud_test = scaler.inverse_transform(hud_test)
display_forecast(pred_hud_series, hud_test, "daily","N-BEATS Generic Model - Hudson Bay", start_date=pd.Timestamp("20131101"))
hb_mod_time = hb_mod_stop - hb_mod_start
hb_pred_time = hb_pred_stop - hb_pred_start
print("Train Time {}".format(hb_mod_time)+"s"+ "\nPrediction Time: {}".format(hb_pred_time)+"s")
#%%
#NBEATS MODEL - Hudson Bay Monthly
#hudson_m = hudson_m.reset_index()
scaler = Scaler()
hudson_m_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            hudson_m, time_col="Date", value_cols="HUDSON BAY SEA ICE EXTENT")).astype(np.float32)
hudson_m_train, hudson_m_val = hudson_m_series.split_after(pd.Timestamp("20041231"))
hudson_m_val, hudson_m_test = hudson_m_val.split_after(pd.Timestamp("20131031"))
hbm_mod_start = time.time()
hudm_model = model_nbeats.fit(hudson_m_train, val_series=hudson_m_val, verbose=True)
hbm_mod_stop = time.time()
hbm_pred_start = time.time()
pred_hudm_series = hudm_model.historical_forecasts(
    hudson_m_series,
    start=pd.Timestamp("20131031"),
    forecast_horizon=7,
    stride=2,
    retrain=False,
    verbose=True,
)
hbm_pred_stop = time.time()
pred_hudm_series = scaler.inverse_transform(pred_hudm_series)
hudson_m_test = scaler.inverse_transform(hudson_m_test)
display_forecast(pred_hudm_series, hudson_m_test, "monthly","N-BEATS Generic Model - Hudson Bay", start_date=pd.Timestamp("20131130"))
hbm_mod_time = hbm_mod_stop - hbm_mod_start
hbm_pred_time = hbm_pred_stop - hbm_pred_start
print("Train Time {}".format(hbm_mod_time)+"s"+ "\nPrediction Time: {}".format(hbm_pred_time)+"s")
#%%
# QUARTERLY YEAR-OVER-YEAR FORECASTING
baf_q = baffin_melt['BAFFIN SEA ICE EXTENT'].groupby(pd.Grouper(freq="q")).mean()
baf_q = baf_q.reset_index()
scaler = Scaler()
baf_q_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            baf_q, time_col="Date", value_cols="BAFFIN SEA ICE EXTENT")).astype(np.float32)
baf_q_train, baf_q_val = baf_q_series.split_after(pd.Timestamp("20070630"))
baf_q_val, baf_q_test = baf_q_val.split_after(pd.Timestamp("20131031"))
ba_q_mod_start = time.time()
baf_q_model = q_model_nbeats.fit(baf_q_train, val_series=baf_q_val, verbose=True)
ba_q_mod_stop = time.time()
ba_q_pred_start = time.time()
pred_q_series = baf_q_model.historical_forecasts(
    baf_q_series,
    start=pd.Timestamp("20140331"),
    forecast_horizon=4,
    stride=1,
    retrain=False,
    verbose=True,
)
ba_q_pred_stop = time.time()
pred_q_series = scaler.inverse_transform(pred_q_series)
baf_q_test = scaler.inverse_transform(baf_q_test)
display_forecast(pred_q_series, baf_q_test, "3-month", "N-BEATS Generic Model - Baffin", start_date=pd.Timestamp("20140331"))
ba_q_mod_time = ba_q_mod_stop - ba_q_mod_start
ba_q_pred_time = ba_q_pred_stop - ba_q_pred_start
print("Train Time {}".format(ba_q_mod_time)+"s"+ "\nPrediction Time: {}".format(ba_q_pred_time)+"s")
#%%
# Beaufort Sea Quarterly
beau_q = beau_melt['BEAUFORT SEA ICE EXTENT'].groupby(pd.Grouper(freq="q")).mean()
beau_q = beau_q.reset_index()
scaler = Scaler()
beau_q_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            beau_q, time_col="Date", value_cols="BEAUFORT SEA ICE EXTENT")).astype(np.float32)
beau_q_train, beau_q_val = beau_q_series.split_after(pd.Timestamp("20070630"))
beau_q_val, beau_q_test = beau_q_val.split_after(pd.Timestamp("20131031"))

be_q_mod_start = time.time()
beau_q_model = q_model_nbeats.fit(beau_q_train, val_series=beau_q_val, verbose=True)
be_q_mod_stop = time.time()
be_q_pred_start = time.time()
pred_beau_q_series = beau_q_model.historical_forecasts(
    beau_q_series,
    start=pd.Timestamp("20140331"),
    forecast_horizon=4,
    stride=1,
    retrain=False,
    verbose=True,
)
be_q_pred_stop = time.time()
pred_beau_q_series = scaler.inverse_transform(pred_beau_q_series)
beau_q_test = scaler.inverse_transform(beau_q_test)
display_forecast(pred_beau_q_series, beau_q_test, "3-month", "N-BEATS Generic Model - Beaufort Sea", start_date=pd.Timestamp("20140331"))
be_q_mod_time = be_q_mod_stop - be_q_mod_start
be_q_pred_time = be_q_pred_stop - be_q_pred_start
print("Train Time {}".format(be_q_mod_time)+"s"+ "\nPrediction Time: {}".format(be_q_pred_time)+"s")

#%%
# Canadian Archipelago Quarterly
ca_q = canarc_melt['CAN. ARCH. SEA ICE EXTENT'].groupby(pd.Grouper(freq="q")).mean()
ca_q = ca_q.reset_index()
scaler = Scaler()
ca_q_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            ca_q, time_col="Date", value_cols="CAN. ARCH. SEA ICE EXTENT")).astype(np.float32)
ca_q_train, ca_q_val = ca_q_series.split_after(pd.Timestamp("20070630"))
ca_q_val, ca_q_test = ca_q_val.split_after(pd.Timestamp("20131031"))

ca_q_mod_start = time.time()
ca_q_model = q_model_nbeats.fit(ca_q_train, val_series=ca_q_val, verbose=True)
ca_q_mod_stop = time.time()
ca_q_pred_start = time.time()
pred_ca_q_series = ca_q_model.historical_forecasts(
    ca_q_series,
    start=pd.Timestamp("20140331"),
    forecast_horizon=4,
    stride=1,
    retrain=False,
    verbose=True,
)
ca_q_pred_stop = time.time()
pred_ca_q_series = scaler.inverse_transform(pred_ca_q_series)
ca_q_test = scaler.inverse_transform(ca_q_test)
display_forecast(pred_ca_q_series, ca_q_test, "3-month", "N-BEATS Generic Model - Canadian Archipelago", start_date=pd.Timestamp("20140331"))
ca_q_mod_time = ca_q_mod_stop - ca_q_mod_start
ca_q_pred_time = ca_q_pred_stop - ca_q_pred_start
print("Train Time {}".format(ca_q_mod_time)+"s"+ "\nPrediction Time: {}".format(ca_q_pred_time)+"s")
#%%
# Hudson Bay Quarterly

hb_q = hudson_melt['HUDSON BAY SEA ICE EXTENT'].groupby(pd.Grouper(freq="q")).mean()
hb_q = hb_q.reset_index()
scaler = Scaler()
hb_q_series = scaler.fit_transform(
        TimeSeries.from_dataframe(
            hb_q, time_col="Date", value_cols="HUDSON BAY SEA ICE EXTENT")).astype(np.float32)
hb_q_train, hb_q_val = hb_q_series.split_after(pd.Timestamp("20070630"))
hb_q_val, hb_q_test = hb_q_val.split_after(pd.Timestamp("20131031"))

hud_q_mod_start = time.time()
hb_q_model = q_model_nbeats.fit(hb_q_train, val_series=hb_q_val, verbose=True)
hud_q_mod_stop = time.time()
hud_q_pred_start = time.time()
pred_hb_q_series = hb_q_model.historical_forecasts(
    hb_q_series,
    start=pd.Timestamp("20140331"),
    forecast_horizon=4,
    stride=1,
    retrain=False,
    verbose=True,
)
hud_q_pred_stop = time.time()
pred_hb_q_series = scaler.inverse_transform(pred_hb_q_series)
hb_q_test = scaler.inverse_transform(hb_q_test)
hb_q_train = scaler.inverse_transform(hb_q_train)
display_forecast(pred_hb_q_series, hb_q_test, forecast_type="3-month", chartname="N-BEATS Generic Model - Hudson Bay", start_date=pd.Timestamp("20140331"))
hud_q_mod_time = hud_q_mod_stop - hud_q_mod_start
hud_q_pred_time = hud_q_pred_stop - hud_q_pred_start
print("Train Time {}".format(hud_q_mod_time)+"s"+ "\nPrediction Time: {}".format(hud_q_pred_time)+"s")
