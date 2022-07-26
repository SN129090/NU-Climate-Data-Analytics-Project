# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 23:04:14 2022

@author: Nicholls
"""

import pandas as pd
import matplotlib.pyplot as plt
#%%
plt.style.use("fivethirtyeight")
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

baffin_melt['monthstr'] = baffin_melt['month'].astype(str)
baffin_melt

#baffin_melt["season"] = ['spring' if baffin_melt[month] == 'Mar' ]
season = []
for x in baffin_melt['month']:
  if x in ['Dec','Jan','Feb']:
    season.append("Winter")
  elif x in ['Mar','Apr','May']:
    season.append("Early Spring")
  elif x in ['Jun']:
    season.append("Spring")
  elif x in ['Jul', 'Aug','Sep']:
    season.append("Summer")
  else:
    season.append("Fall")

baffin_melt['season'] = season
baffin_esp = baffin_melt.drop(baffin_melt[baffin_melt.season != "Early Spring"].index)
baffin_spr = baffin_melt.drop(baffin_melt[baffin_melt.season != "Spring"].index)
baffin_sum = baffin_melt.drop(baffin_melt[baffin_melt.season != "Summer"].index)
baffin_fall = baffin_melt.drop(baffin_melt[baffin_melt.season != "Fall"].index)
baffin_win = baffin_melt.drop(baffin_melt[baffin_melt.season != "Winter"].index)
baffin_esp_y = baffin_esp['BAFFIN SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
baffin_spr_y = baffin_spr['BAFFIN SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
baffin_sum_y = baffin_sum['BAFFIN SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
baffin_fall_y = baffin_fall['BAFFIN SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
baffin_win_y = baffin_win['BAFFIN SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
plt.figure(figsize=(16,8))
plt.title("Baffin Sea Annual Seasonal Comparison")
baffin_esp_y.plot(legend=True, label="Early Spring")
baffin_spr_y.plot(legend=True, label="Spring")
baffin_sum_y.plot(legend=True, label="Summer")
baffin_fall_y.plot(legend=True, label="Fall")
baffin_win_y.plot(legend=True, label="Winter")
#%%

beau_melt['monthstr'] = beau_melt['month'].astype(str)
season = []
for x in beau_melt['month']:
  if x in ['Dec','Jan','Feb']:
    season.append("Winter")
  elif x in ['Mar','Apr','May']:
    season.append("Early Spring")
  elif x in ['Jun']:
    season.append("Spring")
  elif x in ['Jul', 'Aug','Sep']:
    season.append("Summer")
  else:
    season.append("Fall")

beau_melt['season'] = season
beau_esp = beau_melt.drop(beau_melt[beau_melt.season != "Early Spring"].index)
beau_spr = beau_melt.drop(beau_melt[beau_melt.season != "Spring"].index)
beau_sum = beau_melt.drop(beau_melt[beau_melt.season != "Summer"].index)
beau_fall = beau_melt.drop(beau_melt[beau_melt.season != "Fall"].index)
beau_win = beau_melt.drop(beau_melt[beau_melt.season != "Winter"].index)
beau_esp_y = beau_esp['BEAUFORT SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
beau_spr_y = beau_spr['BEAUFORT SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
beau_sum_y = beau_sum['BEAUFORT SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
beau_fall_y = beau_fall['BEAUFORT SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
beau_win_y = beau_win['BEAUFORT SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
plt.figure(figsize=(12,8))
plt.title("Beaufort Sea Annual Seasonal Comparison")
beau_esp_y.plot(legend=True, label="Early Spring")
beau_spr_y.plot(legend=True, label="Spring")
beau_sum_y.plot(legend=True, label="Summer")
beau_fall_y.plot(legend=True, label="Fall")
beau_win_y.plot(legend=True, label="Winter")
#%%

canarc_melt['monthstr'] = canarc_melt['month'].astype(str)
season = []
for x in canarc_melt['month']:
  if x in ['Dec','Jan','Feb']:
    season.append("Winter")
  elif x in ['Mar','Apr','May']:
    season.append("Early Spring")
  elif x in ['Jun']:
    season.append("Spring")
  elif x in ['Jul', 'Aug','Sep']:
    season.append("Summer")
  else:
    season.append("Fall")

canarc_melt['season'] = season
ca_esp = canarc_melt.drop(canarc_melt[canarc_melt.season != "Early Spring"].index)
ca_spr = canarc_melt.drop(canarc_melt[canarc_melt.season != "Spring"].index)
ca_sum = canarc_melt.drop(canarc_melt[canarc_melt.season != "Summer"].index)
ca_fall = canarc_melt.drop(canarc_melt[canarc_melt.season != "Fall"].index)
ca_win = canarc_melt.drop(canarc_melt[canarc_melt.season != "Winter"].index)
ca_esp_y = ca_esp['CAN. ARCH. SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
ca_spr_y = ca_spr['CAN. ARCH. SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
ca_sum_y = ca_sum['CAN. ARCH. SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
ca_fall_y = ca_fall['CAN. ARCH. SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
ca_win_y = ca_win['CAN. ARCH. SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
plt.figure(figsize=(12,8))
plt.title("Canadian Archipelago Annual Seasonal Comparison")
ca_esp_y.plot(legend=True, label="Early Spring")
ca_spr_y.plot(legend=True, label="Spring")
ca_sum_y.plot(legend=True, label="Summer")
ca_fall_y.plot(legend=True, label="Fall")
ca_win_y.plot(legend=True, label="Winter")
#%%

hudson_melt['monthstr'] = hudson_melt['month'].astype(str)
season = []
for x in hudson_melt['month']:
  if x in ['Dec','Jan','Feb']:
    season.append("Winter")
  elif x in ['Mar','Apr','May']:
    season.append("Early Spring")
  elif x in ['Jun']:
    season.append("Spring")
  elif x in ['Jul', 'Aug','Sep']:
    season.append("Summer")
  else:
    season.append("Fall")

hudson_melt['season'] = season
hud_esp = hudson_melt.drop(hudson_melt[hudson_melt.season != "Early Spring"].index)
hud_spr = hudson_melt.drop(hudson_melt[hudson_melt.season != "Spring"].index)
hud_sum = hudson_melt.drop(hudson_melt[hudson_melt.season != "Summer"].index)
hud_fall = hudson_melt.drop(hudson_melt[hudson_melt.season != "Fall"].index)
hud_win = hudson_melt.drop(hudson_melt[hudson_melt.season != "Winter"].index)
hud_esp_y = hud_esp['HUDSON BAY SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
hud_spr_y = hud_spr['HUDSON BAY SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
hud_sum_y = hud_sum['HUDSON BAY SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
hud_fall_y = hud_fall['HUDSON BAY SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
hud_win_y = hud_win['HUDSON BAY SEA ICE EXTENT'].groupby(pd.Grouper(freq="Y")).mean()
plt.figure(figsize=(12,8))
plt.title("Hudson Bay Annual Seasonal Comparison")
hud_esp_y.plot(legend=True, label="Early Spring")
hud_spr_y.plot(legend=True, label="Spring")
hud_sum_y.plot(legend=True, label="Summer")
hud_fall_y.plot(legend=True, label="Fall")
hud_win_y.plot(legend=True, label="Winter")
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
baff_m.drop(index=baff_m.index[0], axis=0, inplace=True)
baff_m.drop(index=baff_m.index[0], axis=0, inplace=True)
baf_q = baffin_melt['BAFFIN SEA ICE EXTENT'].groupby(pd.Grouper(freq="q")).mean()
beau_q = beau_melt['BEAUFORT SEA ICE EXTENT'].groupby(pd.Grouper(freq="q")).mean()
canarc_q = canarc_melt['CAN. ARCH. SEA ICE EXTENT'].groupby(pd.Grouper(freq="q")).mean()
hudson_q = hudson_melt['HUDSON BAY SEA ICE EXTENT'].groupby(pd.Grouper(freq="q")).mean()
#%%
#Winter Levels
plt.figure(figsize=(12,8))
plt.title("Winter Annual Regional Sea Ice Extent Comparison")
ca_win_y.plot(legend=True, label="Canadian Archipelago")
beau_win_y.plot(legend=True, label="Beaufort Sea")
baffin_win_y.plot(legend=True, label="Baffin")
hud_win_y.plot(legend=True, label="Hudson Bay")
#%%
# Summer Levels
plt.figure(figsize=(12,8))
plt.title("Summer Annual Regional Sea Ice Extent Comparison")
ca_sum_y.plot(legend=True, label="Canadian Archipelago")
beau_sum_y.plot(legend=True, label="Beaufort Sea")
baffin_sum_y.plot(legend=True, label="Baffin")
hud_sum_y.plot(legend=True, label="Hudson Bay")
#%%
# Fall Levels
plt.figure(figsize=(12,8))
plt.title("Fall Annual Regional Sea Ice Extent Comparison")
ca_fall_y.plot(legend=True, label="Canadian Archipelago")
beau_fall_y.plot(legend=True, label="Beaufort Sea")
baffin_fall_y.plot(legend=True, label="Baffin")
hud_fall_y.plot(legend=True, label="Hudson Bay")
#%%
#Early Spring Levels
plt.figure(figsize=(12,8))
plt.title("Early Spring Annual Regional Sea Ice Extent Comparison")
ca_esp_y.plot(legend=True, label="Canadian Archipelago")
beau_esp_y.plot(legend=True, label="Beaufort Sea")
baffin_esp_y.plot(legend=True, label="Baffin")
hud_esp_y.plot(legend=True, label="Hudson Bay")
#%%
#Spring Levels
plt.figure(figsize=(12,8))
plt.title("Spring Annual Regional Sea Ice Extent Comparison")
ca_spr_y.plot(legend=True, label="Canadian Archipelago")
beau_spr_y.plot(legend=True, label="Beaufort Sea")
baffin_spr_y.plot(legend=True, label="Baffin")
hud_spr_y.plot(legend=True, label="Hudson Bay")
#%% ACF/PACF Plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(baffin_melt['BAFFIN SEA ICE EXTENT'], lags=40,title="Baffin Sea Ice Daily Autocorrelation", alpha = 0.5,auto_ylims=True)
plot_acf(baff_m, lags=40,title="Baffin Sea Ice Monthly Autocorrelation", alpha = 0.5, auto_ylims=True)
plot_acf(baf_q, lags=40,title="Baffin Sea Ice Quarterly Autocorrelation", alpha = 0.5, auto_ylims=True)
plot_pacf(baffin_melt['BAFFIN SEA ICE EXTENT'], lags=40,title="Baffin Sea Ice Daily PACF", alpha = 0.5,auto_ylims=True)
plot_pacf(baff_m, lags=40,title="Baffin Sea Ice Monthly PACF", alpha = 0.5, auto_ylims=True)
plot_pacf(baf_q, lags=40,title="Baffin Sea Ice Quarterly PACF", alpha = 0.5, auto_ylims=True)


plot_acf(beau_melt['BEAUFORT SEA ICE EXTENT'], lags=40,title="Beaufort Sea Ice Daily Autocorrelation", alpha = 0.5, auto_ylims=True)
plot_acf(beau_m, lags=40,title="Beaufort Sea Ice Monthly Autocorrelation", alpha = 0.5, auto_ylims=True)
plot_acf(beau_q, lags=40,title="Beaufort Sea Ice Quarterly Autocorrelation", alpha = 0.5, auto_ylims=True)
plot_pacf(beau_melt['BEAUFORT SEA ICE EXTENT'], lags=40,title="Beaufort Sea Ice Daily PACF", alpha = 0.5, auto_ylims=True)
plot_pacf(beau_m, lags=40,title="Beaufort Sea Ice Monthly PACF", alpha = 0.5, auto_ylims=True)
plot_pacf(beau_q, lags=40,title="Beaufort Sea Ice Quarterly PACF", alpha = 0.5, auto_ylims=True)
#%%
plot_acf(canarc_melt['CAN. ARCH. SEA ICE EXTENT'], lags=40,title="Canadian Archipelago Sea Ice Daily Autocorrelation", alpha = 0.5, auto_ylims=True)
plot_acf(canarc_m, lags=40,title="Canadian Archipelago Sea Ice Monthly Autocorrelation", alpha = 0.5, auto_ylims=True)
plot_acf(canarc_q, lags=40,title="Canadian Archipelago Sea Ice Quarterly Autocorrelation", alpha = 0.5, auto_ylims=True)
plot_pacf(canarc_melt['CAN. ARCH. SEA ICE EXTENT'], lags=40,title="Canadian Archipelago Sea Ice Daily PACF", alpha = 0.5, auto_ylims=True)
plot_pacf(canarc_m, lags=40,title="Canadian Archipelago Sea Ice Monthly PACF", alpha = 0.5, auto_ylims=True)
plot_pacf(canarc_q, lags=40,title="Canadian Archipelago Sea Ice Quarterly PACF", alpha = 0.5, auto_ylims=True)
#%%
plot_acf(hudson_melt['HUDSON BAY SEA ICE EXTENT'], lags=40,title="Hudson Bay Sea Ice Daily Autocorrelation", alpha = 0.5, auto_ylims=True)
plot_acf(hudson_m, lags=40,title="Hudson Bay Sea Ice Monthly Autocorrelation", alpha = 0.5, auto_ylims=True)
plot_acf(hudson_q, lags=40,title="Hudson Bay Sea Ice Quarterly Autocorrelation", alpha = 0.5, auto_ylims=True)
plot_pacf(hudson_melt['HUDSON BAY SEA ICE EXTENT'], lags=40,title="Hudson Bay Sea Ice Daily PACF", alpha = 0.5, auto_ylims=True)
plot_pacf(hudson_m, lags=40,title="Hudson Bay Sea Ice Monthly PACF", alpha = 0.5, auto_ylims=True)
plot_pacf(hudson_q, lags=40,title="Hudson Bay Sea Ice Quarterly PACF", alpha = 0.5, auto_ylims=True)
#%%
plot_acf(canarc_m, lags=20,title="Canadian Archipelago Sea Ice Monthly Autocorrelation")
plot_acf(hudson_m, lags=20,title="Hudson Bay Sea Ice Monthly Autocorrelation")

plot_pacf(baff_m, lags=20,title="Baffin Sea Ice Monthly PACF")
plot_pacf(beau_m, lags=20,title="Beaufort Sea Ice Monthly PACF")
plot_pacf(canarc_m, lags=20,title="Canadian Archipelago Sea Ice Monthly PACF")
plot_pacf(hudson_m, lags=20,title="Hudson Bay Sea Ice Monthly PACF")
#%%
# Baffin Trends & Seasons - Daily
from statsmodels.tsa.seasonal import seasonal_decompose

Baffin_decomposed = seasonal_decompose(baffin_melt['BAFFIN SEA ICE EXTENT'],  
                            model ='multiplicative')
Baf_trend = Baffin_decomposed.trend
Baf_seasonal = Baffin_decomposed.seasonal
Baf_residual = Baffin_decomposed.resid
plt.figure(figsize=(8,8))
plt.subplot(411)
plt.title("Baffin Decompose - Daily")
plt.plot(baffin_melt['BAFFIN SEA ICE EXTENT'], label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(Baf_trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(Baf_seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(Baf_residual, label='Residual')
plt.legend(loc='upper left')

plt.show()
#%%
# Baffin Trends & Seasons - Monthly
from statsmodels.tsa.seasonal import seasonal_decompose 
Baffin_decomposed = seasonal_decompose(baff_m,  
                            model ='multiplicative')
Baf_trend = Baffin_decomposed.trend
Baf_seasonal = Baffin_decomposed.seasonal
Baf_residual = Baffin_decomposed.resid
plt.figure(figsize=(15,8))
plt.subplot(411)
plt.title("Baffin Decompose - Monthly")
plt.plot(baff_m, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(Baf_trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(Baf_seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(Baf_residual, label='Residual')
plt.legend(loc='upper left')
plt.show()
#%%
# Baffin Trends & Seasons - Quarterly
from statsmodels.tsa.seasonal import seasonal_decompose 
Baffin_decomposed = seasonal_decompose(baf_q,  
                            model ='multiplicative')
Baf_trend = Baffin_decomposed.trend
Baf_seasonal = Baffin_decomposed.seasonal
Baf_residual = Baffin_decomposed.resid
plt.figure(figsize=(15,8))
plt.subplot(411)
plt.title("Baffin Decompose - Quarterly")
plt.plot(baf_q, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(Baf_trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(Baf_seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(Baf_residual, label='Residual')
plt.legend(loc='upper left')
plt.show()
#%%
# Beaufort Sea Trends & Seasons - Daily
Beau_decomposed = seasonal_decompose(beau_melt['BEAUFORT SEA ICE EXTENT'],  
                            model ='multiplicative')
Beau_trend = Beau_decomposed.trend
Beau_seasonal = Beau_decomposed.seasonal
Beau_residual = Beau_decomposed.resid
plt.figure(figsize=(15,8))
plt.subplot(411)
plt.title("Beaufort Sea Decompose - Daily")
plt.plot(beau_melt['BEAUFORT SEA ICE EXTENT'], label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(Beau_trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(Beau_seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(Beau_residual, label='Residual')
plt.legend(loc='upper left')
plt.show()
#%%
# Beaufort Sea Trends & Seasons - Monthly
Beau_decomposed = seasonal_decompose(beau_m,  
                            model ='multiplicative')
Beau_trend = Beau_decomposed.trend
Beau_seasonal = Beau_decomposed.seasonal
Beau_residual = Beau_decomposed.resid
plt.figure(figsize=(15,8))
plt.subplot(411)
plt.title("Beaufort Sea Decompose - Monthly")
plt.plot(beau_m, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(Beau_trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(Beau_seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(Beau_residual, label='Residual')
plt.legend(loc='upper left')
plt.show()
#%%
# Beaufort Sea Trends & Seasons - Quarterly
Beau_decomposed = seasonal_decompose(beau_q,  
                            model ='multiplicative')
Beau_trend = Beau_decomposed.trend
Beau_seasonal = Beau_decomposed.seasonal 
Beau_residual = Beau_decomposed.resid
plt.figure(figsize=(15,8))
plt.subplot(411)
plt.title("Beaufort Sea Decompose - Quarterly")
plt.plot(beau_m, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(Beau_trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(Beau_seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(Beau_residual, label='Residual')
plt.legend(loc='upper left')
plt.show()
#%% Canadian Archipelago Trends & Seasons - Daily
CA_decomposed = seasonal_decompose(canarc_melt['CAN. ARCH. SEA ICE EXTENT'],  
                            model ='multiplicative')
CA_trend = CA_decomposed.trend
CA_seasonal = CA_decomposed.seasonal 
CA_residual = CA_decomposed.resid
plt.figure(figsize=(15,8))
plt.subplot(411)
plt.plot(canarc_melt['CAN. ARCH. SEA ICE EXTENT'], label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(CA_trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(CA_seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(CA_residual, label='Residual')
plt.legend(loc='upper left')
plt.show()
#%% Canadian Archipelago Trends & Seasons - Monthly
CA_decomposed = seasonal_decompose(canarc_m,  
                            model ='multiplicative')
CA_trend = CA_decomposed.trend
CA_seasonal = CA_decomposed.seasonal
CA_residual = CA_decomposed.resid
plt.figure(figsize=(15,8))
plt.subplot(411)
plt.plot(canarc_m, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(CA_trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(CA_seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(CA_residual, label='Residual')
plt.legend(loc='upper left')
plt.show()
#%% Canadian Archipelago Trends & Seasons - Quarterly
CA_decomposed = seasonal_decompose(canarc_q,  
                            model ='multiplicative')
CA_trend = CA_decomposed.trend
CA_seasonal = CA_decomposed.seasonal
CA_residual = CA_decomposed.resid
plt.figure(figsize=(15,8))
plt.subplot(411)
plt.plot(canarc_q, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(CA_trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(CA_seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(CA_residual, label='Residual')
plt.legend(loc='upper left')
plt.show()
#%% Hudson Bay - Daily
hud_decomposed = seasonal_decompose(hudson_melt['HUDSON BAY SEA ICE EXTENT'],  
                            model ='multiplicative')
hud_trend = hud_decomposed.trend
hud_seasonal = hud_decomposed.seasonal
hud_residual = hud_decomposed.resid
plt.figure(figsize=(15,8))
plt.subplot(411)
plt.plot(hudson_melt['HUDSON BAY SEA ICE EXTENT'], label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(hud_trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(hud_seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(hud_residual, label='Residual')
plt.legend(loc='upper left')
plt.show()
#%% Hudson Bay - Monthly
hud_decomposed = seasonal_decompose(hudson_m,  
                            model ='multiplicative')
hud_trend = hud_decomposed.trend
hud_seasonal = hud_decomposed.seasonal
hud_residual = hud_decomposed.resid
plt.figure(figsize=(15,8))
plt.subplot(411)
plt.plot(hudson_m, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(hud_trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(hud_seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(hud_residual, label='Residual')
plt.legend(loc='upper left')
plt.show()
#%% Hudson Bay - Quarterly
hud_decomposed = seasonal_decompose(hudson_q,  
                            model ='multiplicative')
hud_trend = hud_decomposed.trend
hud_seasonal = hud_decomposed.seasonal
hud_residual = hud_decomposed.resid

plt.figure(figsize=(15,8))
plt.subplot(411)
plt.plot(hudson_q, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(hud_trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(hud_seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(hud_residual, label='Residual')
plt.legend(loc='upper left')
plt.show()
