# -*- coding: utf-8 -*-
"""Arctic Sea Ice Data Description & Preparation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1InqcFV5NmfLP_NocNJAyvFa-pndYHy_v
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
plt.style.use("bmh")
plt.plot(baffin_melt['BAFFIN SEA ICE EXTENT'])
plt.ylabel('Sea Ice Extent (km^2)')
plt.xlabel('Year')
plt.title("Daily Sea Ice Extent - Baffin")
#%%
plt.plot(beau_melt['BEAUFORT SEA ICE EXTENT'])
plt.ylabel('Sea Ice Extent (km^2)')
plt.xlabel('Year')
plt.title("Daily Sea Ice Extent - Beaufort Sea")
#%%
plt.plot(canarc_melt['CAN. ARCH. SEA ICE EXTENT'])
plt.ylabel('Sea Ice Extent (km^2)')
plt.xlabel('Year')
plt.title("Daily Sea Ice Extent - Canadian Archipelago")
#%%
plt.plot(hudson_melt['HUDSON BAY SEA ICE EXTENT'])
plt.ylabel('Sea Ice Extent (km^2)')
plt.xlabel('Year')
plt.title("Daily Sea Ice Extent - Hudson Bay")
#%%
# Restoring month from date
baffin_melt['month'] = [d.strftime('%b') for d in baffin_melt.index]
beau_melt['month'] = [d.strftime('%b') for d in beau_melt.index]
canarc_melt['month'] = [d.strftime('%b') for d in canarc_melt.index]
hudson_melt['month'] = [d.strftime('%b') for d in hudson_melt.index]

#%%
sns.boxplot(x='BAFFIN SEA ICE EXTENT', data = baffin_melt)
#%%
sns.boxplot(x='month', y="BAFFIN SEA ICE EXTENT", data = baffin_melt)
#%%
sns.boxplot(x='Year', y="BAFFIN SEA ICE EXTENT", data = baffin_melt)
#%%
sns.boxplot(x='BEAUFORT SEA ICE EXTENT', data = beau_melt)
#%%
sns.boxplot(x='month', y="BEAUFORT SEA ICE EXTENT", data = beau_melt)
#%%
sns.boxplot(x='Year', y="BEAUFORT SEA ICE EXTENT", data = beau_melt)
#%%
sns.boxplot(x='CAN. ARCH. SEA ICE EXTENT', data = canarc_melt)
#%%
sns.boxplot(x='month', y="CAN. ARCH. SEA ICE EXTENT", data = canarc_melt)
#%%
sns.boxplot(x='Year', y="CAN. ARCH. SEA ICE EXTENT", data = canarc_melt)
#%%
sns.boxplot(x='HUDSON BAY SEA ICE EXTENT', data = hudson_melt)
#%%
sns.boxplot(x='month', y="HUDSON BAY SEA ICE EXTENT", data = hudson_melt)
#%%
sns.boxplot(x='Year', y="HUDSON BAY SEA ICE EXTENT", data = hudson_melt)
#%%
sns.histplot(data = baffin_melt, x='BAFFIN SEA ICE EXTENT')
#%%
sns.histplot(data = beau_melt, x='BEAUFORT SEA ICE EXTENT')
#%%
sns.histplot(data = canarc_melt, x='CAN. ARCH. SEA ICE EXTENT')
#%%
sns.histplot(data = hudson_melt, x='HUDSON BAY SEA ICE EXTENT')
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
baffin_melt

sns.boxplot(x='season', y="BAFFIN SEA ICE EXTENT", data = baffin_melt)
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


sns.boxplot(x='season', y="BEAUFORT SEA ICE EXTENT", data = beau_melt)
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


sns.boxplot(x='season', y="CAN. ARCH. SEA ICE EXTENT", data = canarc_melt)
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


sns.boxplot(x='season', y="HUDSON BAY SEA ICE EXTENT", data = hudson_melt)