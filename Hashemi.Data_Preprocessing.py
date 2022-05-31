#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# # Data Ingestion and Exploration

# In[3]:


sigvol = pd.read_csv('significantvolcanoeruptions.csv')
print(sigvol.head())


# In[215]:


# drop first row as its blank
sigvol = sigvol.drop([0])
print(sigvol.head())


# In[216]:


sigvol.info()


# In[217]:


sigvol.isnull().sum()


# # Data Cleaning

# ## Replace missing VEI values with mean

# In[218]:


# Replace missing values in VEI with the mean of remaining items
VEImean = round(sigvol['Volcano Explosivity Index (VEI)'].mean())
print(VEImean)
sigvol['Volcano Explosivity Index (VEI)'] = sigvol['Volcano Explosivity Index (VEI)'].fillna(VEImean)
print(sigvol['Volcano Explosivity Index (VEI)'].value_counts())


# ## Replace Tsunami and Earthquake data

# In[219]:


# replace missing data with 0 and valid data with 1 in Tsunami and Earthquake
sigvol['Associated Tsunami?'] = sigvol['Associated Tsunami?'].fillna(0)
sigvol['Associated Tsunami?'].replace({"TSU": 1}, inplace=True)

sigvol['Associated Earthquake?'] = sigvol['Associated Earthquake?'].fillna(0)
sigvol['Associated Earthquake?'].replace({"EQ": 1}, inplace=True)


# In[220]:


print(sigvol['Associated Tsunami?'].value_counts())
print(sigvol['Associated Earthquake?'].value_counts())


# ## Discover correlation to pick columns to do prediction with

# In[221]:


# determine columns the correlate with DEATHS column
sigvol.corr()


# Columns with correlation:
# 
# Tsunami, Longitude and VEI chosen to keep. Year was considered but the reason for correlation with Year would be because the development of record keeping rather than a reason behind the volcano erruption itself.   
# Longitude mean be correlated to define location on earth and VEI definitely is a good option. Tsunami I was on the fence about because of its imbalance but decided to keep it.
# 

# In[222]:


# Remove all other columns except DEATHS, Longitude, Tsunami and VEI.
msigvol = sigvol[['Associated Tsunami?', 'Longitude', 'Volcano Explosivity Index (VEI)', 'DEATHS']].copy()


# In[223]:


print(msigvol.head())


# ## Prep data for Linear Regression

# In[224]:


# separate rows having null or missing data for prediction
testd = msigvol[msigvol['DEATHS'].isnull()]
print(testd.head())
print(testd.shape)


# In[225]:


# drop null values from orig df
msigvol = msigvol.dropna()
print(msigvol)
print(msigvol.shape)


# In[226]:


y_train = msigvol['DEATHS']
X_train = msigvol.drop("DEATHS", axis=1)
X_test = testd.drop("DEATHS", axis=1)
print(X_train.head())


# ## Model training and results

# In[227]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[228]:


y_pred = lr.predict(X_test)
sigvol.loc[sigvol.DEATHS.isnull(), 'DEATHS'] = y_pred


# In[229]:


print(sigvol['DEATHS'])
print(min(sigvol['DEATHS']))


# ## Clean up resulting DEATHS data to replace negatives with 0

# In[230]:


# convert the negative values for DEATHS to zeros
num = sigvol['DEATHS']._get_numeric_data()
num[num < 0] = 0

sigvol['DEATHS'] = sigvol['DEATHS'].astype(int)
print(sigvol['DEATHS'])
print(min(sigvol['DEATHS']))


# In[231]:


print(sigvol.head())


# ## Create new clean CSV

# In[234]:


sigvol.to_csv('sigvol-clean.csv', index=False)


# In[ ]:


y_train2 = msigvol['DEATHS']
X_train2 = msigvol.drop("DEATHS", axis=1)
X_test2 = testd.drop("DEATHS", axis=1)
print(X_train.head())


# In[ ]:




