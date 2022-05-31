#!/usr/bin/env python
# coding: utf-8

# # Time series analysis and data analysis for the final project
# ## In this effort, I am going to perform time series analysis for VEI pattern and number of deaths in the first step, and then, do some data analysis and select a target dataset based on our goals.

# In[1]:


import pandas as pd
import numpy as np
from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# 
# ## Loading Data

# In[2]:


Data = pd.read_csv('sigvol-clean.csv')
print(Data.head())


# ## Let's see the change of number of deaths during the time

# In[4]:


def plot_df(Data, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    plt.savefig('Numberdeath.png')

plot_df(Data, x=Data.Year, y=Data.DEATHS, title='Number of Deaths')    


# ## And the change of VEI of deaths during the time

# In[4]:


plot_df(Data, x=Data.Year, y=Data["Volcano Explosivity Index (VEI)"], title='Volcano Explosively Index')   


# ## As we can see, the trend for anchient data does not seem accurate. For achieving better and more accurate trend, we are going to plot the trend for the year after year of 1000

# In[201]:


plot_df(Data[(Data['Year'] > 1000)], x=Data[(Data['Year'] > 1000)].Year, y=Data[(Data['Year'] > 1000)]["Volcano Explosivity Index (VEI)"], title='Volcano Explosivity Index (VEI) after 1000')  


# In[216]:


New_Data = Data[(Data['Year'] > 1900)]


# In[94]:


plot_df(Data[(Data['Year'] > 1000)], x=Data[(Data['Year'] > 1000)].Year, y=Data[(Data['Year'] > 1000)].DEATHS, title='Number of Deaths after 1000')  


# ## Now we want to see the linear coefficient between VEI and Latitude, Longitude and number of deaths

# In[202]:


import statsmodels.api as sm

y = Data['Volcano Explosivity Index (VEI)']
x = Data[['Latitude', 'Longitude', 'DEATHS']]

# add constant
x = sm.add_constant(x)

my_model = sm.OLS(y,x, missing='drop')
result = my_model.fit()
print(result.summary())


# ## Let's see the correlation between them

# In[44]:


Data['Volcano Explosivity Index (VEI)'].corr(Data['DEATHS'])


# ## The correlation between number of Deaths and VEI is 0.4 which can be assumed as a moderate correlation.

# In[45]:


Data['Volcano Explosivity Index (VEI)'].corr(Data['Associated Tsunami?'])


# In[46]:


Data['Volcano Explosivity Index (VEI)'].corr(Data['Associated Earthquake?'])


# In[97]:


Data['Volcano Explosivity Index (VEI)'].corr(Data['Latitude'])


# In[98]:


Data['Volcano Explosivity Index (VEI)'].corr(Data['Longitude'])


# ## Other correlations between VEI and other elements can not be considered as they are very low.

# In[104]:


features1=list(['Volcano Explosivity Index (VEI)','Longitude', 'Latitude', 'DEATHS'])

Data[features1].corr()


# # We are going to start to perform time series analysis for VEI and number of death in two different ways
# ## Method one:

# ## For VEI

# In[203]:


y = Data[['Year','Volcano Explosivity Index (VEI)']]
y = y.set_index('Year')
#Data['Volcano Explosivity Index (VEI)']
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 15, 12
rcParams['axes.labelsize'] = 16
rcParams['ytick.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
#decomposition = sm.tsa.seasonal_decompose(y, model='additive')
decomposition = seasonal_decompose(y['Volcano Explosivity Index (VEI)'].values, period=305)
#decompose_result = seasonal_decompose(y, model='multiplicative', period=1)
decomp = decomposition.plot()
decomp.suptitle('VEI', fontsize=22)


# ## For number of death

# In[204]:


y = Data[['Year','DEATHS']]
y = y.set_index('Year')
#Data['Volcano Explosivity Index (VEI)']
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 15, 12
rcParams['axes.labelsize'] = 16
rcParams['ytick.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
#decomposition = sm.tsa.seasonal_decompose(y, model='additive')
decomposition = seasonal_decompose(y['DEATHS'].values, period=305)
#decompose_result = seasonal_decompose(y, model='multiplicative', period=1)
decomp = decomposition.plot()
decomp.suptitle('Deaths', fontsize=22)


# ## I did not like the method one as the trends were not very clear.
# ## Second method for time series analysis:
# ## For VEI

# In[9]:


# import libraries
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
 
# Generate time-series data
total_duration = 650
step = 1
time = np.arange(0, total_duration, step)
 
# Period of the sinusoidal signal in seconds
T= 15
y = Data[['Year','Volcano Explosivity Index (VEI)']] 
# Period component
series_periodic = np.sin((2*np.pi/T)*time)
 
# Add a trend component
k0 = 2
k1 = 2
k2 = 0.05
k3 = 0.001
 
series_periodic = k0*series_periodic
series_trend    = k1*np.ones(len(time))+k2*time+k3*time**2
series          = y['Volcano Explosivity Index (VEI)'] 

# Set frequency using period in seasonal_decompose()
period = int(T/step)
results = seasonal_decompose(series, model='additive', freq=period)

trend_estimate    = results.trend
periodic_estimate = results.seasonal
residual          = results.resid
 
# Plot the time-series componentsplt.figure(figsize=(14,10))
plt.figure(figsize=(14,10))
plt.subplot(221)
plt.plot(series,label='Original time series', color='blue')
plt.plot(trend_estimate ,label='Trend of time series' , color='red')
plt.legend(loc='best',fontsize=20 , bbox_to_anchor=(0.90, -0.05))
plt.subplot(222)
plt.plot(trend_estimate,label='Trend of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(0.90, -0.05))
plt.subplot(223)
plt.plot(periodic_estimate,label='Seasonality of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(0.90, -0.05))
plt.subplot(224)
plt.plot(residual,label='Decomposition residuals of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(1.09, -0.05))
plt.tight_layout()
plt.suptitle('VEI', fontsize=20)
plt.savefig('decomposition.png')


# ## As we can see the trend is clear here (the red line). If we focous on the trend after the year of 2000 (after period of 600 on the x axis), we can see that the VEI has experienced an increasing trend. But if we want to cinsider the bigger picture and focuse on the VEI trend after the year of 1800  (after period of 350 on the x axis), the VEI showed a regular trend that had a sinusoidal trend.   

# ## Time series analysis for number of deaths

# In[11]:


from sklearn import preprocessing
x_train = Data[['Year','DEATHS']]
x_train = preprocessing.normalize(x_train)
print(x_train)

df = pd.DataFrame(data=x_train)


# In[13]:


# import libraries
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
 
# Generate time-series data
total_duration = 1000
step = 1
time = np.arange(0, total_duration, step)
 
# Period of the sinusoidal signal in seconds
T= 15
y = df
# Period component
series_periodic = np.sin((2*np.pi/T)*time)
 
# Add a trend component
k0 = 2
k1 = 2
k2 = 0.05
k3 = 0.001
 
series_periodic = k0*series_periodic
series_trend    = k1*np.ones(len(time))+k2*time+k3*time**2
series          = df.iloc[:,[1]]

# Set frequency using period in seasonal_decompose()
period = int(T/step)
results = seasonal_decompose(series, model='additive', freq=period)

trend_estimate    = results.trend
periodic_estimate = results.seasonal
residual          = results.resid
 
# Plot the time-series componentsplt.figure(figsize=(14,10))
plt.figure(figsize=(14,10))
plt.subplot(221)
plt.plot(series,label='Original time series', color='blue')
plt.plot(trend_estimate ,label='Trend of time series' , color='red')
plt.legend(loc='best',fontsize=20 , bbox_to_anchor=(0.90, -0.05))
plt.subplot(222)
plt.plot(trend_estimate,label='Trend of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(0.90, -0.05))
plt.subplot(223)
plt.plot(periodic_estimate,label='Seasonality of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(0.90, -0.05))
plt.subplot(224)
plt.plot(residual,label='Decomposition residuals of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(1.09, -0.05))
plt.suptitle('Deaths', fontsize=20)
plt.tight_layout()
plt.savefig('decomposition2.png')


# ##  For perfomring time series analysis for number of deaths, we had to normalize number of deaths. If we focous on the trend after the year of 1970 (after period of 500 on the x axis), we can see that the number of death has experienced no sensible change just once and then backed to its normal. 

# ## Now we want to perform time series analysis just for observations after 1900

# In[31]:


New_Data = Data[(Data['Year'] > 1990)]


# ## Time series analysis for VEI

# In[32]:


import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


# In[33]:


# Generate time-series data

total_duration = 100
step = 1
time = np.arange(0, total_duration, step)
 
# Period of the sinusoidal signal in seconds
T= 15
y = New_Data[['Year','Volcano Explosivity Index (VEI)']] 
# Period component
series_periodic = np.sin((2*np.pi/T)*time)
 
# Add a trend component
k0 = 2
k1 = 2
k2 = 0.05
k3 = 0.001
 
series_periodic = k0*series_periodic
series_trend    = k1*np.ones(len(time))+k2*time+k3*time**2
series          = y['Volcano Explosivity Index (VEI)'] 

# Set frequency using period in seasonal_decompose()
period = int(T/step)
results = seasonal_decompose(series, model='additive', freq=period)

trend_estimate    = results.trend
periodic_estimate = results.seasonal
residual          = results.resid
 
# Plot the time-series componentsplt.figure(figsize=(14,10))
plt.subplot(221)
plt.plot(series,label='Original time series', color='blue')
plt.plot(trend_estimate ,label='Trend of time series' , color='red')
plt.legend(loc='best',fontsize=20 , bbox_to_anchor=(0.90, -0.05))
plt.subplot(222)
plt.plot(trend_estimate,label='Trend of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(0.90, -0.05))
plt.subplot(223)
plt.plot(periodic_estimate,label='Seasonality of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(0.90, -0.05))
plt.subplot(224)
plt.plot(residual,label='Decomposition residuals of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(1.09, -0.05))
plt.tight_layout()
plt.savefig('decomposition3.png')


# In[34]:


total_duration = 100
step = 1
time = np.arange(0, total_duration, step)
T= 15
y = New_Data[['Year','Volcano Explosivity Index (VEI)']] 
# Period component
series_periodic = np.sin((2*np.pi/T)*time)
 
# Add a trend component
k0 = 2
k1 = 2
k2 = 0.05
k3 = 0.001
 
series_periodic = k0*series_periodic
series_trend    = k1*np.ones(len(time))+k2*time+k3*time**2
series          = y['Volcano Explosivity Index (VEI)'] 

# Set frequency using period in seasonal_decompose()
period = int(T/step)
results = seasonal_decompose(series, model='additive', freq=period)

trend_estimate    = results.trend
periodic_estimate = results.seasonal
residual          = results.resid
 
# Plot the time-series componentsplt.figure(figsize=(14,10))
plt.figure(figsize=(14,10))
plt.subplot(221)
plt.plot(series,label='Original time series', color='blue')
plt.plot(trend_estimate ,label='Trend of time series' , color='red')
plt.legend(loc='best',fontsize=20 , bbox_to_anchor=(0.90, -0.05))
plt.subplot(222)
plt.plot(trend_estimate,label='Trend of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(0.90, -0.05))
plt.subplot(223)
plt.plot(periodic_estimate,label='Seasonality of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(0.90, -0.05))
plt.subplot(224)
plt.plot(residual,label='Decomposition residuals of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(1.09, -0.05))
plt.tight_layout()
plt.savefig('decomposition3.png')


# ## Since we changed our dataset and wanted to see the time series analysis for our dataset after the year of 1900, the x axis starts from 338 rows which contains year of 1900.

# ## For number of death

# In[35]:


x_train = New_Data[['Year','DEATHS']]
x_train = preprocessing.normalize(x_train)
print(x_train)

df = pd.DataFrame(data=x_train)


# In[37]:


# Generate time-series data
total_duration = 100
step = 1
time = np.arange(0, total_duration, step)
 
# Period of the sinusoidal signal in seconds
T= 15
y = df
# Period component
series_periodic = np.sin((2*np.pi/T)*time)
 
# Add a trend component
k0 = 2
k1 = 2
k2 = 0.05
k3 = 0.001
 
series_periodic = k0*series_periodic
series_trend    = k1*np.ones(len(time))+k2*time+k3*time**2
series          = df.iloc[:,[1]]

# Set frequency using period in seasonal_decompose()
period = int(T/step)
results = seasonal_decompose(series, model='additive', freq=period)

trend_estimate    = results.trend
periodic_estimate = results.seasonal
residual          = results.resid
 
# Plot the time-series componentsplt.figure(figsize=(14,10))
plt.figure(figsize=(14,10))
plt.subplot(221)
plt.plot(series,label='Original time series', color='blue')
plt.plot(trend_estimate ,label='Trend of time series' , color='red')
plt.legend(loc='best',fontsize=20 , bbox_to_anchor=(0.90, -0.05))
plt.subplot(222)
plt.plot(trend_estimate,label='Trend of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(0.90, -0.05))
plt.subplot(223)
plt.plot(periodic_estimate,label='Seasonality of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(0.90, -0.05))
plt.subplot(224)
plt.plot(residual,label='Decomposition residuals of time series',color='blue')
plt.legend(loc='best',fontsize=20, bbox_to_anchor=(1.09, -0.05))
plt.tight_layout()
plt.savefig('decomposition2.png')


# ## Based on what we achieved in our time series analysis, I believe the trend of VEI has increased recent years and we can consider it as an element.
# 
# # Step two:
# ## Now let's explore our dataset more and target a goal for our final dataset.

# In[143]:


print(Data['DEATHS'].max()) 
print(Data['DEATHS'].min()) 


# In[155]:


print(Data['DEATHS'].mean()) 


# In[145]:


print(Data['Volcano Explosivity Index (VEI)'].max()) 
print(Data['Volcano Explosivity Index (VEI)'].min()) 


# In[146]:


print(Data['Volcano Explosivity Index (VEI)'].mean()) 


# In[147]:


print(Data['Volcano Explosivity Index (VEI)'].median()) 


# In[148]:


print(Data['Volcano Explosivity Index (VEI)'].mode()) 


# ## Based on what I learned, I would like to set a target dataset which has observations:
# ## 1- The VEI is more than 2.9 (the mean of VEI column) as a positive factor for our renewable power planet and less than 7 since after 7 I believe volanos can be very distructive for our planets equipments.
# ## 2- The number of death less than 834 (the mean of number of death). As the number of death and VEI have a moderate correlation with each other, if we want to have a few number of death, we lose some important locations with good heating potential (if we assume VEI number for evaluating this factor)
# ## 3- I just serached among the observation after the year of 0
# ## 4- I just consider those locations without tsunami and earthquake since these two can be considered as destructive factors for our renewable energies power planet.

# In[39]:


Target_Data = Data[(Data['DEATHS']<834) & (Data['Year']>0) & (Data['Associated Tsunami?']<=0)]


# In[16]:


Target_Data


# ## Now we sort our data based on each columns and see number of iteation for each elements

# In[17]:


Country = Target_Data.groupby('Country').size()
Country


# In[8]:


Target_Data['Country'].value_counts()


# In[173]:


Target_Data['Location'].value_counts()


# In[183]:


Target_Data['Latitude'].value_counts()


# In[181]:


Target_Data['Longitude'].value_counts()


# In[18]:


Target_Data['Country'].value_counts()[:10].sort_values(ascending=False)


# In[10]:


Target_Data['Location'].value_counts()[:10].sort_values(ascending=False)


# In[11]:


dff = Target_Data['Country'].value_counts()[:20]
dff.plot(kind = 'bar')
plt.ylabel('Number of iteration')
plt.xlabel('Country')
plt.title('20 countries With the higest number of iteration')
plt.show()


# In[12]:


dfff = Target_Data['Location'].value_counts()[:20]
dfff.plot(kind = 'bar')
plt.ylabel('Number of iteration')
plt.xlabel('Location')
plt.title('20 locations With the higest number of iteration')
plt.show()


# In[26]:


New_Data.to_csv('New_Data.csv', index=False)


# In[41]:


Target_Data.to_csv('Target_Data.csv', index=False)


# In[ ]:




