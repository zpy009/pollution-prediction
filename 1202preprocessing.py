
# coding: utf-8

# # 读数据

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from IPython.core.interactiveshell import InteractiveShell 

from datetime import datetime

# In[2]:


InteractiveShell.ast_node_interactivity = "all"
loc_grid=pd.read_csv('Beijing_grid_weather_station.csv',header=None)
aiq1804=pd.read_csv('aiqQuality_201804.csv',header='infer')
aiq1805=pd.read_csv('airq1805.csv',header='infer')
airq17011801=pd.read_csv('airQuality_201701-201801.csv',header='infer')
airq180203=pd.read_csv('airQuality_201802-201803.csv',header='infer')
gridw17011803=pd.read_csv('gridWeather_201701-201803.csv',header='infer')
gridw1804=pd.read_csv('gridWeather_201804.csv',header='infer')
gridw1805=pd.read_csv('gridWeather_20180501-20180502.csv',header='infer')
loc_info=pd.read_csv('location_info.csv',header='infer')
ow17011801=pd.read_csv('observedWeather_201701-201801.csv',header='infer')
ow18021803=pd.read_csv('observedWeather_201802-201803.csv',header='infer')
ow1804=pd.read_csv('observedWeather_201804.csv',header='infer')
ow1805=pd.read_csv('observedWeather_20180501-20180502.csv',header='infer')


# In[2]:


aiq1805.head()


# In[3]:



aiq1804=aiq1804.drop(columns=['id'])
aiq1804.columns.values

aiq1804.columns = ['stationId', 'utc_time', 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
aiq1804.columns.values

airq= pd.concat([airq17011801,airq180203,aiq1804,aiq1805], axis=0,sort=False)

airq.to_csv('airquality.csv',index=False)

print(gridw17011803.columns.values)
print(gridw1804.columns.values)
print(gridw1805.columns.values)

loc_grid.columns=['stationName','latitude','longitude']

gridw1804=gridw1804.drop(columns=['id'])
gridw1805=gridw1805.drop(columns=['id'])
####    17年的数据少了一部分      没有‘’weather    可以考虑先预测 weather   

gridw1804.columns=['stationName', 'utc_time','weather','temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph']
gridw1805.columns=['stationName', 'utc_time','weather','temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph']

#先随便predict 一下grid的   weather吧  （虽然不知道用不用得上）？    不行......先hold住吧   太少了   没法预测     一个月的天气VS  全年天气、、、、

grid2=pd.concat([gridw1804,gridw1805], axis=0,sort=False)
grid25= pd.merge(grid2, loc_grid, how='left')

gridw17011803.head()
grid1= pd.merge(gridw17011803, loc_grid, how='left')

grid25.head()

# 新一轮hold住grid数据     随便可视化一下loc_info
#问题：    1     grid——location 很多 
                    #    2    observation     只有18个  

grid=pd.concat([grid1,grid25], axis=0,sort=False)

grid.tail()

grid.to_csv('gridweather.csv',index=False)


# In[4]:


grid.describe()


# # observed 的data 有异常值999 之类的

# In[5]:


file=[ow17011801,ow18021803,ow1804,ow1805]
for i in file:
    i.describe()
    i.isnull().sum()
    i.info()
    i.columns.values

ow1804=ow1804.drop(columns='id')
ow1805=ow1805.drop(columns='id')
ow1804.columns=['station_id', 'utc_time', 'weather', 'temperature', 'pressure','humidity', 'wind_speed', 'wind_direction']
ow1805.columns=['station_id', 'utc_time', 'weather', 'temperature', 'pressure','humidity', 'wind_speed', 'wind_direction']

ow2=pd.concat([ow18021803,ow1804,ow1805], axis=0,sort=False)
loc_info.columns=['station_id','longitude','latitude','type']


ow17011801['station_id']=ow17011801['station_id'].str.replace('meo','aq')
ow2['station_id']=ow2['station_id'].str.replace('meo','aq')

loc_obser=pd.pivot_table(ow17011801,index=['station_id'],values=['latitude','longitude'])

loc_obser.head()

loc_obser.to_csv('location_observation.csv',index=True)

loc_obser.columns


# In[6]:


loc_info2=loc_info.round({'latitude': 1, 'longitude': 1})
loc_merge=pd.merge(loc_info2,loc_grid,how='left')
loc_merge.to_csv('location_summary.csv',index=False)


# In[7]:


loc_merge2=pd.merge(loc_merge,loc_info,how='left',on='station_id')


# In[8]:


loc_info.head()
loc_info2.head()
loc_merge2.head()


# In[9]:


loc_merge2=loc_merge2.drop(columns=['longitude_x','latitude_x','type_x'])
loc_merge2.head()


tmp=np.array([loc_merge2['longitude_y'],loc_merge2['latitude_y']]).T
#调用python关于机器学习sklearn库中的KMeans
from sklearn.cluster import KMeans
#设置分为18类，并训练数据
kms=KMeans(n_clusters=18)
y=kms.fit_predict(tmp)
y
#将分类结果以散点图形式展示
fig = plt.figure(figsize=(10,6))
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.scatter(loc_merge2['longitude_y'], loc_merge2['latitude_y'], c=y, marker='x')   
plt.title("Kmeans-location Data")   
plt.show() 


# In[11]:


loc_merge2['loc_cluster']=y

loc_merge2.head()

# 保留一位小数之后呢就可以把他们merge起来啦

# 开始处理重复值   replace by均值 啥的       

## 从385420 变成 378945条

airq=airq.drop_duplicates()

# 重复的都是整条重复的 ....不用想replace by mean ....

airq.isnull().sum()

loc_merge2.columns

loc_merge2.columns=['stationId','stationName', 'longitude', 'latitude', 'type','loc_cluster']

airq2=pd.merge(airq,loc_merge2,how='left')

airq2.columns=['stationId', 'utc_time', 'PM25', 'PM10', 'NO2', 'CO', 'O3', 'SO2',       'stationName', 'longitude', 'latitude', 'type', 'loc_cluster']


# In[12]:


full_air=pd.merge(airq2,grid,how='left',on=['stationName', 'utc_time'])
full_air=full_air.drop(columns=['longitude_y','latitude_y'])
full_air.isnull().sum()


# In[13]:


full_air.columns=['stationId', 'utc_time', 'PM25', 'PM10', 'NO2', 'CO', 'O3', 'SO2',       'stationName', 'longitude', 'latitude', 'type', 'loc_cluster',       'temperature', 'pressure', 'humidity', 'wind_direction',       'wind_speed', 'weather']


# In[14]:


full_air['year'], full_air['month'],full_air['date']= full_air['utc_time'].str.split('-', 2).str
full_air['date']=full_air['date'].str[:2]
full_air['time']=full_air['utc_time'].str[10:13]
full_air[['year','month','date','time']] = full_air[['year','month','date','time']] .apply(pd.to_numeric)
full_air['utc_time']= [datetime.strptime(x,'%Y-%m-%d %H:%M:%S') for x in full_air['utc_time']]
full_air['week']=full_air['utc_time'].dt.dayofweek
def weekday(num):
    if num>4:
        return 1
    else: 
        return 0


# In[15]:


full_air['weekday']=full_air['week'].apply(weekday)


# In[22]:


full_air.columns


# # 思想：1、fill the part-missing value(not the  time-missing)
#                                                                        #1.1fill the pollution and weather data with the median (loc_cluster, utc_time)
#                                                                        #1.2fill the pollution and weather with the median(stationId,  time   year   month  )
#                                                                        #原则：从小的范围  fill 到大的范围  尽可能的保留更加准确的值 

# In[21]:


# 设索引
full_air.set_index(['loc_cluster','utc_time'], inplace=True)
####PM25
pm25_median = full_air.groupby(['loc_cluster','utc_time']).PM25.median()
full_air.PM25.fillna(pm25_median, inplace=True)

##PM10
PM10_median = full_air.groupby(['loc_cluster','utc_time']).PM10.median()
full_air.PM10.fillna(PM10_median, inplace=True)

#NO2
NO2_median = full_air.groupby(['loc_cluster','utc_time']).NO2.median()
full_air.NO2.fillna(NO2_median, inplace=True)

#CO
CO_median = full_air.groupby(['loc_cluster','utc_time']).CO.median()
full_air.CO.fillna(CO_median, inplace=True)

#O3
O3_median = full_air.groupby(['loc_cluster','utc_time']).O3.median()
full_air.O3.fillna(O3_median, inplace=True)

#SO2
SO2_median = full_air.groupby(['loc_cluster','utc_time']).SO2.median()
full_air.SO2.fillna(SO2_median, inplace=True)

#temp
temperature_median = full_air.groupby(['loc_cluster','utc_time']).temperature.median()
full_air.temperature.fillna(temperature_median, inplace=True)

#presure
pressure_median = full_air.groupby(['loc_cluster','utc_time']).pressure.median()
full_air.pressure.fillna(pressure_median, inplace=True)

####humidity            3922
humidity_median = full_air.groupby(['loc_cluster','utc_time']).humidity.median()
full_air.humidity.fillna(humidity_median, inplace=True)

#wind_direction
wind_direction_median = full_air.groupby(['loc_cluster','utc_time']).wind_direction.median()
full_air.wind_direction.fillna(wind_direction_median, inplace=True)

#wind_speed_kph
wind_speed_median = full_air.groupby(['loc_cluster','utc_time']).wind_speed.median()
full_air.wind_speed.fillna(wind_speed_median, inplace=True)

full_air.reset_index(inplace=True)


# In[23]:


full_air.isnull().sum()


# # 第二次fillna   

# In[24]:


# 设索引     stationId,  utc_time   year   month     ['stationId']

full_air.set_index(['stationId','time','year','month'], inplace=True)
####PM25
pm25_median = full_air.groupby(['stationId','time','year','month']).PM25.median()
full_air.PM25.fillna(pm25_median, inplace=True)

##PM10
PM10_median = full_air.groupby(['stationId','time','year','month']).PM10.median()
full_air.PM10.fillna(PM10_median, inplace=True)

#NO2
NO2_median = full_air.groupby(['stationId','time','year','month']).NO2.median()
full_air.NO2.fillna(NO2_median, inplace=True)

#CO
CO_median = full_air.groupby(['stationId','time','year','month']).CO.median()
full_air.CO.fillna(CO_median, inplace=True)

#O3
O3_median = full_air.groupby(['stationId','time','year','month']).O3.median()
full_air.O3.fillna(O3_median, inplace=True)

#SO2
SO2_median = full_air.groupby(['stationId','time','year','month']).SO2.median()
full_air.SO2.fillna(SO2_median, inplace=True)

#temp
temperature_median = full_air.groupby(['stationId','time','year','month']).temperature.median()
full_air.temperature.fillna(temperature_median, inplace=True)

#presure
pressure_median = full_air.groupby(['stationId','time','year','month']).pressure.median()
full_air.pressure.fillna(pressure_median, inplace=True)


####humidity            3922
humidity_median = full_air.groupby(['stationId','time','year','month']).humidity.median()
full_air.humidity.fillna(humidity_median, inplace=True)

#wind_direction
wind_direction_median = full_air.groupby(['stationId','time','year','month']).wind_direction.median()
full_air.wind_direction.fillna(wind_direction_median, inplace=True)

#wind_speed_kph
wind_speed_median = full_air.groupby(['stationId','time','year','month']).wind_speed.median()
full_air.wind_speed.fillna(wind_speed_median, inplace=True)

full_air.reset_index(inplace=True)
'''
PM25                8335
PM10                9745
NO2                 9083
CO                  9053
O3                  9195
SO2                 9053
'''
###just  left   ：   pm25      pm10        no2   co    o3     so2


# In[25]:


full_air.isnull().sum()### still 1300+    data need to be fill


# In[26]:


# 设索引     stationId,  utc_time   year   month     ['stationId']

full_air.set_index(['type','time','year','month'], inplace=True)
####PM25
pm25_median = full_air.groupby(['type','time','year','month']).PM25.median()
full_air.PM25.fillna(pm25_median, inplace=True)

##PM10
PM10_median = full_air.groupby(['type','time','year','month']).PM10.median()
full_air.PM10.fillna(PM10_median, inplace=True)

#NO2
NO2_median = full_air.groupby(['type','time','year','month']).NO2.median()
full_air.NO2.fillna(NO2_median, inplace=True)

#CO
CO_median = full_air.groupby(['type','time','year','month']).CO.median()
full_air.CO.fillna(CO_median, inplace=True)

#O3
O3_median = full_air.groupby(['type','time','year','month']).O3.median()
full_air.O3.fillna(O3_median, inplace=True)

#SO2
SO2_median = full_air.groupby(['type','time','year','month']).SO2.median()
full_air.SO2.fillna(SO2_median, inplace=True)

#temp
temperature_median = full_air.groupby(['type','time','year','month']).temperature.median()
full_air.temperature.fillna(temperature_median, inplace=True)

#presure
pressure_median = full_air.groupby(['type','time','year','month']).pressure.median()
full_air.pressure.fillna(pressure_median, inplace=True)


####humidity            3922
humidity_median = full_air.groupby(['type','time','year','month']).humidity.median()
full_air.humidity.fillna(humidity_median, inplace=True)

#wind_direction
wind_direction_median = full_air.groupby(['type','time','year','month']).wind_direction.median()
full_air.wind_direction.fillna(wind_direction_median, inplace=True)

#wind_speed_kph
wind_speed_median = full_air.groupby(['type','time','year','month']).wind_speed.median()
full_air.wind_speed.fillna(wind_speed_median, inplace=True)

full_air.reset_index(inplace=True)



full_air.to_pickle("fullair1_withNA111.pkl")

