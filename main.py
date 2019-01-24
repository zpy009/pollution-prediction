
# coding: utf-8

# In[1]:


from lightgbm import LGBMRegressor
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer

full_air= pkl.load(open("fullair1_withNA111.pkl", "rb"))

print('-----------load preprocessed data-----------------------')

print('-----------divide into stations ----preoprocess for model lgb   again---------')
def merge_loc(station):#station='fangshan_aq'
    station_data =full_air[full_air['stationId'] == station].drop(['stationId','time', 'year', 'month','date','week','weather'], axis=1)
    print(station, len(station_data))
    #station_data.head()
    #station_data['utc_time']= [datetime.strptime(x,'%Y-%m-%d %H:%M:%S') for x in station_data['utc_time']]
    station_data.set_index(['utc_time'], inplace=True)
    station_data=station_data.resample('H').ffill(limit=24)
    #station_data=station_data.reset_index()
    #station_data.head()
    #station_data.isnull().sum()
    #station_data.to_csv('process_'+ station + '.csv', index=True)
    return station_data


full_air.columns

stations = set(full_air['stationId'])
full_stations={}
for station in stations:
    full_station=merge_loc(station)
    full_stations[station] =  full_station
    print(station,full_station.shape)

print('take a glance of pollutions')
# # 虽然是连续的 但是    起伏其实很大的。

# In[3]

station='wanshouxigong_aq'
pic=full_stations[station]
# specify columns to plot  'PM10', 'NO2', 'CO', 'O3', 'SO2',
features = ['PM25',  'temperature', 'pressure',       'humidity', 'wind_direction', 'wind_speed']
pic_start_time = "2018/03/28 00:00:00"
pic_end_time = "2018/04/30 00:00:00"
# plot each column
plt.figure()
for i,feature in enumerate(features):
    plt.subplot(len(features), 1, i+1)
    #f,ax = plt.subplots(figsize = (12,9))
    plt.plot(pic.loc[pic_start_time:pic_end_time, feature])
    plt.title(feature, loc='right')
plt.show()


# In[4]:


features = ['PM10',  'temperature', 'pressure',       'humidity', 'wind_direction', 'wind_speed']
plt.figure()
for i,feature in enumerate(features):
    plt.subplot(len(features), 1, i+1)
    #f,ax = plt.subplots(figsize = (12,9))
    plt.plot(pic.loc[pic_start_time:pic_end_time, feature])
    plt.title(feature, loc='right')
plt.show()


# ### O3跟温度很有关系哦

# In[5]:


features = ['O3',  'temperature', 'pressure',       'humidity', 'wind_direction', 'wind_speed']
plt.figure()
for i,feature in enumerate(features):
    plt.subplot(len(features), 1, i+1)
    #f,ax = plt.subplots(figsize = (12,9))
    plt.plot(pic.loc[pic_start_time:pic_end_time, feature])
    plt.title(feature, loc='right')
plt.show()


# In[6]:


for station in stations:
    full_station=full_stations[station]
    print(station,full_station.isnull().sum())


# In[7]:
print('-------define feature engineering function----------------')


def score(y_true, y_predict):
    dividend= np.abs(np.array(y_true) - np.array(y_predict))
    denominator = np.array(y_true) + np.array(y_predict)
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator!=0, casting='unsafe'))

n_folds = 5
scoring = make_scorer(score, greater_is_better=False)

def smape_cv(model, train_x, train_y):
    kf = KFold(n_folds, shuffle=True, random_state=True).get_n_splits(train_x)
    smape= -cross_val_score(model, train_x, train_y, scoring=scoring, cv = kf)
    print("score: {:.4f} ({:.4f})".format(smape.mean(), smape.std()))
    return(smape)


# In[8]:


def feature_vec(df, feature, N):
    rows = df.shape[0]
    column_n = [None] * N + [df[feature][i - N] for i in range(N, rows)]
    column_name = "{}_{}".format(feature, N)
    df[column_name] = column_n


# In[9]:



station='liulihe_aq'
test_start_time = "2018-05-01 00:00:00"
train_start_time = "2017/01/11 00:00:00"
train_end_time = "2018/04/30 00:00:00"

print('train_35*3model')

features={}
models={}
scores=[]
test_xs={}
scalers={}
for station in stations:
    airq=full_stations[station].drop(columns=['loc_cluster','stationName','longitude', 'latitude', 'type'], axis=1)#.loc[train_start_time : train_end_time]
    for N in range(1,96):
        feature_vec(airq,'temperature', N)
        feature_vec(airq,'pressure', N)
        feature_vec(airq,'humidity', N)
        feature_vec(airq,'wind_speed', N)
        feature_vec(airq,'wind_direction', N)
    for N in range(48, 168):
        feature_vec(airq, 'PM25', N)
        feature_vec(airq, 'PM10', N)
        feature_vec(airq, 'O3', N)
#    airq= airq.dropna(thresh=100)
    airq.fillna(airq.median(),inplace=True)
    print(station)
    train_set=airq.loc[train_start_time : train_end_time]
    train_x=train_set.iloc[:,6:]
    test_set=airq.loc[test_start_time :]
    test_x=test_set.iloc[:,6:]
    scaler = StandardScaler()
    scaler.fit(train_x)
    scalers[station]=scaler
    train_x= scaler.transform(train_x)
    test_x= scaler.transform(test_x)
    test_xs[station]=test_x
    for feature in ['PM25','PM10','O3']:
        train_y=train_set.loc[:,feature]
        model_lgb = LGBMRegressor(n_iterations=300,learning_rate=0.08,drop_rate=0.05,max_depth=8,lambda_l1=0.1)
        model_lgb.fit(train_x,train_y)
        joblib.dump(model_lgb, 'lgb_weather_wind_%s_%s.pkl' % (station, feature))
        score=smape_cv(model_lgb,train_x,train_y)
        scores.append(score)
        models[(station,feature)]=model_lgb

        '''
        model_lgb = LGBMRegressor(objective='mse',n_iterations=300,learning_rate=0.08,lambda_l1=0.1,drop_rate=0.05,lambda_l2=0.08,max_depth=8)
        model_lgb.fit(x, y)
        joblib.dump(model_lgb, '%s_%s.pkl' % (station, feature))
   
   '''
print(np.mean(scores))
print(np.std(scores))


# In[21]:


train_set.column_




# #### mentougou_aq
# score: 0.6640 (0.0781)
# score: 0.5390 (0.0592)
# score: 0.5439 (0.1190)
# wanshouxigong_aq
# score: 0.6388 (0.0434)
# score: 0.5255 (0.0269)
# score: 0.6107 (0.1039)
# qianmen_aq
# score: 0.6132 (0.0448)
# score: 0.5317 (0.0222)
# score: 0.6266 (0.1013)
# yongdingmennei_aq
# score: 0.6184 (0.0345)
# score: 0.5224 (0.0335)
# score: 0.6612 (0.1194)
# liulihe_aq
# score: 0.5777 (0.0686)
# score: 0.4309 (0.0349)
# score: 0.5237 (0.1154)
# miyunshuiku_aq
# score: 0.7261 (0.0350)
# score: 0.5564 (0.0725)
# score: 0.3161 (0.0354)
# daxing_aq
# score: 0.6323 (0.0427)
# score: 0.4908 (0.0200)
# score: 0.6332 (0.1262)
# yizhuang_aq
# score: 0.6526 (0.0840)
# score: 0.5133 (0.0292)
# score: 0.6560 (0.1869)
# donggaocun_aq
# score: 0.6478 (0.0644)
# score: 0.5465 (0.0549)
# score: 0.4494 (0.0619)
# dongsihuan_aq
# score: 0.6457 (0.0678)
# score: 0.5190 (0.0427)
# score: 0.6553 (0.1226)
# pinggu_aq

# In[17]:

print('predict and output')
s = pd.read_csv('sample_submission.csv')
for station in stations:
    test_set=test_xs[station]
    for index,feature in enumerate(['PM25', 'PM10', 'O3']):
        model = models[(station,feature)]
        y= np.expand_dims(model.predict(test_set),axis=0)
        for i in range(48):
            s.loc[s[s['test_id'] == '%s#%d' % (station, i)].index[0], feature] = y[0][i] if y[0][i] > 0 else abs(y[0][i])


s.to_csv('preaaual.csv',index=False)




