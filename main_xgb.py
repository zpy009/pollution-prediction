
# coding: utf-8

# # 读数据

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import pickle as pkl
from sklearn.externals import joblib
full_air= pkl.load(open("fullair1_withNA111.pkl", "rb"))

print('-----------load preprocessed data-----------------------')

print('-----------divide into stations ----preoprocess for model xgb   again---------')
def merge_loc(station):#station='fangshan_aq'
    station_data =full_air[full_air['stationId'] == station].drop(['stationId','time', 'year', 'month','date','week','weekday','weather'], axis=1)
    print(station, len(station_data))
    station_data.head()
    #station_data['utc_time']= [datetime.strptime(x,'%Y-%m-%d %H:%M:%S') for x in station_data['utc_time']]
    station_data.set_index(['utc_time'], inplace=True)
    station_data=station_data.resample('H').ffill(limit=24)
    #station_data=station_data.reset_index()
    #station_data.head()
    #station_data.isnull().sum()
   # station_data.to_csv('process_'+ station + '.csv', index=True)
    return station_data


# In[35]:


stations = set(full_air['stationId'])
full_stations={}
for station in stations:
    full_station=merge_loc(station)
    original_names = full_station.columns.values.tolist()
    full_stations[station] =  full_station
    print(full_station.shape)



for station in stations:
    full_station=full_stations[station]
    print(station,full_station.isnull().sum())


# In[5]:


p = pd.Panel(full_stations)


# In[7]:


p.to_pickle("3dDATA_WITHNA.pkl")


# In[7]:


stations


print('-------define feature engineering function----------------')

def feature_vec(df, feature, N):
    rows = df.shape[0]
    column_n = [None] * N + [df[feature][i - N] for i in range(N, rows)]
    column_name = "{}_{}".format(feature, N)
    df[column_name] = column_n


# In[73]:


print('----------define score function and cross validation function')

def score(y_true, y_predict):
    dividend= np.abs(np.array(y_true) - np.array(y_predict))
    denominator = np.array(y_true) + np.array(y_predict)
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator!=0, casting='unsafe'))

n_folds = 5
scoring = make_scorer(score, greater_is_better=False)
def smape_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=True).get_n_splits(train_x)
    smape= -cross_val_score(model, train_x, train_y, scoring=scoring, cv = kf)
    print("\n score: {:.4f} ({:.4f})\n".format(smape.mean(), smape.std()))
    return(smape)


station='liulihe_aq'
dev_start_time = "2018-05-01 00:00:00"
train_start_time = "2017/01/10 00:00:00"
train_end_time = "2018/04/24 00:00:00"
test_start_time = "2018/05/01 00:00:00"


def score2(estimator, X, y_true):
    y_predict = estimator.predict(X)
    dividend= np.abs(np.array(y_true) - np.array(y_predict))
    denominator = np.array(y_true) + np.array(y_predict)
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator!=0, casting='unsafe'))


# # xgb单条的

# In[108]:

print('----------train model--------and save-------------------')
models={}
scores=[]
test_xs={}
for station in stations:
    airq=full_stations[station].drop(columns=['loc_cluster','stationName','longitude', 'latitude', 'type'], axis=1)#.loc[train_start_time : train_end_time]
    airq.fillna(airq.median(),inplace=True)
    print(station)
    for feature in ['PM25','PM10','O3']:
        for N in range(48, 168):
            feature_vec(airq, feature, N)

        train_set=airq.loc[train_start_time : train_end_time]
        train_x=train_set.iloc[:,9:]
        train_y=train_set.loc[:,feature]

        test_set=airq.loc[test_start_time :]
        test_x=test_set.iloc[:,9:]
        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x= scaler.transform(train_x)
        test_x= scaler.transform(test_x)
        test_xs[(station,feature)]=test_x
        grid_param={'reg_alpha':[0.08,0.012,0.2]}
        grid = GridSearchCV(estimator = XGBRegressor(n_iterations=300,learning_rate=0.08,drop_rate=0.05,max_depth=8,lambda_l1=0.1), param_grid=grid_param, scoring=score2, cv=5)

        grid.fit(train_x, train_y)
        print(feature,grid.best_score_)
        scores.append(grid.best_score_)
        models[(station,feature)]=grid
        joblib.dump(grid, 'lgb%s_%s.pkl' % (station, feature))
        '''
        model_lgb = LGBMRegressor(objective='mse',n_iterations=300,learning_rate=0.08,lambda_l1=0.1,drop_rate=0.05,lambda_l2=0.08,max_depth=8)
        model_lgb.fit(x, y)
        joblib.dump(model_lgb, '%s_%s.pkl' % (station, p))
   
   '''
print(np.mean(scores))
print(np.std(scores))


# In[119]:

print('predict and output')

# feature importance
model=models[(station,feature)]
#print(model.best_estimator_.feature_importances_)
# plot
plt.bar(range(len(model.best_estimator_.feature_importances_)), model.best_estimator_.feature_importances_)
plt.title((station, feature))
plt.show()


# In[110]:


s = pd.read_csv('sample_submission.csv')
for station in stations:
    
    for index,feature in enumerate(['PM25', 'PM10', 'O3']):
        test_set=test_xs[(station,feature)]
        model = models[(station,feature)]
        y= np.expand_dims(model.predict(test_set),axis=0)
        for i in range(48):
            s.loc[s[s['test_id'] == '%s#%d' % (station, i)].index[0], feature] = y[0][i] if y[0][i] > 0 else abs(y[0][i])


# In[112]:


s.to_csv('111.csv',index=False)

