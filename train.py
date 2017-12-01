import random
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV

def read_sample_from(fname, n=1000):
    with open(fname,'r') as f:
        f_len = sum(1 for line in f) - 1
        
    skip = sorted(random.sample(range(1,f_len),f_len-n))
    
    return pd.read_csv(
        fname, 
        low_memory=False, usecols=range(27),
        skiprows=skip
    )

# df = read_sample_from('./data/311_Service_Requests_from_2011.csv', 10000)
df = pd.read_csv('./data/311_Service_Requests_from_2011.csv', usecols=range(1,27), low_memory=False )
print("File read complete: ", len(df), 'rows')
df = df[['Created Date', 'Closed Date', 'Agency', 'Agency Name',
       'Complaint Type', 'Descriptor', 'Location Type', 'Incident Zip',
       'Incident Address', 'Street Name', 'Cross Street 1', 'Cross Street 2',
       'Intersection Street 1', 'Intersection Street 2', 'Address Type',
       'City', 'Landmark', 'Facility Type', 'Status', 'Due Date',
       'Resolution Description', 'Resolution Action Updated Date',
       'Community Board', 'Borough', 'X Coordinate (State Plane)', 'Y Coordinate (State Plane)']]
df.columns = ['Created Date', 'Closed Date', 'Agency', 'Agency Name',
       'Complaint Type', 'Descriptor', 'Location Type', 'Incident Zip',
       'Incident Address', 'Street Name', 'Cross Street 1', 'Cross Street 2',
       'Intersection Street 1', 'Intersection Street 2', 'Address Type',
       'City', 'Landmark', 'Facility Type', 'Status', 'Due Date',
       'Resolution Description', 'Resolution Action Updated Date',
       'Community Board', 'Borough', 'X', 'Y']

A = df[df.Status == 'Closed'].copy()
del df
A = A[A['Closed Date'].notnull()]
A.fillna(0, inplace=True)
A['Closed Date'] = pd.to_datetime(A['Closed Date'], format='%m/%d/%Y %H:%M:%S %p')
A['Created Date'] = pd.to_datetime(A['Created Date'], format='%m/%d/%Y %H:%M:%S %p')
A['Created Month'] = A['Created Date'].apply( lambda x: x.month)

A['Response Time'] = (A['Closed Date'] - A['Created Date']) / np.timedelta64(1, 'h')

A = A[['Agency', 'Complaint Type', 'Location Type', 
       'City', 'Borough', 'Facility Type', 'Created Month', 'X', 'Y', 'Response Time']]

for col in A.columns:
    if A[col].dtype not in ['float64', 'int', 'int64']:
        dummies = pd.get_dummies(A[col])
        A[dummies.columns] = dummies

A = A[A['Response Time'] != 0]
A = A[A.columns[6:]]

train, test = train_test_split(A,test_size=0.2)
print("Number of training observations: ", len(train))
print("Number of features: ", len(train.columns))

training_x = train[[col for col in train.columns if col != 'Response Time']].values
training_y = train['Response Time'].values
testing_x = test[[col for col in test.columns if col != 'Response Time']].values
testing_y = test['Response Time'].values

cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 100, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'reg:linear'}

print("training columns")
print(train.columns)    
# gbm = GridSearchCV(xgb.XGBRegressor(**ind_params), cv_params, scoring = 'r2', cv=5)
gbm = xgb.XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=3, min_child_weight=3)
gbm = gbm.fit(training_x, training_y)
# print("Grid Scores")
# print(gbm.grid_scores_)

print("Training Complete")
print("score: ", gbm.score(testing_x, testing_y))

with open('model_02.pkl', 'wb') as f:
    pickle.dump(gbm, f)

import json
cols = [col for col in train.columns if col != 'Response Time']
with open('cols.json', 'w') as fi:
    json.dump(cols, fi)

