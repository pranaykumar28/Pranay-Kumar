# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 13:30:36 2018
@author: User
"""
import xgboost as xgb
import numpy as np
import datetime as dt
import pandas as pd
import isodate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("C:/Users/Aditya/Desktop/pranay/hiringtask-master/machine_learning/ad_org/data/mn/ad_org_train.csv")

# Cleaning the data
def clean(train):
    for i in range(len(train)):
        train.loc[i, 'duration'] = isodate.parse_duration(train.loc[i, 'duration']).total_seconds()
    
    train['published'] =  pd.to_datetime(train['published'])
    
    train['published'] = train['published'].dt.year+train['published'].dt.month/12+train['published'].dt.day/360
    
    
    train['category'] = train['category'].astype('category')
    train['published'] = train['published'].astype('float')
    train['category'] = pd.Categorical.from_array(train.category).codes
    
    train[['views','likes','dislikes','comment']] = train[['views','likes','dislikes','comment']].astype(str)
    train.loc[train['views'] == 'F', 'views'] = None
    train.loc[train['likes'] == 'F', 'likes'] = None
    train.loc[train['dislikes'] == 'F', 'dislikes'] = None
    train.loc[train['comment'] == 'F', 'comment'] = None

clean(train)
# MAKING DMAT
target = train['adview']
train1 = train.drop(['vidid','adview'],axis=1)

target1 = np.log(target+1)
train1 = np.log(train1.astype(float)+1)

xgtrain = xgb.DMatrix(train1.values,target1.values)


model = xgb.XGBRegressor(objective = "reg:linear", colsample_bytree =  0.9, learning_rate = 0.1,
                max_depth = 6, alpha = 5, subsample = 0.7)
X_train, X_test, y_train, y_test = train_test_split(train1, target1, test_size=0.3, random_state=420)

# fit model on training data
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=500, eval_metric="rmse", eval_set=eval_set, verbose=True)


#IMPORTING AND CLEANING TEST DATA# 

test = pd.read_csv("C:/Users/Aditya/Desktop/pranay/hiringtask-master/machine_learning/ad_org/data/mn/ad_org_test.csv")
clean(test)
test1 = test.drop(['vidid'],axis=1)
test1 = np.log(test1.astype(float)+1)
# making predictions

y_pred = model.predict(test1)
y_pred = np.exp(y_pred)

submission = pd.DataFrame({'vid_id': test['vidid'], 'ad_view': y_pred})
submission = submission[['vid_id', 'ad_view']]
submission.to_csv('submission_ad_org.csv', index=False)