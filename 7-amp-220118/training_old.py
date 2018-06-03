#!/usr/bin/env python3
import xgboost as xgb
import pandas as pd
import numpy as np
import schedule
import time
import random

from sklearn import datasets, linear_model
from sklearn.linear_model import Lasso

from sklearn.externals import joblib

def mm(x,a=2,b = 0.5):
    y = a*pow(x,b)  
    return y

def job ():    
    df_read = pd.read_csv('export.csv')
    column_itunes = ['artist_id', 'trend', 'track_id', 'paid_youtube_impressions',
           'paid_youtube_views', 'paid_facebook_clicks',
           'paid_facebook_impressions', 'paid_twitter_impressions',
           'paid_facebook_video', 'paid_facebook_instagram_video_impressions',
           'paid_facebook_instagram_post_impressions', 'paid_twitter_clicks',
           'paid_facebook_instagram_video_clicks',
           'paid_facebook_instagram_post_clicks', 'itunes_downloads',
           'capitalfm_com_impressions','capitalfm_com_clicks', 
           'loopme_impressions', 'loopme_clicks',
           'jet_capital_impacts', 'jet_kiss_impacts', 'jet_heart_impacts']
    column_spotify = ['artist_id', 'trend', 'track_id', 'paid_youtube_impressions',
           'paid_youtube_views', 'paid_facebook_clicks',
           'paid_facebook_impressions', 'paid_twitter_impressions',
           'paid_facebook_video', 'paid_facebook_instagram_video_impressions',
           'paid_facebook_instagram_post_impressions', 'paid_twitter_clicks',
           'paid_facebook_instagram_video_clicks',
           'paid_facebook_instagram_post_clicks', 'spotify_streams',
           'capitalfm_com_impressions','capitalfm_com_clicks', 
           'loopme_impressions', 'loopme_clicks',
           'jet_capital_impacts', 'jet_kiss_impacts', 'jet_heart_impacts']
    col = ['paid_youtube_impressions',
           'paid_youtube_views', 'paid_facebook_clicks',
           'paid_facebook_impressions', 'paid_twitter_impressions',
           'paid_facebook_video', 'paid_facebook_instagram_video_impressions',
           'paid_facebook_instagram_post_impressions', 'paid_twitter_clicks',
           'paid_facebook_instagram_video_clicks',
           'paid_facebook_instagram_post_clicks',
           'capitalfm_com_impressions','capitalfm_com_clicks', 
           'loopme_impressions', 'loopme_clicks',
           'jet_capital_impacts', 'jet_kiss_impacts', 'jet_heart_impacts']
    df = df_read[column_itunes]

    df[col] = df[col].apply(lambda x:mm(x))
    df['ma']= df.groupby('track_id')['itunes_downloads'].apply(pd.rolling_mean,1, min_periods=1)
    df['ma'] =df.groupby('track_id')['ma'].shift(1)
    df['ma1']= df.groupby('track_id')['itunes_downloads'].apply(pd.rolling_mean,2, min_periods=1)
    df['ma1'] =df.groupby('track_id')['ma1'].shift(1)
    df['std'] = df.groupby('track_id')['itunes_downloads'].apply(pd.rolling_std,2, min_periods=1)
    df['std'] = df.groupby('track_id')['std'].shift(1)
    csv = df[['artist_id','track_id','trend','ma','ma1','std']]
    csv = csv.fillna(0)
    d = pd.DataFrame(csv)
    filename = 'CSV/artist_model_itunes.csv'
    d.to_csv(filename, index=False, encoding='utf-8')
    d1 = df[df['itunes_downloads'] > 0]
    d1 = d1[d1['itunes_downloads'] <20000]
    d1 = d1.reset_index(drop=True)
    d1 = d1[pd.notnull(d1['itunes_downloads'])]
    df1=d1.fillna(0)
    df1.drop(['track_id'],axis=1,inplace = True)
    df1['artist_id'] = df1['artist_id'].astype('category')
    df2 = pd.get_dummies(df1, dummy_na=True)
    df3=df2
    df3=pd.DataFrame.sort_index(df3,axis=1)
    Y_train = df3['itunes_downloads']
    X_train = df3.drop('itunes_downloads',axis=1)
    T_train_xgb = xgb.DMatrix(X_train, Y_train)
    keys =[X_train.columns.get_loc(c) for c in X_train.columns if c in col]
    feature_monotones = [0] * (len(X_train.columns) - len (keys))
    for key in keys :
        feature_monotones.insert(key-1,1)
    monotone_constraints = '(' + ','.join([str(m) for m in feature_monotones]) + ')'
    
    
    params = {"objective": "reg:linear","eta": 0.15,"max_depth" : 3, 
           "subsample" : 0.9,"colsample_bytree" : 0.9,
              "eval_metric" : "rmse","nthread" : 10,"silent":1,'gamma':0, 'lambda':0.01,
              'min_child_weight':1,'monotone_constraints':monotone_constraints}

    #bst_cv = xgb.cv(params, T_train_xgb, 500, nfold = 5, early_stopping_rounds=10,seed = 101)
    gbm = xgb.train(params, T_train_xgb, num_boost_round = 50) 

    joblib.dump(gbm, 'models/artist-reg-itunes.pkl')
    model_columns = X_train.columns
    joblib.dump(model_columns, 'models/artist-model-itunes-col.pkl')


    ##spotify

    df_spotify = df_read[column_spotify]

    df_spotify[col] = df_spotify[col].apply(lambda x:mm(x))
    df_spotify['ma']= df_spotify.groupby('track_id')['spotify_streams'].apply(pd.rolling_mean,1, min_periods=1)
    df_spotify['ma'] =df_spotify.groupby('track_id')['ma'].shift(1)
    df_spotify['ma1']= df_spotify.groupby('track_id')['spotify_streams'].apply(pd.rolling_mean,2, min_periods=1)
    df_spotify['ma1'] =df_spotify.groupby('track_id')['ma1'].shift(1)
    df_spotify['std'] = df_spotify.groupby('track_id')['spotify_streams'].apply(pd.rolling_std,2, min_periods=1)
    df_spotify['std'] = df_spotify.groupby('track_id')['std'].shift(1)
    csv_spotify = df_spotify[['artist_id','track_id','trend','ma','ma1','std']]
    csv_spotify = csv_spotify.fillna(0)
    d_spotify = pd.DataFrame(csv_spotify)
    filename = 'CSV/artist_model_spotify.csv'
    d_spotify.to_csv(filename, index=False, encoding='utf-8')
    #df_spotify = df_spotify[df_spotify['spotify_streams'] > 10]
    #df_spotify = df_spotify[df_spotify['spotify_streams'] < 4524003]
    d1_spotify = df_spotify[df_spotify['spotify_streams'] > 0]
    d1_spotify = d1_spotify[d1_spotify['spotify_streams'] < 4524003]
    d1_spotify = d1_spotify.reset_index(drop=True)
    d1_spotify = d1_spotify[pd.notnull(d1_spotify['spotify_streams'])]
    df1_spotify=d1_spotify.fillna(0)
    df1_spotify.drop(['track_id'],axis=1,inplace = True)
    df1_spotify['artist_id'] = df1_spotify['artist_id'].astype('category')
    df2_spotify = pd.get_dummies(df1_spotify, dummy_na=True)
    df3_spotify=df2_spotify
    df3_spotify=pd.DataFrame.sort_index(df3_spotify,axis=1)
    Y_train_spotify = df3_spotify['spotify_streams']
    X_train_spotify = df3_spotify.drop('spotify_streams',axis=1)

    T_train_xgb = xgb.DMatrix(X_train_spotify, Y_train_spotify)
    
    keys =[X_train_spotify.columns.get_loc(c) for c in X_train_spotify.columns if c in col]
    feature_monotones = [0] * (len(X_train_spotify.columns) - len (keys))
    for key in keys :
        feature_monotones.insert(key-1,1)
    monotone_constraints = '(' + ','.join([str(m) for m in feature_monotones]) + ')'
   
    
    params = {"objective": "reg:linear","eta": 0.15,"max_depth" : 3, 
           "subsample" : 0.9,"colsample_bytree" : 0.9,
              "eval_metric" : "rmse","nthread" : 10,"silent":1,'gamma':0, 'lambda':0.01,
              'min_child_weight':1,'monotone_constraints':monotone_constraints}

    #bst_cv = xgb.cv(params, T_train_xgb, 500, nfold = 5, early_stopping_rounds=10,seed = 101)
    gbm = xgb.train(params, T_train_xgb, num_boost_round = 50)

    joblib.dump(gbm, 'models/artist-reg-spotify.pkl')

    model_columns = X_train_spotify.columns
    joblib.dump(model_columns, 'models/artist-model-spotify-col.pkl')

    return "success"

schedule.every(1).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
