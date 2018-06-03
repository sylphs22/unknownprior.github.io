#!/usr/bin/env python3
import xgboost as xgb
from flask import Flask, jsonify,request
import pandas as pd
import numpy 
import datetime
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)

@app.route('/similar', methods=['POST'])
def predict():
    json_ = request.get_json(force=True)

    query_df = pd.DataFrame.from_dict([json_], orient='columns')
    a = []
    for i in range(1,len(list(query_df))):
        query = query_df.loc[0][i]
        a.append(query)
    df = pd.read_csv('export.csv')
    #import pdb; pdb.set_trace()
    b = query_df ['Target output'].iloc[0]
    sales_match = sum(a)
    number_of_weeks = len(a)
    df = df.fillna(0)
    df_compare = df[df.trend <= number_of_weeks]


    #df_compare = df[df['trend'].isin([1,2,3])]
    #grouped = df_compare.groupby('track_id')
    #df_compare = grouped.filter(lambda x: len(x) > 1)

    df_compare.itunes_downloads = df_compare.itunes_downloads.fillna(0)
    df_compare.spotify_streams = df_compare.spotify_streams.fillna(0)
    df_compare.apple_music = df_compare.apple_music.fillna(0)

    # compute sums for the distance calculation
    df_compare['itunes_sum'] = df_compare.groupby('track_id')['itunes_downloads'].cumsum()
    df_compare['spotify_sum'] = df_compare.groupby('track_id')['spotify_streams'].cumsum()
    df_compare['apple_sum'] = df_compare.groupby('track_id')['apple_music'].cumsum()

    # calculate distance based on type of sales
    if (b == 'Itunes Downloads'):
        df_compare['distance'] = abs(df_compare.itunes_sum - sales_match)
    elif (b == 'Apple Music Streams'):
        df_compare['distance'] = abs(df_compare.apple_sum - sales_match)
    else:
        df_compare['distance'] = abs(df_compare.spotify_sum - sales_match)
    df_compare['cosine_test'] = 0
    new = [df_compare.cosine_test[0]]
    for i in range(1, len(df_compare.index)):
        if df_compare.trend.iloc[i] == number_of_weeks and b =='Itunes Downloads':
            new.append(cosine_similarity(a, df_compare.itunes_downloads[(i-number_of_weeks+1):i+1]))
        elif df_compare['trend'].iloc[i] == number_of_weeks and b == 'Apple Music Streams':
            new.append(cosine_similarity(a, df_compare.apple_music[(i-number_of_weeks+1):i+1]))
        elif df_compare['trend'].iloc[i] == number_of_weeks and b == 'Spotify Streams':
            new.append(cosine_similarity(a, df_compare.spotify_streams[(i-number_of_weeks+1):i+1]))
        else:
            new.append(numpy.nan)
            
    df_compare['cosine_test'] = new
    df_compare = df_compare[df_compare.trend == number_of_weeks]
    similar_artists = df_compare.sort_values(by=['cosine_test'], ascending=False)
    similar_artists['distance_rank'] = similar_artists['distance'].rank(ascending=1)
    for index in list(similar_artists['distance_rank'].index):
        if similar_artists.distance_rank[index] < len(similar_artists) * .075:
            track_id=(similar_artists.track_id[index])
            artist_id=(similar_artists.artist_id[index])
            break
 
    df = pd.DataFrame(columns=['track_id','artist_id'])
    df.loc[0] = [track_id,artist_id]
    df1=df.append([df]*len(a), ignore_index=True)
    a.append(0)
    if (b == 'Itunes Downloads'):
       df2 = pd.DataFrame({'itunes_downloads':a})
       df1 = pd.merge(df1, df2, left_index=True, right_index=True)
       df1['ma']= df1.groupby('track_id')['itunes_downloads'].apply(pd.rolling_mean,1, min_periods=1)
       df1['ma'] =df1.groupby('track_id')['ma'].shift(1)
       df1['ma1']= df1.groupby('track_id')['itunes_downloads'].apply(pd.rolling_mean,2, min_periods=1)
       df1['ma1'] =df1.groupby('track_id')['ma1'].shift(1)
       df1['std'] = df1.groupby('track_id')['itunes_downloads'].apply(pd.rolling_std,2, min_periods=1)
       df1['std'] = df1.groupby('track_id')['std'].shift(1)
       df1['trend'] = df1.index+1
       csv = df1[['artist_id','track_id','trend','ma','ma1','std']]
       csv = csv.fillna(0)
       d = pd.DataFrame(csv)
       filename = 'CSV/artist_model_itunes_simdb.csv'
       d.to_csv(filename, index=False, encoding='utf-8')

    elif (b == 'Spotify Streams'):
       df2 = pd.DataFrame({'spotify_streams':a})
       df1 = pd.merge(df1, df2, left_index=True, right_index=True)
       df1['ma']= df1.groupby('track_id')['spotify_streams'].apply(pd.rolling_mean,1, min_periods=1)
       df1['ma'] =df1.groupby('track_id')['ma'].shift(1)
       df1['ma1']= df1.groupby('track_id')['spotify_streams'].apply(pd.rolling_mean,2, min_periods=1)
       df1['ma1'] =df1.groupby('track_id')['ma1'].shift(1)
       df1['std'] = df1.groupby('track_id')['spotify_streams'].apply(pd.rolling_std,2, min_periods=1)
       df1['std'] = df1.groupby('track_id')['std'].shift(1)
       df1['trend'] = df1.index+1
       csv = df1[['artist_id','track_id','trend','ma','ma1','std']]
       csv = csv.fillna(0)
       d = pd.DataFrame(csv)
       filename = 'CSV/artist_model_spotify_simdb.csv'
       d.to_csv(filename, index=False, encoding='utf-8')
       
    return df.to_json(orient='records')
  

if __name__ == '__main__':
     app.run(port=8081)
