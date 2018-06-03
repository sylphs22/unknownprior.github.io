import pickle
import pandas as pd
import random
import numpy as np, numpy.random
import xgboost as xgb


from flask import Flask, jsonify,request
from sklearn.externals import joblib
from scipy.optimize import minimize
app = Flask(__name__)

@app.route('/const', methods=['POST'])
def main():
    random.seed(9001)
    #clf = joblib.load('model/artist-reg.pkl')
    #model_columns = joblib.load('model/artist-model-col.pkl')
    json_ = request.get_json(force=True)
    #import pdb; pdb.set_trace()
    query_df = pd.DataFrame.from_dict([json_], orient='columns')
    app.track_id = query_df['track_id'].iloc[0]
    app.artist_id = query_df['artist_id'].iloc[0]
    release = query_df['Release Date']
    start = query_df['Campaign Launch']
    end = query_df['Campaign End']
    
    #
    x = pd.to_datetime(start,format='%d/%m/%Y') - pd.to_datetime(release,format='%d/%m/%Y')
    app.trend = int(x / np.timedelta64(1, 'W'))+1
    #y = pd.to_datetime(end,format='%d/%m/%Y') - pd.to_datetime(start,format='%d/%m/%Y')
    #app.weeks = int(y / np.timedelta64(1, 'W'))+1
    app.weeks =1

    app.budget = query_df['budget'].iloc[0]
    if app.budget == 'no constraint':
        app.budget = 10000*app.weeks
    else:
        app.budget = app.budget 
               
    app.target_output = query_df['Target Output'].iloc[0]
    
   
    
    if (app.target_output == 'Itunes Downloads'):
       #app.clf = joblib.load('models/artist-reg-itunes.pkl')
       app.clf = joblib.load('models/artist-reg-itunes.pkl')
       app.model_columns = joblib.load('models/artist-model-itunes-col.pkl')
       app.csv=pd.read_csv('CSV/artist_model_itunes.csv')
       
    elif (app.target_output == 'Spotify Streams'):
       app.clf = joblib.load('models/artist-reg-spotify.pkl')
       app.model_columns = joblib.load('models/artist-model-spotify-col.pkl')
       app.csv=pd.read_csv('CSV/artist_model_spotify.csv')
    
    app.facebook_post_cpm = query_df ['facebook_post_cpm'].iloc[0]
    app.facebook_post_ctr = query_df ['facebook_post_ctr'].iloc[0]/100
    app.facebook_video_cpm = query_df ['facebook_video_cpm'].iloc[0]
    app.facebook_video_ctr = query_df ['facebook_video_ctr'].iloc[0]/100 
    app.facebook_video_vtr = query_df ['facebook_video_vtr'].iloc[0]/100
    app.youtube_cpm = query_df ['youtube_cpm'].iloc[0]
    app.youtube_vtr = query_df ['youtube_vtr'].iloc[0]/100
    app.instagram_post_cpm = query_df ['instagram_post_cpm'].iloc[0]
    app.instagram_post_ctr = query_df ['instagram_post_ctr'].iloc[0]/100
    app.instagram_video_cpm = query_df ['instagram_video_cpm'].iloc[0]
    app.instagram_video_ctr = query_df ['instagram_video_cpm'].iloc[0]/100
    app.twitter_cpm = query_df ['twitter_cpm'].iloc[0]
    app.twitter_ctr= query_df ['twitter_ctr'].iloc[0]/100
    app.dg_capital_cpm =query_df ['capitalfm_com_cpm'].iloc[0] 
    app.dg_capital_ctr = query_df ['capitalfm_com_ctr'].iloc[0]/100
    app.dg_loop_cpm = query_df ['loopme_cpm'].iloc[0]
    app.dg_loop_ctr = query_df ['loopme_ctr'].iloc[0]/100
    app.radio_capital_cpm = query_df ['jet_capital_cpm'].iloc[0]
    app.radio_heart_cpm = query_df ['jet_heart_cpm'].iloc[0]
    app.radio_kiss_cpm = query_df ['jet_kiss_cpm'].iloc[0]
    app.facebook_post_min_spend = query_df ['facebook_post_min_spend'].iloc[0]
    app.facebook_video_min_spend = query_df ['facebook_video_min_spend'].iloc[0]
    app.youtube_min_spend = query_df ['youtube_min_spend'].iloc[0]
    app.instagram_post_min_spend = query_df ['instagram_post_min_spend'].iloc[0]
    app.instagram_video_min_spend = query_df ['instagram_video_min_spend'].iloc[0]
    app.twitter_min_spend = query_df ['twitter_min_spend'].iloc[0]
    app.dg_capital_min_spend = query_df ['capitalfm_com_min_spend'].iloc[0]
    app.dg_loop_min_spend = query_df ['loopme_min_spend'].iloc[0]
    app.radio_capital_min_spend =query_df ['jet_capital_min_spend'].iloc[0]
    app.radio_heart_min_spend = query_df ['jet_heart_min_spend'].iloc[0]
    app.radio_kiss_min_spend=query_df ['jet_kiss_min_spend'].iloc[0] 
    app.facebook_post_max_spend = query_df ['facebook_post_max_spend'].iloc[0]
    app.facebook_video_max_spend = query_df ['facebook_video_max_spend'].iloc[0]
    app.youtube_max_spend = query_df ['youtube_max_spend'].iloc[0]
    app.instagram_post_max_spend = query_df ['instagram_post_max_spend'].iloc[0]
    app.instagram_video_max_spend = query_df ['instagram_video_max_spend'].iloc[0]
    app.twitter_max_spend = query_df ['twitter_max_spend'].iloc[0]
    app.dg_capital_max_spend = query_df ['capitalfm_com_max_spend'].iloc[0]
    app.dg_loop_max_spend = query_df ['loopme_max_spend'].iloc[0]
    app.radio_capital_max_spend =query_df ['jet_capital_max_spend'].iloc[0]
    app.radio_heart_max_spend = query_df ['jet_heart_max_spend'].iloc[0]
    app.radio_kiss_max_spend = query_df ['jet_kiss_max_spend'].iloc[0]
    #import pdb; pdb.set_trace()
    output = sim_time(app.budget,app.weeks)
    labels = ['capitalfm_com_clicks','capitalfm_com_impressions',
              'loopme_clicks','loopme_impressions','jet_capital_impacts',
              'jet_heart_impacts','jet_kiss_impacts',
              'paid_facebook_post_impressions','paid_facebook_clicks',
              'paid_facebook_video','paid_youtube_impressions','paid_youtube_views',
              'paid_facebook_instagram_video_impressions',
              'paid_facebook_instagram_video_clicks',
              'paid_facebook_instagram_post_impressions',
              'paid_facebook_instagram_post_clicks','paid_twitter_impressions',
              'paid_twitter_clicks','paid_facebook_video_impressions']
    #itunes =['itunes_downloads']
    output_list = output[0]
    output_itunes = output[1]*(-1)
    output_df = pd.DataFrame.from_records(output_list,columns=labels)
    itunes_df = pd.DataFrame({'Itunes Downloads': [output_itunes]})
    output_df_new = pd.concat([output_df, itunes_df], axis=1).astype(int)
    #return jsonify({'output': output[0]})
    return output_df_new.to_json(orient='records')

def sim(budget) :
    res_input = []
    res_out = []
    np.random.seed(0)
    sample = np.random.dirichlet(np.ones(11),size=25) 
    #import pdb; pdb.set_trace()
    
    #facebook_post_max_spend =   
    for i in range(len(sample)):
        #query_df = main()  
        #facebook_post_max_spend = query_df ['facebook_post_max_spend'].iloc[0]               
        b= np.array(sample[i])
        a = np.array([app.facebook_post_max_spend,app.facebook_video_max_spend,app.youtube_max_spend,app.instagram_post_max_spend,\
                     #app.instagram_video_max_spend,
                     0,
                     app.twitter_max_spend,app.dg_capital_max_spend,app.dg_loop_max_spend,\

                      app.radio_capital_max_spend,app.radio_heart_max_spend,app.radio_kiss_max_spend])
        c = np.multiply(a,b)
        #import pdb; pdb.set_trace()
        #c = a
        facebook_post_max_spend = c[0]
        facebook_video_max_spend = c[1]
        youtube_max_spend = c[2]
        instagram_post_max_spend = c[3]
        instagram_video_max_spend = c[4]
        twitter_max_spend = c[5]
        dg_capital_max_spend = c[6]
        dg_loop_max_spend = c[7]
        radio_capital_max_spend =c[8]
        radio_heart_max_spend = c[9]
        radio_kiss_max_spend = c[10]
        facebook_post_min_clicks = (app.facebook_post_min_spend*1000/app.facebook_post_cpm)*app.facebook_post_ctr
        facebook_video_min_clicks = (app.facebook_video_min_spend*1000/app.facebook_video_cpm)*app.facebook_video_ctr
        facebook_video_min_views = (app.facebook_video_min_spend*1000/app.facebook_post_cpm)*app.facebook_video_vtr
        youtube_min_views = (app.youtube_min_spend*1000/app.youtube_cpm)*app.youtube_vtr
        instagram_post_min_clicks = (app.instagram_post_min_spend*1000/app.instagram_post_cpm)*app.instagram_post_ctr
        instagram_video_min_clicks = (app.instagram_video_min_spend*1000/app.instagram_video_cpm)*app.instagram_video_ctr
        twitter_min_clicks = (app.twitter_min_spend*1000/app.twitter_cpm)*app.twitter_ctr
        dg_capital_min_clicks = (app.dg_capital_min_spend*1000/app.dg_capital_cpm)*app.dg_capital_ctr
        dg_loop_min_clicks = (app.dg_loop_min_spend*1000/app.dg_loop_cpm)*app.dg_loop_ctr
        facebook_post_max_clicks = (app.facebook_post_max_spend*1000/app.facebook_post_cpm)*app.facebook_post_ctr
        facebook_video_max_clicks = (app.facebook_video_max_spend*1000/app.facebook_video_cpm)*app.facebook_video_ctr
        facebook_video_max_views = (app.facebook_video_max_spend*1000/app.facebook_post_cpm)*app.facebook_video_vtr
        youtube_max_views = (app.youtube_max_spend*1000/app.youtube_cpm)*app.youtube_vtr
        instagram_post_max_clicks = (app.instagram_post_max_spend*1000/app.instagram_post_cpm)*app.instagram_post_ctr
        instagram_video_max_clicks = (app.instagram_video_max_spend*1000/app.instagram_video_cpm)*app.instagram_video_ctr
        twitter_max_clicks = (app.twitter_max_spend*1000/app.twitter_cpm)*app.twitter_ctr
        dg_capital_max_clicks = (app.dg_capital_max_spend*1000/app.dg_capital_cpm)*app.dg_capital_ctr
        dg_loop_max_clicks = (app.dg_loop_max_spend*1000/app.dg_loop_cpm)*app.dg_loop_ctr
        bounds = [(dg_capital_min_clicks,dg_capital_max_clicks),\
          (app.dg_capital_min_spend*1000/app.dg_capital_cpm,dg_capital_max_spend*1000/app.dg_capital_cpm),\
          (dg_loop_min_clicks,dg_loop_max_clicks),\
          (app.dg_loop_min_spend*1000/app.dg_loop_cpm,dg_loop_max_spend*1000/app.dg_loop_cpm),\
          (app.radio_capital_min_spend*1000/app.radio_capital_cpm,radio_capital_max_spend*1000/app.radio_capital_cpm),\
          (app.radio_heart_min_spend*1000/app.radio_heart_cpm,radio_heart_max_spend*1000/app.radio_heart_cpm),\
          (app.radio_kiss_min_spend*1000/app.radio_kiss_cpm,radio_kiss_max_spend*1000/app.radio_kiss_cpm),\
          (app.facebook_post_min_spend*1000/app.facebook_post_cpm,facebook_post_max_spend*1000/app.facebook_post_cpm),\
          (facebook_post_min_clicks,facebook_post_max_clicks),\
          (facebook_video_min_views,facebook_video_max_views),\
          (app.youtube_min_spend*1000/app.youtube_cpm,youtube_max_spend*1000/app.youtube_cpm),\
          (youtube_min_views,youtube_max_views),\
          (app.instagram_video_min_spend*1000/app.instagram_video_cpm,app.instagram_video_max_spend*1000/app.instagram_video_cpm),\
          (instagram_video_min_clicks,instagram_video_max_clicks),\
          (app.instagram_post_min_spend*1000/app.instagram_post_cpm,instagram_post_max_spend*1000/app.instagram_post_cpm),\
          (instagram_post_min_clicks,instagram_post_max_clicks),\
          (app.twitter_min_spend*1000/app.twitter_cpm,twitter_max_spend*1000/app.twitter_cpm),(twitter_min_clicks,twitter_max_clicks),\
         (app.facebook_video_min_spend*1000/app.facebook_video_cpm,facebook_video_max_spend*1000/app.facebook_video_cpm)]
        lb = [dg_capital_min_clicks, app.dg_capital_min_spend*1000/app.dg_capital_cpm,
              dg_loop_min_clicks,app.dg_loop_min_spend*1000/app.dg_loop_cpm,app.radio_capital_min_spend*1000/app.radio_capital_cpm,
             app.radio_heart_min_spend*1000/app.radio_heart_cpm,app.radio_kiss_min_spend*1000/   app.radio_kiss_cpm,app.facebook_post_min_spend*1000/app.facebook_post_cpm,
             facebook_post_min_clicks,facebook_video_min_views,app.youtube_min_spend*1000/app.youtube_cpm,youtube_min_views,
             app.instagram_video_min_spend*1000/app.instagram_video_cpm,instagram_video_min_clicks,app.instagram_post_min_spend*1000/app.instagram_post_cpm,
             instagram_post_min_clicks,app.twitter_min_spend*1000/app.twitter_cpm,twitter_min_clicks,app.facebook_video_min_spend*1000/app.facebook_video_cpm]

        ub = [dg_capital_max_clicks, dg_capital_max_spend*1000/app.dg_capital_cpm,
              dg_loop_max_clicks,dg_loop_max_spend*1000/app.dg_loop_cpm,radio_capital_max_spend*1000/app.radio_capital_cpm,
             radio_heart_max_spend*1000/app.radio_heart_cpm,radio_kiss_max_spend*1000/app.radio_kiss_cpm,facebook_post_max_spend*1000/app.facebook_post_cpm,
             facebook_post_max_clicks,facebook_video_max_views,youtube_max_spend*1000/app.youtube_cpm,youtube_max_views,
             instagram_video_max_spend*1000/app.instagram_video_cpm,instagram_video_max_clicks,instagram_post_max_spend*1000/app.instagram_post_cpm,
             instagram_post_max_clicks,twitter_max_spend*1000/app.twitter_cpm,twitter_max_clicks,facebook_video_max_spend*1000/app.facebook_video_cpm]
        cons = ({'type': 'ineq', 'fun': lambda x : g1(x,budget)},
        {'type': 'ineq', 'fun': lambda x : g2(x)},
        {'type': 'ineq', 'fun': lambda x : g3(x)},
       {'type': 'ineq', 'fun': lambda x :g4(x)},
        {'type': 'ineq', 'fun': lambda x :g5(x)},
       {'type': 'ineq', 'fun': lambda x :g6(x)},
        {'type': 'ineq', 'fun': lambda x :g7(x)},
       {'type': 'ineq', 'fun': lambda x :g8(x)},
       {'type': 'ineq', 'fun': lambda x :g9(x)})
        
        lb = np.array(lb)
        ub = np.array (ub)
        ab = (lb+ub)/2 
        res = minimize(f, ub,  method='slsqp', constraints = cons,bounds = bounds,jac = False)
        
        res_input.append(res.x)
        res_out.append(res.fun)
        #print(res)
        #bo.fx_opt
    ind = np.argmin(res_out)
    max_sales = min (res_out)
    return [res_input[ind],max_sales]
        
def sim_time(budget,week = 4):##look to address trend
    if  week > 1 :
        np.random.seed(9001)
        sample = np.random.dirichlet(np.ones(week),size=week*4)
    else :
        np.random.seed(9001)
        sample = np.random.dirichlet(np.ones(week),size=1)
    res_time_input = []
    res_time_out = []
    total_sales = []
    for frac in sample:
        sale = 0
        res_time = []
        
        actual_budget = 0
        extra = 0
        for fract in frac:             
            budget1 = budget*fract
            if budget1 > 8000:
               budget1 = 8000 + mm(budget1 - 8000)
             
            res = sim(budget1)
            sale = sale + res[1]-0.1*res[1]
            res_time.append(res[0])
            #trend = trend +1
            #print (sale)
        res_time_input.append(res_time)
        total_sales.append(sale)
    index = np.argmin(total_sales)
    max_tot_sales = min(total_sales)
    return [res_time_input[index],max_tot_sales]  
def mm(x,a=2,b = 0.5):
    y = a*pow(x,b)    
    return y
def f(x):
    #x = reshape(x,19)###
    #n = x.shape[0]
    test_q1 = {}
    test_q1['capitalfm_com_clicks']= x[0]
    test_q1['capitalfm_com_impressions'] = x[1]
    test_q1['loopme_clicks'] = x[2]
    test_q1['loopme_impressions'] = x[3]
    test_q1['jet_capital_impacts']=x[4]
    test_q1['jet_heart_impacts'] = x[5]
    test_q1['jet_kiss_impacts'] = x[6]
    test_q1['paid_facebook_post_impressions']= x[7]
    test_q1['paid_facebook_clicks'] = x[8]
    test_q1['paid_facebook_video'] = x[9]
    test_q1['paid_youtube_impressions'] = x[10]
    test_q1['paid_youtube_views']=x[11]
    test_q1['paid_facebook_instagram_video_impressions'] = x[12]
    test_q1['paid_facebook_instagram_video_clicks'] = x[13]
    test_q1['paid_facebook_instagram_post_impressions'] = x[14]
    test_q1['paid_facebook_instagram_post_clicks']=x[15]
    test_q1['paid_twitter_impressions'] = x[16]
    test_q1['paid_twitter_clicks'] = x[17]
    test_q1['paid_facebook_video_impressions'] = x[18]
    test_q1['paid_facebook_impressions'] = x[7]+x[18]
    test_q1['trend'] = app.trend
    test_q1['track_id'] = app.track_id
    test_q1['artist_id'] = app.artist_id
    
    #csv=pd.read_csv('artist_model_itunes.csv')
    
    csv=app.csv
    d1 = csv[csv['track_id'] == app.track_id]
    if app.trend > max (d1['trend']): 
        val = max (d1['trend'])
    else : 
        val =  app.trend
    ma = csv.loc[(csv['track_id'] == app.track_id) & (csv['trend'] == val), 'ma'].iloc[0]
    ma1 = csv.loc[(csv['track_id'] == app.track_id) & (csv['trend'] == val), 'ma1'].iloc[0]
    std = csv.loc[(csv['track_id'] == app.track_id) & (csv['trend'] == val), 'std'].iloc[0]
    test_q1['ma'] = ma
    test_q1['ma1'] = ma1
    test_q1['std'] = std   
    
    query_test_df = pd.DataFrame.from_dict([test_q1], orient='columns')

    query_test_df.drop(['paid_facebook_post_impressions'],axis=1,inplace = True)
    query_test_df.drop(['paid_facebook_video_impressions'],axis=1,inplace = True)
    query_test_df.drop(['track_id'],axis=1,inplace = True)
    clf = app.clf
    model_columns = app.model_columns
    
    col2 = ['paid_youtube_impressions',
       'paid_youtube_views', 'paid_facebook_clicks',
       'paid_facebook_impressions', 'paid_twitter_impressions',
       'paid_facebook_video', 'paid_facebook_instagram_video_impressions',
       'paid_facebook_instagram_post_impressions', 'paid_twitter_clicks',
       'paid_facebook_instagram_video_clicks',
       'paid_facebook_instagram_post_clicks',
       'capitalfm_com_impressions','capitalfm_com_clicks', 
       'loopme_impressions', 'loopme_clicks',
       'jet_capital_impacts', 'jet_kiss_impacts', 'jet_heart_impacts']
    query_test_df[col2] = query_test_df[col2].apply(lambda x:mm(x))

    query_test_df["artist_id"] = query_test_df["artist_id"].astype('category')
    
    query_test_df1 = pd.get_dummies(query_test_df, dummy_na=True)
    for col in model_columns:
        if col not in query_test_df1.columns:
            query_test_df1[col] = 0
    query_test_df1 = pd.DataFrame.sort_index(query_test_df1,axis=1)
    
    itunes = (clf.predict(xgb.DMatrix(query_test_df1)))*(-1)
    #itunes = (clf.predict(query_test_df1))*(-1)#### maximize and not mimimize so negative sign
    return itunes
def g1(x,budget):
    return (x[1]*app.dg_capital_cpm/1000+x[3]*app.dg_loop_cpm/1000 
               + x[4]*app.radio_capital_cpm/1000+x[5]*app.radio_heart_cpm/1000+x[6]*app.radio_kiss_cpm/1000 
             + x[7]*app.facebook_post_cpm/1000 + x[10]*app.youtube_cpm/1000 + x[12]*app.instagram_video_cpm /1000
              +x[14]*app.instagram_post_cpm/1000 + 
            x[16]*app.twitter_cpm/1000+x[18]*app.facebook_video_cpm/1000)*(-1) + budget
def g2(x):
    return x[1]*app.dg_capital_ctr - x[0]
def g3(x):
    return x[3]*app.dg_loop_ctr - x[2]
def g4 (x):
    return x[7]*app.facebook_post_ctr - x[8]
def g5(x):
    return x[10]*app.youtube_vtr-x[11]
def g6(x):
    return x[12]*app.instagram_video_ctr-x[13]
def g7(x):
    return x[14]*app.instagram_post_ctr-x[15]
def g8(x):
    return x[16]*app.twitter_ctr-x[17]
def g9(x):
    return x[18]*app.facebook_video_vtr - x[9]
if __name__ == '__main__':
    
    random.seed(9001)
    app.run(port=8080)
