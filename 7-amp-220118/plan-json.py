import json
import requests
url= "http://127.0.0.1:8080/const"
json2 = {"Target Output":"Itunes Downloads","track_id":419,"artist_id":267,"Release Date":'21/04/2017',"Campaign Launch":'21/04/2017',
     "Campaign End":'27/04/2017','budget':1000,'facebook_post_cpm':4.85,'facebook_post_ctr':2.5,
     'facebook_video_cpm':3.71,'facebook_video_ctr':1.49,'facebook_video_vtr':10,
     'youtube_cpm':7.53,'youtube_vtr':26.19,'instagram_post_cpm':3.71,'instagram_post_ctr':0.78,
     'instagram_video_cpm':6.36,'instagram_video_ctr':0.93,'twitter_cpm':6.98,
     'twitter_ctr':1.97,'capitalfm_com_cpm':20.67,'capitalfm_com_ctr':14.29,'loopme_cpm':18,
     'loopme_ctr':10.41,'jet_capital_cpm':1.30,'jet_heart_cpm':1.18,'jet_kiss_cpm':1.25,
     'facebook_post_min_spend':0,'facebook_video_min_spend':0,'youtube_min_spend':0,
     'instagram_post_min_spend':0,'instagram_video_min_spend':0,'twitter_min_spend':0,'capitalfm_com_min_spend':0,
     'loopme_min_spend':0,'jet_capital_min_spend':0,'jet_heart_min_spend':0,'jet_kiss_min_spend':0,
     'facebook_post_max_spend':1000,'facebook_video_max_spend':1000,'youtube_max_spend':1000,'instagram_post_max_spend':1000,
     'instagram_video_max_spend':1000,'twitter_max_spend':1000,'capitalfm_com_max_spend':1000,'loopme_max_spend':1000,
     'jet_capital_max_spend':1000,'jet_heart_max_spend':1000,'jet_kiss_max_spend':1000}


r = requests.post(url,data=json.dumps(json2))
print (r.json())

