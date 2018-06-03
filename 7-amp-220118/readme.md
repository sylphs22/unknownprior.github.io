######ML module documentation#######
## requirement install
	*install python 3.6
	#install in python 3.6  requirements.txt using following 
	pip install -r requirements.txt
	

##web app (advert planner model) Artist in database
    * Start with python training.py and run this in background
    * Start with python 'plan-api.py' in background
    * The 'plan-api.py' takes values as input provided by the user in 'Frontend' of the sheet
    * The notes are provided in the 'Frontend' sheet which decribes the values that are the input to the 'plan-api.py'.
    * The input given to the 'plan-api.py' is also in json format which is available in 'plan-json.py'
    * The json format input is to be mapped with the 'Frontend' values provided by the user 
    * The output can be checked by running python 'plan-json.py'
    * The output of the 'plan-api.py' shall be returned in json format and populated in the google sheet shared
	
##web app (advert planner model) Artist not in database
    * start with python similarity.py which inputs sales in week1, sales in week2 ... and ouputs similar 'artist_id and track_id'
    * Map the 'artist_id and track_id'  with the artist and track name of 7-amp db
    * start python plan-sim-api.py and run it to get the output
    * The output can be checked by running python 'plan-sim-json.py'
    * The output of the 'plan-sim-api.py' shall be returned in json format and populated in the google sheet shared
