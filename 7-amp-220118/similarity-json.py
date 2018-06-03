import json
import requests
url= "http://127.0.0.1:8081/similar"

json2= {"sales_in_week1" : 4720,"sales_in_week2" : 6562,"sales_in_week3" : 7621,"Target output": 'Itunes Downloads'}

r = requests.post(url,data=json.dumps(json2))
print (r.json())
