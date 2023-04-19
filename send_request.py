import requests
import json

url = 'http://localhost:5000/linear-regression'
data = {
    "features": [1, 2, 3, 4, 5],
    "labels": [100, 200, 300, 400, 500]
}

headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers)

if response.status_code == 200:
    print("Response received:")
    print(response.json())
else:
    print("Error:", response.status_code, response.text) 
