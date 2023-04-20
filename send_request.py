import requests
import json
import pandas as pd
import numpy as np

url = 'http://localhost:5000/linear-regression'
data = pd.read_csv('Book2.csv')
mileage = np.array(data['Mileage'].tolist())
mileage_normalized = (mileage - np.mean(mileage)) / np.std(mileage)
cost = data['Cost'].str.replace('[£,]', '', regex=True).astype(float).tolist()

payload = {
    'mileage': mileage_normalized.tolist(),
    'cost': cost
}

json_payload = json.dumps(payload)

headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json_payload, headers=headers)

if response.status_code == 200:
    print("Response received:")
    print(response.json())
else:
    print("Error:", response.status_code, response.text)
