import requests

import json

url = "	http://dummy.restapiexample.com/api/v1/employees"

data = requests.get(url).json()

print(data)
