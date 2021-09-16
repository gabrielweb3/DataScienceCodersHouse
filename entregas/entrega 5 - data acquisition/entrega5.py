
"""
Entrega 5
data acquisition
adquisicion de json desde web
https://jsonplaceholder.typicode.com/users

1. escribir codigo que lea json https://jsonplaceholder.typicode.com/users
2. pasarlo a dataframe de pandas

"""
# librerias de manejo de datos
import pandas as pd
import json
import requests

url = 'https://jsonplaceholder.typicode.com/users'

response = requests.get(url)
text = response.text

json = json.loads(text)

users = pd.DataFrame(json)

usuarios = pd.DataFrame.from_dict(json)
