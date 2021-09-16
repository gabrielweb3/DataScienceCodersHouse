# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 22:02:24 2021

@author: gabri
"""

import requests
import json
import pandas as pd

url = 'https://raw.githubusercontent.com/bttmly/nba/master/data/teams.json'

respuesta = requests.get(url)
print(respuesta)

texto = respuesta.text

jsondata = json.loads(texto)

df = pd.DataFrame.from_dict(jsondata)
print(df.head())