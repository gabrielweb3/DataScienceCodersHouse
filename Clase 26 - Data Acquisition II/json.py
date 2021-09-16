# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 20:37:42 2021

@author: gabri
"""
import pandas as pd
import json
file_json = open('ejemplo_1.json')
data = json.load(file_json)
print(data)

# y = json.loads(data)

pandas_json = pd.read_json('ejemplo_1.json',orient='column')
