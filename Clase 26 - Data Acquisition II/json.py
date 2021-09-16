# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 20:37:42 2021

@author: gabri
"""

import json
file_json = open('ejemplo_1.json')
data = json.load(file_json)
print(data)

# y = json.loads(data)