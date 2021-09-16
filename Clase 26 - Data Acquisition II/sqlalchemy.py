# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 20:31:04 2021

@author: gabri
"""

import sqlalchemy as db

engine = db.create_engine('sqlite://nba_salary.sqlite')
connection = engine.connect()

import pandas as pd
df = pd.read_sql_query('SELECT * FROM Season_Stats',
                       connection)