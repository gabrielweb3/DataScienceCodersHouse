# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 20:53:22 2021

@author: gabri
"""

import sqlite3
import pandas as pd

sql = 'SELECT name FROM sqlite_master WHERE type = "table"'
sql2 = 'SELECT * FROM NBA_season1718_salary'
sql3 = 'SELECT * FROM Seasons_Stats'

con = sqlite3.connect('nba_salary.sqlite')
df0 = pd.read_sql_query(sql,con)
df1 = pd.read_sql_query(sql2,con)
df2 = pd.read_sql_query(sql3,con)

jugadores_10M = df1[df1['season17_18']>10000000]

df3 = pd.concat([df2,jugadores_10M],ignore_index=True)

