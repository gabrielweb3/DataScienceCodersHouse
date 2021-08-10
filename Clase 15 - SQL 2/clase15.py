# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 22:10:49 2021

@author: gabri
"""

import sqlite3 as s3

con = s3.connect('example.db')
con = s3.connect('example.db')

# crear e insertar datos
cur = con.cursor()

cur.execute('''CREATE TABLE products
               (ProductID int,ProductName text,SupplierID int,CategoryID int, Unit text,Price real)''')

# insert row of data
cur.execute('INSERT INTO products VALUES(78,"Queso Crema",2,4,"1 kg pkg.",30)')