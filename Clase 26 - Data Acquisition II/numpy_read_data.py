# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 22:13:59 2021

@author: gabri
"""

import pandas as pd
import numpy as np
import time

filename = 'MNIST_test.txt'
data = np.loadtxt(filename,
                  delimiter=',', # separador
                  skiprows=2,   # saltear filas
                  usecols=[0,2], # intervalo de filas leidas
                  )
# %%time
data1 = np.loadtxt(filename,delimiter=',')


data2 = np.fromfile(filename,dtype=np.float64)
