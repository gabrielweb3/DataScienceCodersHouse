'''
ejercicio de clases
fusion de dataframes

1 - leer los excel clase0 y clase1 y realizar merge 
con diferentesestrategias (inner, outer, left, right),
que se observa?

2 - con los mismos dataframes utilizar concat,
 que se observa?


'''


import pandas as pd 


clase0 = pd.read_excel('clase0.xlsx')
clase1 = pd.read_excel('clase1.xlsx')


# 1 merge

# inner: use intersection of keys from both frames,
# similar to a SQL inner join; 
# preserve the order of the left keys.
inner = pd.merge(clase0,clase1,on=['key1','key2'])
# inner = pd.merge(clase0,clase1)

# outer: use union of keys from both frames,
# similar to a SQL full outer join; sort keys 
# lexicographically.
outer = pd.merge(clase0,clase1,how='outer',on=['key1','key2'])
# outer = pd.merge(clase0,clase1,how='outer')

# left: use only keys from left frame, similar to 
# a SQL left outer join; preserve key order.
left = pd.merge(clase0,clase1,how='left',on=['key1','key2'])
# left = pd.merge(clase0,clase1,how='left')

# right: use only keys from right frame, 
# similar to a SQL right outer join; preserve key order.
right = pd.merge(clase0,clase1,how='right',on=['key1','key2'])
# right = pd.merge(clase0,clase1,how='right')


# 2 concatenar
concatenados_filas = pd.concat([clase0,clase1])
concatenados_columnas = pd.concat([clase0,clase1],axis=1)