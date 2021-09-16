import pandas as pd

df1 = pd.read_csv('encuesta.txt',sep=',')
print(df1.head)
# funciona mal


df2 = pd.read_csv('encuesta.txt',sep='/t')
print(df2.head)


df3 = pd.read_csv('encuesta.txt')
print(df3.head)

df4 = pd.read_csv('encuesta.txt',sep=' ')
print(df4.head)


df5 = pd.read_table('encuesta.txt')
print(df5.head)


df6 = pd.read_table('encuesta.txt',sep=' ')
print(df6.head)


df7 = pd.read_table('encuesta.txt',sep=' ',header=None)
print(df7.head)


# df8 = pd.read_csv('encuesta.txt',sep=',',parse_dates=['ID'])
# print(df8.head)

df9 = pd.read_csv('data.csv')
print(df9.head)