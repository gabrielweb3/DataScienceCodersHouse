import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import StandardScaler

from matplotlib.dates import bytespdate2num
from datetime import datetime
from matplotlib import dates as mpl_dates
 

archivo='con_fecha_filtAG31_F.txt'
data=np.loadtxt(archivo,delimiter='\t',converters={0:bytespdate2num('%d/%m/%Y %H:%M')},skiprows=0,usecols=[0,1,2,3,4,5,6,7])

archivo='con_fecha_filtAG31.txt'
data2=np.loadtxt(archivo,delimiter='\t',converters={0:bytespdate2num('%d/%m/%Y %H:%M')},skiprows=0,usecols=[0,1,2,3,4,5,6,7])


fecha=data[1:12000,0]

potencia=data[1:12000,1]
viento=data[1:12000,2]
vientoDev=data[1:12000,3]
pitch=data[1:12000,4]
dir_relativa=data[1:12000,5]
temp=data[1:12000,6]
estado=data[1:12000,7]
tiempo=np.arange(1,12000)
tiempo=np.c_[tiempo]


fecha2=data2[12001:,0]
potencia2=data2[12001:,1]
viento2=data2[12001:,2]
vientoDev2=data2[12001:,3]
pitch2=data2[12001:,4]
dir_relativa2=data2[12001:,5]
temp2=data2[12001:,6]
tiempo2=np.arange(12001,27454)
tiempo2=np.c_[tiempo2]

plt.plot(viento,potencia,linestyle='', marker='o', markersize=0.5, color='b')

plt.show()

plt.plot(viento,pitch,linestyle='', marker='o', markersize=0.5,color='b')

plt.show()


#______________________________________





#______________________________________

datos_entrada=np.c_[viento,vientoDev,temp,dir_relativa]
#datos_entrada=np.c_[viento,vientoDev,temp]
#datos_entrada=np.c_[viento]


datos_entrada2=np.c_[viento2,vientoDev2,temp2,dir_relativa2]
#datos_entrada2=np.c_[viento2,vientoDev2,temp2]
#datos_entrada2=np.c_[viento2]



datos_salida=np.c_[pitch/30]

datos_salida2=np.c_[pitch2/30]

sc=StandardScaler()


#_________________________________________________

#datos_entrada=sc.fit_transform(datos_entrada)

#sc2=StandardScaler()
#datos_entrada2=sc2.fit_transform(datos_entrada2)

#_____________________________________________________
datos_entrada_todos=np.concatenate((datos_entrada,datos_entrada2), axis=0)

datos_entrada_todos=sc.fit_transform(datos_entrada_todos)



datos_entrada=datos_entrada_todos[:11999,:]
datos_entrada2=datos_entrada_todos[11999:,:]
#_____________________________________________________
model = Sequential()
model.add(Dense(100, input_dim=4, activation='softplus'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='tanh'))
 
model.compile(loss='mean_absolute_error',#mean_squared_error
              optimizer='adam',
              metrics=['binary_accuracy'])
 
model.fit(datos_entrada, datos_salida, epochs=100)
 
# evaluamos el modelo
scores = model.evaluate(datos_entrada, datos_salida)
 
testeo=model.predict(datos_entrada)

resultado=model.predict(datos_entrada2)




plt.plot(viento,pitch/30,linestyle='', marker='o', markersize=0.5)
plt.plot(viento,testeo,linestyle='', marker='o', markersize=1,alpha=0.5,color='#FFFB00')

plt.show()




plt.plot(viento2,resultado,linestyle='', marker='o', markersize=1,alpha=0.5,color='b')

plt.show()

plt.plot(viento2,resultado,linestyle='', marker='o', markersize=1,alpha=0.5,color='b')
plt.plot(viento2, datos_salida2,linestyle='', marker='o', markersize=1,alpha=0.5, color='r')

plt.show()


resta=datos_salida2-resultado
resta_test=datos_salida-testeo




resta_test=resta_test*30
resta=resta*30


plt.plot_date(fecha,resta_test,linestyle='', marker='o', markersize=0.5, color='g')
plt.plot_date(fecha2,resta,linestyle='', marker='o', markersize=0.5, color='r')

plt.gcf().autofmt_xdate
date_format=mpl_dates.DateFormatter('%d/%m/%Y')
plt.gca().xaxis.set_major_formatter(date_format)

plt.show()


#_________________________________
datos_1=np.c_[fecha, resta_test]
datos_2=np.c_[fecha2,resta]

ventana=6*24*5  #El último valor son los días de la ventana

df_1=pd.DataFrame(data=datos_1, columns=['fecha', 'resta'])
df_1["prom"]=df_1.resta.rolling(ventana).mean()
df_1["cuan"]=df_1.resta.rolling(ventana).quantile(0.50)

df_2=pd.DataFrame(data=datos_2, columns=['fecha', 'resta'])
df_2["prom"]=df_2.resta.rolling(ventana).mean()
df_2["cuan"]=df_2.resta.rolling(ventana).quantile(0.50)



#_______________________________________



plt.plot_date(fecha,df_1.prom.values,'g-')
#plt.plot_date(fecha,df_1.cuan.values,'y-')

plt.plot_date(fecha2,df_2.prom.values,'r-')
#plt.plot_date(fecha2,df_2.cuan.values,'b-')

plt.gcf().autofmt_xdate
date_format=mpl_dates.DateFormatter('%d/%m/%Y')
plt.gca().xaxis.set_major_formatter(date_format)

plt.show()




