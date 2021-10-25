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
 
#####################################################
#Cargo datos desde txt
#####################################################
archivo='con_fecha_filtAG31_F.txt'
data=np.loadtxt(archivo,delimiter='\t',converters={0:bytespdate2num('%d/%m/%Y %H:%M')},skiprows=0,usecols=[0,1,2,3,4,5,6,7])

archivo='con_fecha_filtAG31.txt'
data2=np.loadtxt(archivo,delimiter='\t',converters={0:bytespdate2num('%d/%m/%Y %H:%M')},skiprows=0,usecols=[0,1,2,3,4,5,6,7])

#####################################################
#Variables
#####################################################
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

#Grafico portneica en fucnión de viento para ver puentos
plt.plot(viento,potencia,linestyle='', marker='o', markersize=0.5)
plt.show()
#####################################
#Configuro datos de entrada para entrenar el modelo
#Elijo las variblaes de entrenamiento
#datos_entrada=np.c_[viento,vientoDev,temp,dir_relativa]
datos_entrada=np.c_[viento,vientoDev,temp]
#datos_entrada=np.c_[viento]

#####################################
#configuro los datos para probar el modelo luego de entrenado
#datos_entrada2=np.c_[viento2,vientoDev2,temp2,dir_relativa2]
datos_entrada2=np.c_[viento2,vientoDev2,temp2]
#datos_entrada2=np.c_[viento2]

#####################################
#Copnfiguro la salida para entrenar el modelo
datos_salida=np.c_[potencia/2000]

#####################################
#Configuro la salida para comparar el resultado del modelo
datos_salida2=np.c_[potencia2/2000]

########################################################
#Escalar datos
########################################################
#Es necesario escalar los datos para que tengan valores adecuados al entrenamiento
#Hay varias formas de hacerlo pero en este caso elijo una de la libreria Sklearn
sc=StandardScaler()
#Es necesario que se aplique el escalado al conjutno de datos de entrenamiento mas los de testeo...
#...para no tener diferencias en las pruebas

#Para eso concateno los datos de entrenamiento y de prueba en una misma matriz
datos_entrada_todos=np.concatenate((datos_entrada,datos_entrada2), axis=0)
#Aplico el escalado al conjunto de datos
datos_entrada_todos=sc.fit_transform(datos_entrada_todos)

#Separo los datos de entrenamiento y prueba nuevamente
datos_entrada=datos_entrada_todos[:11999,:]
datos_entrada2=datos_entrada_todos[11999:,:]


#########################################################
#Configuración de la red neuronal
########################################################
#inicilizo la red
model = Sequential()
#Configuro el numero de entradas y el numero de neuronas de la primer capa
model.add(Dense(100, input_dim=3, activation='softplus'))
#Agrego una capa
model.add(Dense(100, activation='softplus'))
#Agrego una capa
model.add(Dense(10, activation='relu'))
#Configuro la salida
model.add(Dense(1, activation='tanh'))

#Configuro el cálculo del error
model.compile(loss='mean_squared_error',#mean_absolute_error
              optimizer='adam',
              metrics=['binary_accuracy'])

#Entrenamiento del modelo 
model.fit(datos_entrada, datos_salida, epochs=50, verbose=False)




#############################################################
#Evaluación del modelo
#############################################################

 
# evaluo el modelo comparandolo contra la salida real (la función me permite medir el error pero no lo hago)
#Se deja esto como ejemplo aunque no se usa
scores = model.evaluate(datos_entrada, datos_salida)

#Guardo la prediccion del modelo aplicada a los datos de entrenamiento para ver que tan bien ajusta
testeo=model.predict(datos_entrada)


#Guardo la predicción del modelo sobre los datos de prueba
resultado=model.predict(datos_entrada2)


#Grafico los puntos de potencia en función de viento y predicción en fucnión de viento 
plt.plot(viento,potencia/2000,linestyle='', marker='o', markersize=0.5)
plt.plot(viento,testeo,linestyle='', marker='o', markersize=1,alpha=0.5,color='#FFFB00')
plt.show()



#Grafico la predicción usando los datos de prueba
plt.plot(viento2,resultado,linestyle='', marker='o', markersize=1,alpha=0.5,color='b')
plt.show()


#Grafico la predicción sobre los datos de prueba y la salida real
plt.plot(viento2,resultado,linestyle='', marker='o', markersize=1,alpha=0.5,color='b')
plt.plot(viento2, datos_salida2,linestyle='', marker='o', markersize=1,alpha=0.5, color='r')
plt.show()


#Voy a calcular el error entre la predicción del modelo y la salida...
#...real tanto en los datos de entrenamiento como en los datos de prueba

resta=datos_salida2-resultado
resta_test=datos_salida-testeo



#Escalo nuevamente ya que la salida del modelo está p.u. y no en potencia nominal 
resta_test=resta_test*2000
resta=resta*2000

#Grafico la resta de la salida real menos la prediccion tanto para el periodo...
#...de entrenamiento como para el periodo de prueba

plt.plot_date(fecha,resta_test,linestyle='', marker='o', markersize=0.5, color='g')
plt.plot_date(fecha2,resta,linestyle='', marker='o', markersize=0.5, color='r')

#Configuo el eje de las x para mostrar las fechas
plt.gcf().autofmt_xdate
date_format=mpl_dates.DateFormatter('%d/%m/%Y')
plt.gca().xaxis.set_major_formatter(date_format)

plt.show()




##############################################################
#Análisis estadistico
##############################################################

#Como la prediccion no es exacta es interesante hacer un estudio estadístico...
#...de la resta entre la salida real y la predicción

#Para esto voy a usar una estructura de Data Frame de la librería Pandas

#Primero concateno los datos en una matriz de numpy
datos_1=np.c_[fecha, resta_test]    
datos_2=np.c_[fecha2,resta]         

#Voy a realizar el promedio móvil en una ventana de tiempo
#Configuro la ventana de tiempo
ventana=6*24*3  #El último valor son los días de la ventana


#Armo la estructura de Data Frames de cada resultado
df_1=pd.DataFrame(data=datos_1, columns=['fecha', 'resta'])

#Creo nuevas columnas en el DF con el promedio y el cuantil 0.5
df_1["prom"]=df_1.resta.rolling(ventana).mean()
df_1["cuan"]=df_1.resta.rolling(ventana).quantile(0.50)

df_2=pd.DataFrame(data=datos_2, columns=['fecha', 'resta'])
df_2["prom"]=df_2.resta.rolling(ventana).mean()
df_2["cuan"]=df_2.resta.rolling(ventana).quantile(0.50)



#Grafico los resultados


#plt.plot_date(fecha,df_1.prom.values,'g-')
plt.plot_date(fecha,df_1.cuan.values,'g-')

#plt.plot_date(fecha2,df_2.prom.values,'r-')
plt.plot_date(fecha2,df_2.cuan.values,'r-')

plt.gcf().autofmt_xdate
date_format=mpl_dates.DateFormatter('%d/%m/%Y')
plt.gca().xaxis.set_major_formatter(date_format)

plt.show()


