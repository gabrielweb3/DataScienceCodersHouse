# procesamiento de datos
import pandas as pd
import numpy as np
# visualizacion de datos
import seaborn as sns
import matplotlib.pyplot as plt
# clasificaicon de datos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

# definicion de dataframe principal
ag01 = pd.DataFrame()

# carga de datos
def carga_de_datos():
    global ag01
    ag01 = pd.read_csv('ARI_datos_crudos.csv')

# informacion basica de datos
def informacion_y_descripcion():
    print(ag01.info())
    print(ag01.describe())

#################### limpieza de datos ##############################
def limpieza_basica():
    global ag01
    # elimino columna de indice de los datos crudos
    ag01 = ag01.drop([ag01.columns[0]],axis=1)
    ag01 = ag01.drop([ag01.columns[6]],axis=1) # columna de disponibilidad no sirve para nada
    
    # renombro columas para facilitar analisis
    # ag01.columns = ag01.rename(columns={ag01.columns[0]: 'Fecha',
    #                             ag01.columns[1]: 'Velocidad',
    #                             ag01.columns[2]: 'Velocidad_std',
    #                             ag01.columns[3]: 'Temperatura',
    #                             ag01.columns[4]: 'Potencia',
    #                             ag01.columns[5]: 'Estado de sistema',
    #                             ag01.columns[6]: 'Disponibilidad',}, inplace=True)
    
    # columns = ['Fecha','Velocidad','Velocidad_std',
    #                         'Temperatura','Potencia','Estado de sistema',
    #                         'Disponibilidad']
    # for col in range(0,len(ag01.columns)):
    #     ag01.columns[col] = columns[col]
    
    # convierto columnas que necesito del dataframe a valores numericos
    for col in range(1,len(ag01.columns)):
        ag01[ag01.columns[col]] = pd.to_numeric(ag01[ag01.columns[col]],errors='coerce')
    
    # convierto columna 'fecha' en tipo datetime
    ag01['fecha'] = pd.to_datetime(ag01['fecha'])
    
    # se filtran nan para poder hacer los analisis de las variables mas ordenadamente
    ag01 = ag01.dropna()

##############################################################################
######################## ANALISIS UNIVARIADO #################################
##############################################################################
# Se hara analisis unicamente de las variables mas importantes, que son las que se
# utilizaran para el modelado
def analisis_univariado():
    def univariado_velocidad():
        global ag01
        # velocidad de viento
        media_de_velocidad_todo_el_perdiodo = ag01[ag01.columns[1]].mean()
        mediana_de_velocidad_todo_el_perdiodo = ag01[ag01.columns[1]].median()
        moda_de_velocidad_todo_el_perdiodo = ag01[ag01.columns[1]].mode()
        print('Media, mediana y moda de velocidad de viento para todo el conjunto de datos')
        print('Media de velocidad de viento: ',media_de_velocidad_todo_el_perdiodo)
        print('Mediana de velocidad de viento: ',mediana_de_velocidad_todo_el_perdiodo)
        print('Moda de velocidad de viento: ',moda_de_velocidad_todo_el_perdiodo)
        
        sns.set_theme(style="whitegrid")
        
        line_velocidad = sns.lineplot(x=ag01['fecha'],y=ag01[ag01.columns[1]],palette='pastel')
        line_velocidad.set_title('Evolucion temporal de velocidad de viento')
        line_velocidad.set_ylabel('Velocidad de viento')
        line_velocidad.set_xlabel('Fecha')
        line_velocidad.grid(True)
        line_velocidad.set_xlim(0,25)
        
        dist_velocidad = sns.distplot(ag01[ag01.columns[1]])
        dist_velocidad.set_title('Distribucion de Weibull de la velocidad de viento')
        dist_velocidad.set_ylabel('Frecuencia relativa')
        dist_velocidad.set_xlabel('Velocidad de viento')
        dist_velocidad.grid(True)
        
        box_velocidad = sns.boxplot(y=ag01[ag01.columns[1]])
        box_velocidad.set_title('Distribucion de velocidad de viento')
        box_velocidad.set_ylabel('Velocidad de viento')
    
    def univariado_potencia():
        global ag01
        
        # potencia electrica
        media_de_potencia_todo_el_perdiodo = ag01[ag01.columns[4]].mean()
        mediana_de_potencia_todo_el_perdiodo = ag01[ag01.columns[4]].median()
        moda_de_potencia_todo_el_perdiodo = ag01[ag01.columns[4]].mode()
        print('Media, mediana y moda de Potencia electrica para todo el conjunto de datos')
        print('Media de Potencia electrica: ',media_de_potencia_todo_el_perdiodo)
        print('Mediana de Potencia electrica: ',mediana_de_potencia_todo_el_perdiodo)
        print('Moda de Potencia electrica: ',moda_de_potencia_todo_el_perdiodo)
        
        line_potencia = sns.lineplot(x=ag01['fecha'],y=ag01[ag01.columns[4]],palette='pastel')
        line_potencia.set_title('Evolucion temporal de la potencia electrica')
        line_potencia.set_ylabel('Potencia Electrica')
        line_potencia.set_xlabel('Fecha')
        line_potencia.grid(True)
        
        dist_potencia = sns.distplot(ag01[ag01.columns[4]])
        dist_potencia.set_title('Distribucion la Potencia Electrica')
        dist_potencia.set_ylabel('Frecuencia relativa')
        dist_potencia.set_xlabel('Potencia Electrica')
        dist_potencia.grid(True)
        
        box_potencia = sns.boxplot(y=ag01[ag01.columns[4]])
        box_potencia.set_title('Distribucion de la Potencia Electrica')
        box_potencia.set_ylabel('Potencia Electrica')
    
    def univariado_temperatura():
        global ag01
        # temperatura del ambiente 
        media_de_temperatura_todo_el_perdiodo = ag01[ag01.columns[3]].mean()
        mediana_de_temperatura_todo_el_perdiodo = ag01[ag01.columns[3]].median()
        moda_de_temperatura_todo_el_perdiodo = ag01[ag01.columns[3]].mode()
        print('Media, mediana y moda de temperatura ambiente para todo el conjunto de datos')
        print('Media de temperatura ambiente: ',media_de_temperatura_todo_el_perdiodo)
        print('Mediana de temperatura ambiente: ',mediana_de_temperatura_todo_el_perdiodo)
        print('Moda de temperatura ambiente: ',moda_de_temperatura_todo_el_perdiodo)
        
        line_temperatura = sns.lineplot(x=ag01['fecha'],y=ag01[ag01.columns[3]],palette='pastel')
        line_temperatura.set_title('Evolucion temporal de la temperatura ambiente')
        line_temperatura.set_ylabel('Temperatura Ambiente')
        line_temperatura.set_xlabel('Fecha')
        line_temperatura.grid(True)
        
        dist_temperatura = sns.distplot(ag01[ag01.columns[3]])
        dist_temperatura.set_title('Distribucion la Temperatura Ambiente')
        dist_temperatura.set_ylabel('Frecuencia relativa')
        dist_temperatura.set_xlabel('Temperatura Ambiente')
        dist_temperatura.grid(True)
        
        box_temperatura = sns.boxplot(y=ag01[ag01.columns[3]])
        box_temperatura.set_title('Distribucion de la Temperatura Ambiente')
        box_temperatura.set_ylabel('Temperatura Ambiente')
        
    univariado_velocidad()
    univariado_potencia()
    univariado_temperatura()

##############################################################################
######################## ANALISIS BIVARIADO ##################################
##############################################################################
# Se hara analisis unicamente de la potencia y la velocidad, que son las que se
# mas importantes para calcular la curva de potencia
# visualizaciones de curva de potencia
def analisis_multivariado():
    
    # import plotly.express as px
    # from plotly.offline import plot
    # fig = px.scatter(x=ag01_filtrado[ag01_filtrado.columns[1]], y=ag01_filtrado[ag01_filtrado.columns[4]], trendline="ols")
    # plot(fig)
    
    
    def multivariado_potencia_velocidad():
        global ag01
        sns.set_theme(color_codes=True)
        curva_de_potencia = sns.scatterplot(x=ag01[ag01.columns[1]],y=ag01[ag01.columns[4]],
                                            marker="+",
                                            color='b')
        curva_de_potencia.set_title('Curva de Potencia (todo el conjunto de datos son filtros)')
        curva_de_potencia.set_ylabel('Potencia')
        curva_de_potencia.set_xlabel('Velocidad')                            
                
        regresion_lineal = sns.regplot(x=ag01[ag01.columns[1]],y=ag01[ag01.columns[4]],
                                       color='r',marker="+")
        print('Coeficiente de correlacion en curva de potencia sin filtrar')
        print('R2 = 0.686510')
    
    multivariado_potencia_velocidad()
    
    analisis_de_componentes_principales()

##############################################################################
###################### DATA CLEANING #########################################
##############################################################################
def filtrado_de_datos():
    global ag01, ag01_filtrado
    
    ag01_filtrado = ag01.copy(deep=True)
    
    # filtros por norma y caracteristicas de maquina
    
    # filtros de velocidad
    ag01_filtrado = ag01[ag01[ag01.columns[1]]>=3.5]
    ag01_filtrado = ag01_filtrado[ag01_filtrado[ag01_filtrado.columns[1]]<23]
    
    # filtros de potencia
    ag01_filtrado = ag01_filtrado[ag01_filtrado[ag01_filtrado.columns[4]]>350]
    
    # filtros de estado de sistema
    ag01_filtrado = ag01_filtrado[ag01_filtrado[ag01_filtrado.columns[5]]!=0]
    
    # vectores de rango 
    velocidades_minimas = np.arange(3.5,25,0.4)
    for i in range(0,len(velocidades_minimas)):
        velocidades_minimas[i] = (velocidades_minimas[i]**2)/5
    
    # filtros de banda envolvente
    # # defino bandas de potencia y de velocidad
    zona_1 = [0,50]; velocidad_min_max_1 = [1.5,6]
    zona_2 = [51,100]; velocidad_min_max_2 = [1.6,6]
    zona_3 = [101,150]; velocidad_min_max_3 = [1.8,6]
    zona_4 = [151,200]; velocidad_min_max_4 = [2,7]
    zona_5 = [201,250]; velocidad_min_max_5 = [2.3,7]
    zona_6 = [251,300]; velocidad_min_max_6 = [2.5,6.6]
    zona_7 = [301,350]; velocidad_min_max_7 = [3.5,6.6]
    zona_8 = [351,400]; velocidad_min_max_8 = [3.7,6.8]
    zona_9 = [401,450]; velocidad_min_max_9 = [4.3,7]
    zona_10 = [451,500]; velocidad_min_max_10 = [4.67,7.3]
    zona_11 = [501,550]; velocidad_min_max_11 = [5.1,7.45]
    zona_12 = [551,600]; velocidad_min_max_12 = [5.32,7.5]
    zona_13 = [601,650]; velocidad_min_max_13 = [5.59,7.72]
    zona_14 = [651,700]; velocidad_min_max_14 = [5.74,7.82]
    zona_15 = [701,750]; velocidad_min_max_15 = [6.07,8.1]
    zona_16 = [751,800]; velocidad_min_max_16 = [6.28,8.3]
    zona_17 = [801,850]; velocidad_min_max_17 = [6.426,8.5]
    zona_18 = [851,900]; velocidad_min_max_18 = [6.66,8.66]
    zona_19 = [901,950]; velocidad_min_max_19 = [6.8,8.77]
    zona_20 = [951,1000]; velocidad_min_max_20 = [6.9,8.92]
    zona_21 = [1001,1050]; velocidad_min_max_21 = [7.09,9]
    zona_22 = [1051,1100]; velocidad_min_max_22 = [7.17,9.2]
    zona_23 = [1101,1150]; velocidad_min_max_23 = [7.25,9.35]
    zona_24 = [1151,1200]; velocidad_min_max_24 = [7.35,9.55]
    zona_25 = [1201,1250]; velocidad_min_max_25 = [7.46,9.72]
    zona_26 = [1251,1300]; velocidad_min_max_26 = [7.6,9.76]
    zona_27 = [1301,1350]; velocidad_min_max_27 = [7.7,9.8]
    zona_28 = [1351,1400]; velocidad_min_max_28 = [7.76,9.9]
    zona_29 = [1401,1450]; velocidad_min_max_29 = [7.95,10.05]
    zona_30 = [1451,1500]; velocidad_min_max_30 = [8.02,10.2]
    zona_31 = [1501,1550]; velocidad_min_max_31 = [8.15,10.3]
    zona_32 = [1551,1600]; velocidad_min_max_32 = [8.24,10.4]
    zona_33 = [1601,1650]; velocidad_min_max_33 = [8.34,10.53]
    zona_34 = [1651,1700]; velocidad_min_max_34 = [8.42,10.6]
    zona_35 = [1701,1750]; velocidad_min_max_35 = [8.48,10.74]
    zona_36 = [1751,1800]; velocidad_min_max_36 = [8.62,10.82]
    zona_37 = [1801,1850]; velocidad_min_max_37 = [8.75,10.9]
    zona_38 = [1851,1900]; velocidad_min_max_38 = [8.77,11.2]
    zona_39 = [1901,1950]; velocidad_min_max_39 = [8.79,11.7]
    zona_40 = [1951,2050]; velocidad_min_max_40 = [9.038,25]
    
    # # filtro condicional para bandas 
    ag01_filtrado = ag01_filtrado[(ag01_filtrado[ag01_filtrado.columns[4]] >= zona_2[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_2[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_2[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_2[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_3[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_3[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_3[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_3[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_4[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_4[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_4[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_4[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_5[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_5[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_5[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_5[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_6[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_6[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_6[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_6[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_7[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_7[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_7[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_7[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_8[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_8[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_8[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_8[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_9[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_9[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_9[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_9[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_10[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_10[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_10[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_10[1])
                                
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_11[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_11[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_11[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_11[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_12[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_12[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_12[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_12[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_13[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_13[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_13[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_13[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_14[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_14[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_14[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_14[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_15[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_15[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_15[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_15[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_16[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_16[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_16[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_16[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_17[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_17[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_17[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_17[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_18[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_18[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_18[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_18[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_19[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_19[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_19[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_19[1])
                                  
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_20[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_20[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_20[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_20[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_21[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_21[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_21[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_21[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_22[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_22[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_22[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_22[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_23[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_23[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_23[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_23[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_24[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_24[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_24[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_24[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_25[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_25[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_25[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_25[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_26[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_26[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_26[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_26[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_27[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_27[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_27[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_27[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_28[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_28[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_28[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_28[1])
                                  
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_29[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_29[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_29[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_29[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_30[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_30[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_30[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_30[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_31[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_31[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_31[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_31[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_32[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_32[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_32[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_32[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_33[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_33[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_33[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_33[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_34[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_34[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_34[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_34[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_35[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_35[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_35[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_35[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_36[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_36[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_36[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_36[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_37[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_37[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_37[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_37[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_38[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_38[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_38[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_38[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_39[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_39[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_39[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_39[1])
                                  | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_40[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_40[1]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_40[0]) 
                                  & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_40[1])
                                    ]
    
    # reseteo index
    ag01_filtrado = ag01_filtrado.reset_index(drop=True)
    
    # creo columna en dataframe principal que marca con un true los datos que pertenecen a la curva de potencia filtrada
    ag01['dentro de curva'] = ag01_filtrado[ag01_filtrado.columns[4]] >= zona_2[0] & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_2[1])& (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_2[0])& (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_2[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_3[0])& (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_3[1])& (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_3[0])& (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_3[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_4[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_4[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_4[0])& (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_4[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_5[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_5[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_5[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_5[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_6[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_6[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_6[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_6[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_7[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_7[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_7[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_7[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_8[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_8[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_8[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_8[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_9[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_9[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_9[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_9[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_10[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_10[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_10[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_10[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_11[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_11[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_11[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_11[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_12[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_12[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_12[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_12[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_13[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_13[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_13[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_13[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_14[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_14[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_14[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_14[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_15[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_15[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_15[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_15[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_16[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_16[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_16[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_16[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_17[0])  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_17[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_17[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_17[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_18[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_18[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_18[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_18[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_19[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_19[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_19[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_19[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_20[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_20[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_20[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_20[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_21[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_21[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_21[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_21[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_22[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_22[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_22[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_22[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_23[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_23[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_23[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_23[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_24[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_24[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_24[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_24[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_25[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_25[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_25[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_25[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_26[0])  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_26[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_26[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_26[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_27[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_27[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_27[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_27[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_28[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_28[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_28[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_28[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_29[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_29[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_29[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_29[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_30[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_30[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_30[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_30[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_31[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_31[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_31[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_31[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_32[0])  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_32[1])  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_32[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_32[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_33[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_33[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_33[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_33[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_34[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_34[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_34[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_34[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_35[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_35[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_35[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_35[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_36[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_36[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_36[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_36[1]) | (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_37[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_37[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_37[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_37[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_38[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_38[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_38[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_38[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_39[0])  & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_39[1])  & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_39[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_39[1])| (ag01_filtrado[ag01_filtrado.columns[4]] >= zona_40[0]) & (ag01_filtrado[ag01_filtrado.columns[4]] <= zona_40[1]) & (ag01_filtrado[ag01_filtrado.columns[1]] >= velocidad_min_max_40[0]) & (ag01_filtrado[ag01_filtrado.columns[1]] <= velocidad_min_max_40[1])
    
    # clasificacion de datos
    # cambio datos true por 1
    ag01['dentro de curva'] = ag01['dentro de curva'].map({ag01['dentro de curva'].unique()[0]:1,ag01['dentro de curva'].unique()[1]:0})
    # y como los datos que no pertenecen a la curva estan marcados con un nan, los cambio por 0

    # # filtro de datos consecutivos para valores numericos
    # datos_consecutivos = 3
    
    # for var in range(1,4):    
 
    #     # filtro de valores consecutivos
    #     filtrados = ag01.groupby((ag01[ag01.columns[var]].shift() != ag01[ag01.columns[var]]).\
    #                           cumsum()).filter(lambda x: len(x) >= datos_consecutivos)  
    #     # esta linea filtra finalmente los datos convirtiendolos de nueva a df
    #     ag01[ag01.columns[var]] = np.where(ag01.index.isin(filtrados.index), np.nan, ag01[ag01.columns[var]][:])
        
      
# visualizacion despues de filtrado
def visualizacion_de_filtrados():
    global ag01,ag01_filtrado
    curva_de_potencia_filtrada = sns.scatterplot(x=ag01_filtrado[ag01_filtrado.columns[1]],
                                                 y=ag01_filtrado[ag01_filtrado.columns[4]],
                                                 marker="+",
                                                 color='g')
    curva_de_potencia_filtrada.set_title('Curva de Potencia (todo el conjunto de datos con filtros)')
    curva_de_potencia_filtrada.set_ylabel('Potencia')
    curva_de_potencia_filtrada.set_xlabel('Velocidad')               
    curva_de_potencia_filtrada.grid(True)             
    print('Coeficiente de correlacion en curva de potencia sin filtrar')
    print('R2 = 0.686510')
    
    # regresion_lineal = sns.regplot(x=ag01_filtrado[ag01_filtrado.columns[1]],
    #                                y=ag01_filtrado[ag01_filtrado.columns[4]],
    #                                color='g',marker="+")
    print('Coeficiente de correlacion de curva de potencia despues de filtrado')
    print('R2 = 0.923241')
    
   
###############################################################################
###################### ALGORITMO DE CLASIFICACION DE DATOS ####################
###############################################################################
# algoritmo de clasificacion RANDOM FOREST
def clasificacion_por_random_forest():
    global ag01
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    features = [ag01.columns[1],ag01.columns[2],ag01.columns[3],ag01.columns[4],ag01.columns[5]]
    target = ['dentro de curva']
    
    #Seleccionamos todas las columnas
    X = ag01[features]
    
    #Defino los datos correspondientes a las etiquetas
    y = ag01[target]
    
    #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    #Se escalan todos los datos
    from sklearn.preprocessing import StandardScaler
    escalar = StandardScaler()
    X_train = escalar.fit_transform(X_train)
    X_test = escalar.transform(X_test)
    
    # #Creamos un arbol de decisión sencillo y lo fiteamos
    tree = DecisionTreeClassifier(max_depth=5,criterion= 'gini',random_state = 123)
    tree.fit(X_train, y_train)
    
    y_test_pred = tree.predict(X_test) #Prediccion en Test
    
    from sklearn.metrics import accuracy_score

    #Calculo el accuracy en Test
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print('% de aciertos sobre el set de evaluación 1:',test_accuracy)
    
    model = RandomForestClassifier(random_state=7, n_estimators=150, criterion="entropy",
                                    class_weight="balanced", max_features="sqrt", n_jobs=6) # 99997, 614
    
    model.fit(X_train, y_train)
    
    y_test_pred = model.predict(X_test) #Prediccion en Test
    
    #Verifico la matriz de Confusión
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix
    # muestro graficamente matriz de confusion 
    confusion_matrix(y_test, y_test_pred)
    
    matriz = confusion_matrix(y_test, y_test_pred)
    print('Matriz de Confusión:')
    print(matriz)
    
    matriz_LR = plot_confusion_matrix(model, X_test, y_test_pred,
                                      cmap='cividis')  
    plt.show()
    
    y_train_pred = model.predict(X_train)
    
    test_accuracy = accuracy_score(y_train, y_train_pred)
    print('Precision del conjunto de train:')
    print(test_accuracy)    
    
    # matrices de confusion
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # train set
    matriz_train = confusion_matrix(y_train, y_train_pred)  
    train_matrriz = ConfusionMatrixDisplay(confusion_matrix=matriz_train)
    train_matrriz.plot()
    
    
    y_test_pred = model.predict(X_test)
    matriz_test = confusion_matrix(y_test, y_test_pred)  
    test_matrriz = ConfusionMatrixDisplay(confusion_matrix=matriz_test)
    test_matrriz.plot()
    
    #Calculo la exactitud del modelo
    from sklearn.metrics import accuracy_score
    
    #Calculo el accuracy en Test
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print('% de aciertos sobre el set de evaluación del random forest:',test_accuracy)


# algoritmo de prediccion de curva de potencia
# redes neuronales
def prediccion_de_potencia():
    # librerias para visualizacion
    import plotly.express as px
    from plotly.offline import plot
    
    global ag01_filtrado
    
    features = [ag01_filtrado.columns[1],
                ag01_filtrado.columns[2],
                ag01_filtrado.columns[3]]
    
    target = [ag01_filtrado.columns[4]]
    
    X = ag01_filtrado[features]
    y = ag01_filtrado[target]

    # divido datos
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=1)

    # curva de potencia con los datos de entrada
    datos_para_curva = []
    for i in range(0,len(y_test)):
        datos_para_curva.append(y_test.values[i][0])
    scatter = px.scatter(x=X_test[ag01_filtrado.columns[1]],
                         y=datos_para_curva)
    scatter.update_layout(title_text="Curva de Potencia Datos de Entrada")
    plot(scatter,filename='Curva de Potencian Datos de Entrada.html')
    scatter.show()

    # escalo datos
    from sklearn import preprocessing
    scaler_train = preprocessing.MinMaxScaler()
    X_train = scaler_train.fit_transform(X_train)
    X_test = scaler_train.transform(X_test)

    # importo libreria de ml
    import tensorflow as tf
    
    # creo capaz ocultasy de salida
    oculta1 = tf.keras.layers.Dense(units=3, input_shape=[3])
    oculta2 = tf.keras.layers.Dense(units=3)
    salida = tf.keras.layers.Dense(units=1)
    
    # creo modelo de rn
    modelo = tf.keras.Sequential([oculta1, oculta2, salida])
    
    # compilacion de modelo
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss='mean_squared_error')
    
    # entreno modelo
    print("Comenzando entrenamiento...")
    historial = modelo.fit(X_train, y_train, epochs=100, verbose=False)
    print("Modelo entrenado!")
    
    # visualizacion de evolucion de la funcion de perdida en funcion del epoch
    import matplotlib.pyplot as plt
    plt.xlabel("# Epoca")
    plt.ylabel("Magnitud de pérdida")
    plt.plot(historial.history["loss"])
    
    # prediccion
    prediccion = modelo.predict(X_test)
    
    # preparo datos para graficar
    predecido = []; real = []
    for i in range(0,len(prediccion)):
        predecido.append(prediccion[i][0])
        real.append(y_test.values[i][0])
    
    # regresion lineal
    scatter = px.scatter(x=predecido,y=real)
    scatter.update_layout(title_text="Regresion Realidad vs Prediccion")
    plot(scatter,filename='Regresion Realidad vs Prediccion.html')


# funcion principal
def funcion_principal():
    carga_de_datos()
    informacion_y_descripcion()
    limpieza_basica()
    analisis_univariado()
    analisis_multivariado()
    filtrado_de_datos()
    visualizacion_de_filtrados()
    clasificacion_por_random_forest()

funcion_principal()



# modelo ml para prediccion de velocidad
# defino features y target
# features = ['CH1Avg','CH1SD','CH1Max','CH1Min']
# features = ['CH1Avg','CH1SD','CH1Max','CH1Min']
features = ['CH1Avg','CH1SD','CH1Max','CH1Min','CH3Avg','CH3SD','CH3Max','CH3Min']
target = ['CH2Avg']
X = datos[features]
y = datos[target]

# divido datos
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=1)


# regresion lineal con datos de test para comparar con regresion predecida
datos_para_regresion = []
for i in range(0,len(y_test)):
    datos_para_regresion.append(y_test.values[i][0])
scatter = px.scatter(x=datos_para_regresion,y=X_test['CH1Avg'], trendline="ols")
scatter.update_layout(title_text="Regresion Datos de Entrada")
plot(scatter,filename='Regresion Datos de Entrada.html')
'''
Resultados regresion
y = 0.994807.X-0.078781
R2 = 0.991647
'''
# distribucion de los datos de entrada
import plotly.figure_factory as ff
hist_data = [datos_para_regresion,X_test['CH1Avg']]
group_labels = ['Velocidad_1','Velocidad_2']
dist_in = ff.create_distplot(hist_data, group_labels, bin_size=.2)
dist_in.show()

# escalo datos
from sklearn import preprocessing
scaler_train = preprocessing.MinMaxScaler()
X_train = scaler_train.fit_transform(X_train)
X_test = scaler_train.transform(X_test)

# importo libreria de ml
import tensorflow as tf

# creo capaz ocultasy de salida
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[8])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)

# creo modelo de rn
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

# compilacion de modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error')

# entreno modelo
print("Comenzando entrenamiento...")
historial = modelo.fit(X_train, y_train, epochs=100, verbose=False)
print("Modelo entrenado!")

# visualizacion de evolucion de la funcion de perdida en funcion del epoch
import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

# prediccion
prediccion = modelo.predict(X_test)

# comparacion perdiccion realidad

# preparo datos para graficar
predecido = []; real = []
for i in range(0,len(prediccion)):
    predecido.append(prediccion[i][0])
    real.append(y_test.values[i][0])

# regresion lineal
scatter = px.scatter(x=real,y=predecido, trendline="ols")
scatter.update_layout(title_text="Regresion Realidad vs Prediccion")
plot(scatter,filename='Regresion Realidad vs Prediccion.html')
'''
Resultados regresion
y = 1.02469.X-0.236943
R2 = 0.992132
'''

# no tiene sentido hacer grafica lineal porque el set de test esta tomado de manera random
# por lo que no tiene una continuidad temporal
# # line plot
# line = px.line(x=datos['Fecha_hora'][len(datos['Fecha_hora'])-len(real)-1:-1],y=[real,predecido])
# line.update_layout(title_text="Evolucion temporal Realidad vs Prediccion")
# plot(line,filename="Evolucion temporal Realidad vs Prediccion.html")

# promedios de real y predecido
print(f'Promedio de velocidad real:{np.mean(real)}')
print(f'Promedio de velocidad predecida:{np.mean(predecido)}')

# calculo de ECM
ECM_prediccion = mean_squared_error(y_test, prediccion)
print('Raiz del Error cuadratico medio:')
print(np.sqrt(ECM_prediccion))