'''
Data Wrangling
Leer w_data.csv y p_data.csv, fucionarlos a partir de una 
variable que tengan en comun y realizar los siguientes
ejercicios:
    1. cuantos ID tienen en comun. Nombrar cuales no tienen
    en comun
    2. corregir los target a solo dos valores distintitos
    '<=50k' y '>50k'
    3. calcular la proporcion que hay entre personas con 
    salarios >50k respecto del total para entre 
    personas del mismo sexo, raza y educacion
    por ejemplo, cual es el pocentaje de mujeres con un salario 
    mayor a 50k y compararlo con el de los hombre.
    lo mismo para personas de distintas razas y educacion
'''
# importo librerias necesarias
import pandas as pd
from plotly.offline import plot
import plotly.express as px

# importo datos con los separadores adecuados
p = pd.read_csv('p_data.csv', sep = ';')
w = pd.read_csv('w_data.csv', sep = ';')

# joineo ambos dataframes por la columna ID
data = p.merge(w, how = 'inner', on = 'ID')

# 1 - cuantos ID tienen en comun
cuanto_ID_en_comun = pd.concat([p['ID'],w['ID']],axis=1,keys=['ID1', 'ID2'])
cuanto_ID_en_comun = cuanto_ID_en_comun.dropna()
cuanto_ID_en_comun.insert(2,'coinciden',cuanto_ID_en_comun['ID2']==cuanto_ID_en_comun['ID2'])
cantidad_de_coincidencias = cuanto_ID_en_comun.groupby(['coinciden'])['coinciden'].count()
print('Cantidad de valores de ID coincidentes: ',cantidad_de_coincidencias[1])

# 2 - corregir los target a los valores distintos a <=50k y >50k
print('Valores unicos de la columna (antes de filtrar): ',data['Target'].unique())
# data['Target'] = data['Target'].replace(to_replace = np.nan, value = 'a reemplazar')
data['Target'] = data['Target'].str.replace(data['Target'].unique()[3], data['Target'].unique()[0], regex=True)
data['Target'] = data['Target'].str.replace(data['Target'].unique()[3], data['Target'].unique()[1], regex=True)
data = data.dropna()
print('Valores unicos de la columna (luego de filtrar): ',data['Target'].unique())

# 3 - calcular proporcion de personas con salarios menores a 50k, discriminando
# por sexo, raza y educacion
# filtro dataframe solo para quedarme con Target >50K
data_filtrado = data[data['Target']==' >50K']
# agrupacion de datos con dataframe filtrado
discriminacion_por_sexo = pd.DataFrame(data_filtrado.groupby(['Sex'])['Target'].count())
discriminacion_por_raza = pd.DataFrame(data_filtrado.groupby(['Race'])['Target'].count())
discriminacion_por_educacion = pd.DataFrame(data_filtrado.groupby(['Education'])['Target'].count())
# definicion de metricas con respecto a la totalidad de los casos
discriminacion_por_sexo['% sobre el total'] = 100*discriminacion_por_sexo['Target']/len(data['Target'])
discriminacion_por_raza['% sobre el total'] = 100*discriminacion_por_raza['Target']/len(data['Target'])
discriminacion_por_educacion['% sobre el total'] = 100*discriminacion_por_educacion['Target']/len(data['Target'])
# definicion de metricas con respecto a las personas que ganan mas de 50k
discriminacion_por_sexo['% sobre cantidad que ganan 50K'] = 100*discriminacion_por_sexo['Target']/len(data_filtrado['Target'])
discriminacion_por_raza['% sobre cantidad que ganan 50K'] = 100*discriminacion_por_raza['Target']/len(data_filtrado['Target'])
discriminacion_por_educacion['% sobre cantidad que ganan 50K'] = 100*discriminacion_por_educacion['Target']/len(data_filtrado['Target'])

# visualizacion de datos
# discriminacion por sexo
fig = px.bar(discriminacion_por_sexo, x=discriminacion_por_sexo.index,
             y='% sobre cantidad que ganan 50K')
fig.update_layout(title_text="Porcentaje de personas que ganan mas de 50K segun su sexo")
plot(fig)
# discriminacion por educacion
fig3 = px.bar(discriminacion_por_educacion, x=discriminacion_por_educacion.index,
             y='% sobre cantidad que ganan 50K')
fig3.update_layout(title_text="Porcentaje de personas que ganan mas de 50K segun su educacion")
plot(fig3)
# discriminacion por raza
fig2 = px.pie(discriminacion_por_raza, values='% sobre cantidad que ganan 50K',
              names=discriminacion_por_raza.index, title='Porcentaje de personas que ganan mas de 50K segun su raza')
plot(fig2)
