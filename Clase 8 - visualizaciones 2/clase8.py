"""
clase 8 
visualizaciones 2
MATPLOTLIB
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# cambio estilo de graficos
mpl.style.use('bmh')

# si pongo eso actualiza todos los graficos a estilo seaborn
# sns.set()

# importa datos de lluvias
df_lluvias = pd.read_csv('pune_1965_to_2002 (1).csv')

fig, ax = plt.subplots(nrows=3,ncols=1,
                       figsize=(12,5),sharex=True,
                       sharey=True)

ax[0].plot(df_lluvias.index,df_lluvias['Jan'],label='Precipitaciones Enero')
ax[1].plot(df_lluvias.index,df_lluvias['Feb'],label='Precipitaciones Febrero',color='C1')
ax[2].plot(df_lluvias.index,df_lluvias['Mar'],label='Precipitaciones Marzo',color='C2')

ax[0].set_title('Precipitaciones del primer trimestre')
ax[2].set_xlabel('Ano')
ax[1].set_ylabel('Precipitaciones (mm)')

ax[0].legend()
ax[1].legend()
ax[2].legend()

fig.savefig('trimestral subplot MPL.pdf')

# ver parametros de configuracion de matplotlib
# mpl.rcParams.keys()
# resetear parametros de fabrica
mpl.rcParams.update(mpl.rcParamsDefault)

#otras configuraciones
# plt.rc('axes', titlelocation='left', titlecolor='firebrick', ...)
# plt.rc('grid', color='black', linestyle='-.', ...)

# configurando parametros de matplotlib
# mpl.rcParams['axes.titleweight'] = 'bold' 
# mpl.rcParams['axes.titlelocation'] = 'left' 
# mpl.rcParams['axes.titlecolor'] = 'firebrick' 
# mpl.rcParams['axes.labelcolor'] = 'blue' 
# mpl.rcParams['axes.labelsize'] = '10' 
# mpl.rcParams['axes.labelweight'] = 'light' 
# mpl.rcParams['axes.linewidth'] = '1' 
# mpl.rcParams['grid.color'] = 'black' 
# mpl.rcParams['grid.linestyle'] = '-.' 
# mpl.rcParams['grid.linewidth'] = '2' 

fig,ax = plt.subplots(figsize=(7,4))
ax.scatter(df_lluvias['Aug'],df_lluvias['Sep'],c=df_lluvias.index)
ax.set_title('Correlacion de lluvias Agosto/Setiembre')
ax.set_xlabel('Agosto')
ax.set_ylabel('Setiembre')

fig.savefig('ScaterRelacion MPL.pdf')