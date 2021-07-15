# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:04:24 2021

@author: gabri
"""
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

#carga y limpieza de nan
pinguinos = sns.load_dataset('penguins')
pinguinos = pinguinos.dropna()

promedios_especies = pinguinos.groupby(['species']).mean()

# graficos de colores por especie

#primero divido todo el las especies de pinguinos
adelie = pinguinos[pinguinos['species']=='Adelie']
chinstrap = pinguinos[pinguinos['species']=='Chinstrap']
gentoo = pinguinos[pinguinos['species']=='Gentoo']

fig,ax = plt.subplots()
ax.scatter(adelie['bill_length_mm'],adelie['body_mass_g'])
ax.scatter(chinstrap['bill_length_mm'],chinstrap['body_mass_g'])
ax.scatter(gentoo['bill_length_mm'],gentoo['body_mass_g'])

ax.set_title('Correlacion largo/masa')
ax.set_ylabel('masa corporal (g)')
ax.set_xlabel('largo de pico(mm)')
fig.savefig('comparacionpinguinos.pdf')

sns.pairplot(data=pinguinos,hue='species')
sns.savefig('comparacion sns.png')
