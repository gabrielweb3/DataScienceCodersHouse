# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 21:13:08 2021

@author: gabri
"""
# Import modules for data manipulation, maths, and plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set style of plotting area
import matplotlib.style as style
style.use('seaborn')

raw_anscombe = "https://raw.githubusercontent.com/andrewhetherington/python-projects/master/Blog%E2%80%94Anscombe's%20Quartet/anscombes.csv"
anscombe = pd.read_csv(raw_anscombe)

anscombe.columns = anscombe.columns.str.split('_', expand=True)
anscombe.columns = anscombe.columns.swaplevel(1,0)

print(anscombe)


def plot_data(trendlines=False):
    # Set up plotting area - we're going to have four plots in a 2 by 2 grid
    fig = plt.figure(figsize=(12,12))
    # --- FIRST PLOT ---
    # Add top left subplot within plotting area
    ax1 = fig.add_subplot(221)
    # Plot data
    ax1.scatter(anscombe["I"]["x"], anscombe["I"]["y"])
    # Add text
    ax1.text(x=11.5, y = 4.5, s = "",
                fontsize = 300, alpha = .10, ha="center")
    # Set x-axis limits
    ax1.set_xlim(3,20)
    # Set y-axis limits
    ax1.set_ylim(3,13)
    # Remove gridlines
    ax1.grid(False)

    # --- SECOND PLOT ---
    # Add top right subplot within plotting area
    ax2 = fig.add_subplot(222)
    # Plot data
    ax2.scatter(anscombe["II"]["x"], anscombe["II"]["y"])
    # Add text
    ax2.text(x=11.5, y = 4.5, s = "",
                fontsize = 300, alpha = .10, ha="center")
    # Set x-axis limits
    ax2.set_xlim(3,20)
    # Set y-axis limits
    ax2.set_ylim(3,13)
    # Remove gridlines
    ax2.grid(False)

    # --- THIRD PLOT ---
    # Add bottom left subplot within plotting area
    ax3 = fig.add_subplot(223)
    # Plot data
    ax3.scatter(anscombe["III"]["x"], anscombe["III"]["y"])
    # Add text
    ax3.text(x=11.5, y = 4.5, s = "",
                fontsize = 300, alpha = .10, ha="center")
    # Set x-axis limits
    ax3.set_xlim(3,20)
    # Set y-axis limits
    ax3.set_ylim(3,13)
    # Remove gridlines
    ax3.grid(False)

    # --- FOURTH PLOT ---
    # Add bottom left subplot within plotting area
    ax4 = fig.add_subplot(224)
    # Plot data
    ax4.scatter(anscombe["IV"]["x"], anscombe["IV"]["y"])
    # Add text
    ax4.text(x=11.5, y = 4.5, s = "",
            fontsize = 300, alpha = .10, ha="center")
    # Set x-axis limits
    ax4.set_xlim(3,20)
    # Set y-axis limits
    ax4.set_ylim(3,13)
    # Remove gridlines
    ax4.grid(False)    
    
    # Code for plotting trendlines, if desired
    if trendlines == True:
        
        # I miss using R
        def abline(slope, intercept):
            """Plot a line from slope and intercept"""
            axes = plt.gca()
            x_vals = np.array(ax1.get_xlim())
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, '--')
        
        # Loop through each dataset/plot, adding and labelling the trendline
        axs = [ax1, ax2, ax3, ax4]

        for i in range(0,4):
            ax = axs[i]
            z = zs[i]
    
            x_vals = np.array(ax.get_xlim())
            y_vals = z[1] + z[0] * x_vals
            ax.plot(x_vals, y_vals, '--')
            ax.text(x=10, y=4, s='y = {0:.2f}x + {1:.2f}'.format(z[0],z[1]), fontsize=20, alpha=0.75)

# Create plots without trendline
plot_data()
plt.savefig("anscombe_plotted")

# ESTADISTICAS DE FIGURAS
print("Coeficiente de correlación para:\n")

print("Dataset I: " + str(anscombe["I"]["x"].corr(anscombe["I"]["y"])))
print("Dataset II: " + str(anscombe["II"]["x"].corr(anscombe["II"]["y"])))
print("Dataset III: " + str(anscombe["III"]["x"].corr(anscombe["III"]["y"])))
print("Dataset IV: " + str(anscombe["IV"]["x"].corr(anscombe["IV"]["y"])))

print("\n\n(Promedio de los valores de x, promedio de los valores de y) para:\n")

print("Dataset I: (" + str(round(anscombe["I"]["x"].mean(),2)) + ", " + str(round(anscombe["I"]["y"].mean(),2)) + ")")
print("Dataset II: (" + str(round(anscombe["II"]["x"].mean(),2)) + ", " + str(round(anscombe["II"]["y"].mean(),2)) + ")")
print("Dataset III: (" + str(round(anscombe["III"]["x"].mean(),2)) + ", " + str(round(anscombe["III"]["y"].mean(),2)) + ")")
print("Dataset IV: (" + str(round(anscombe["IV"]["x"].mean(),2)) + ", " + str(round(anscombe["IV"]["y"].mean(),2)) + ")")

print("\n\n(Varianza de los valores de x, varianza de los valores de y) para:\n")

print("Dataset I: (" + str(round(anscombe["I"]["x"].var(),2)) + ", " + str(round(anscombe["I"]["y"].var(),2)) + ")")
print("Dataset II: (" + str(round(anscombe["II"]["x"].var(),2)) + ", " + str(round(anscombe["II"]["y"].var(),2)) + ")")
print("Dataset III: (" + str(round(anscombe["III"]["x"].var(),2)) + ", " + str(round(anscombe["III"]["y"].var(),2)) + ")")
print("Dataset IV: (" + str(round(anscombe["IV"]["x"].var(),2)) + ", " + str(round(anscombe["IV"]["y"].var(),2)) + ")")