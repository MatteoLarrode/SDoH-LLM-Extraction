import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import re
from IPython.display import display, Markdown
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Colour palette
# From https://brand.ifrc.org/ifrc-brand-system/basics/colour
colour_palette = {
    'ifrc_red': '#EE2435',
    'ifrc_darkblue': '#011E41',
    'dark_green': '#009775',
    'medium_green': '#00AB84',
    'light_green': '#47D7AC',
    'medium_blue': '#8DCDE2',
    'light_blue': '#CCf5FC',
    'medium_orange': '#FF8200',
    'light_orange': '#FFB25B',
    'medium_purple': '#512D6D',
    'light_purple': '#958DBE',
    'grey': '#A7A8AA',
}

def set_visualization_style():
    plt.style.use('seaborn-v0_8-colorblind')
    #font_path = {include font path}
    #font_manager.fontManager.addfont(font_path)
    #prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = 'sans-serif'
    #plt.rcParams['font.sans-serif'] = prop.get_name()
    plt.rcParams.update({
        'text.usetex': False,
        #'font.family': 'serif',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 1.5,
        'lines.markersize': 8,
        'figure.figsize': (10, 6),
        'axes.grid': False, 
        'axes.spines.top': False,  # Remove top spine
        'axes.spines.right': False,  # Remove right spine
        # Add this line to use ASCII hyphen instead of Unicode minus
        'axes.unicode_minus': False
    })

