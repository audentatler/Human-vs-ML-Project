# How to run: python3 -m getting_started.make_plots
import os
from data.fetch_data import get_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates
import numpy as np

# Original 2D Plot Code
def make_plot(factor_1, factor_2):
    factor_1_label = factor_1.replace('_', ' ')
    factor_2_label = factor_2.replace('_', ' ')
    
    df, target_name = get_data()

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=factor_1,
        y=factor_2,
        hue=target_name,
        style=target_name,
        s=90
    )

    plt.title(f'Fertility Diagnosis: {factor_1_label} vs {factor_2_label}')
    plt.xlabel(f'{factor_1_label}')
    plt.ylabel(f'{factor_2_label}')
    plt.legend(title='Fertility Issues')
    plt.grid(True)
    plt.savefig(f'getting_started/plots/{factor_1_label}_v_{factor_2_label}.png', dpi=150)
    plt.close()

# Retrieving Data
df, target_name = get_data()

#### ALL OF THE BELOW STUFF FROM THIS YOUTUBE VIDEO: https://www.youtube.com/watch?v=E3oTdfKHKCY
# DOES NOT PERTAIN TO MY FINAL PROJECT JUST HERE FOR REFERENCE
# Normalizing Data

#print(df.head())
#print(df.describe())

# Using KMeans clustering
#kmeans = KMeans(n_clusters = 3).fit(df.drop("diagnosis", axis = 1))

#df['class'] = kmeans.labels_.astype(str)

#print(df.head())

def scatter_plot_matrix(df):
    dims = df.drop(["diagnosis", 'class'], axis = 1).columns
    g = sns.pairplot (
        data = df,
        vars = dims, 
        hue = 'class',
        palette = 'tab10',
        corner = True,
        plot_kws = {'s': 8, 'edgecolor': 'w', 'alpha': 0.7}
    )
    g.fig.set_size_inches(7, 7)
    g.fig.suptitle('Fertility Diagnoses', y = 1.02)
    for ax in g.axes.flatten():
        if ax:
            ax.xaxis.label.set_size(5)
            ax.yaxis.label.set_size(5)


    os.makedirs("practice_plots", exist_ok=True)
    plt.title('Fertility Diagnoses.')
    plt.legend(title='Factors')
    plt.xlabel('diagnosis')
    plt.ylabel('class')
    plt.grid(True)
    plt.savefig(f'getting_started/plots/extra_dimensional_plot_labeled.png', dpi=150)
    plt.close()

    print(plt.show())

#scatter_plot_matrix(df)

# Paralell Coordinate Plot
def PCP(df):
    df['class'] = df['class'].astype(str)

    cols_to_plot = [c for c in df.columns if c not in ['diagnosis', 'class']]
    
    plt.figure(figsize=(12,5))

    unique_classes = df['class'].unique()
    palette = sns.color_palette("tab10", len(unique_classes))

    parallel_coordinates(df, class_column='class', cols=cols_to_plot, color=palette, linewidth = 0.5)

    plt.xticks(range(len(cols_to_plot)), cols_to_plot, rotation = 0)
    plt.yticks(rotation=0)

    plt.title('Parallel Coordinates Plot')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.grid(False)
    plt.savefig(f'data/practice_plots/parallel_coordinate_plot.png', dpi=150)
    plt.close()

    print(plt.show())

#PCP(df)

# Sampling data
# COME BACK TO THIS TO CHECK WORK
#data_sampled = df.sample(100)

#scatter_plot_matrix(data_sampled)

# PCA Projection
#pca = PCA(2)
#data_projected = pca.fit_transform(df.drop(['diagnosis', 'class'], axis = 1))
#print(data_projected[:5])

# 3D PLOT CODE
# Copied from https://www.geeksforgeeks.org/python/three-dimensional-plotting-in-python-using-matplotlib/
def make_3d (x_val, y_val, z_val):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    target_colors = list(map(lambda s: 'grey' if s == 'N' else 'r', df[target_name]))

    z = np.linspace(0, 1, 100)
    x = z * np.sin(25 * z)
    y = z * np.cos(25 * z) 

    ax.scatter(df[x_val], df[y_val], df[z_val], c= target_colors)
    ax.set_title('3D Scatter Plot')

    plt.savefig(f'getting_started/plots/{x_val}_v_{y_val}_v_{z_val}.png')

#make_3d('age', 'hrs_sitting', 'child_diseases', )
#make_3d('alcohol', 'smoking', 'high_fevers')
#make_3d('child_diseases', 'age', 'surgical_intervention')

# Plot I Used For My Algorithms
#make_3d('accident', 'surgical_intervention', 'smoking')

#x_values = df.drop('accident', axis = 1).values
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x_values)
#df[df.columns[1:10]] = x_scaled

#y_values = df.drop('surgical_intervention', axis = 1).values
#min_max_scaler = preprocessing.MinMaxScaler()
#y_scaled = min_max_scaler.fit_transform(y_values)
#df[df.columns[1:10]] = y_scaled

#z_values = df.drop('smoking', axis = 1).values
#min_max_scaler = preprocessing.MinMaxScaler()
#z_scaled = min_max_scaler.fit_transform(z_values)
#df[df.columns[1:10]] = z_scaled


                           