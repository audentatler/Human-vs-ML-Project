from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates
import seaborn as sns
#...........................................
# Copied from UCi



def get_data():  
    # fetch dataset 
    fertility = fetch_ucirepo(id=244) 
  
    # data (as pandas dataframes) 
    X = fertility.data.features 
    y = fertility.data.targets 
  
    # metadata 
    #print(fertility.metadata) 
  
    # variable information 
    #print(fertility.variables) 

    feature_names = fertility.variables[fertility.variables['role'] == 'Feature']['name'].tolist()
    target_name = fertility.variables[fertility.variables['role'] == 'Target']['name'].values[0]

    df = pd.DataFrame(fertility.data.features, columns=feature_names)
    df[target_name] = fertility.data.targets

    return df, target_name
#...........................................


#print(df.dtypes)
#print(df.describe())

