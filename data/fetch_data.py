from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import os

#...........................................
# Copied from UCi



def get_dataX():  
    # fetch dataset 
    fertility = fetch_ucirepo(id=244) 
  
    # data (as pandas dataframes) 
    X = fertility.data.features 
  
    # metadata 
    print(fertility.metadata) 
  
    # variable information 
    print(fertility.variables) 

    return X
#...........................................

def get_dataY():  
    # fetch dataset 
    fertility = fetch_ucirepo(id=244) 
  
    # data (as pandas dataframes) 
    y = fertility.data.targets 
  
    # metadata 
    print(fertility.metadata) 
  
    # variable information 
    print(fertility.variables) 

    return y
#...........................................

x = get_dataX()
y = get_dataY()
plt.plot(x, y)
plt.show()