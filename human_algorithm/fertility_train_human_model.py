import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data.fetch_data import get_data
from human_algorithm.fertility_human_classifier import humanAlgorithm
from sklearn.model_selection import train_test_split
import numpy as np


# This section of code separates the whole data-set into training and testing data.
df, target_name = get_data()
train_df, test_df = train_test_split(
    df,
    test_size=0.02,
    random_state=42,
    stratify=df[target_name]
)
# USE DF.SHAPE AND MAKE NEW DATA SET
# This section of code applies the human classification algorithm to the test data.
# Optimal approach using apply()
test_df['altered'] = None
test_df['normal'] = None


hN, hA = humanAlgorithm(df['accident'], df['surgical_intervention'], df['smoking'])

for index, row in test_df.iterrows():
    surg = row['surgical_intervention']
    acc = row['accident']
    smoke = row['smoking']
    
    # Unpack the two return values
    
    test_df.loc[index, 'altered'] = hN
    test_df.loc[index, 'normal'] = hA

print(test_df['altered'])
print(test_df['normal'])
# The correct way to pair them row-by-row
test_df['human_prediction'] = list(zip(test_df['altered'], test_df['normal']))



test_df['correct'] = test_df['human_prediction'] == test_df[target_name]
accuracy = (test_df['human_prediction'] == test_df[target_name]).mean()
print(f"Human classifier accuracy: {accuracy:.2%}")

# Here we print the confusion matrix to see how well the human classifier performed on the test-data subset.
test_df['human_prediction'] = test_df['human_prediction'].apply(tuple)

# Now run your crosstab
# Convert lists to tuples so they are hashable
test_df['human_prediction'] = test_df['human_prediction'].apply(tuple)

#print(test_df.columns)
# Now run your crosstab
#test_df.fillna
#conf_matrix = pd.crosstab(
    #test_df[target_name],
    #test_df['human_prediction'],
    #rownames=['Actual'],
    #colnames=['Predicted']
#)
#print(conf_matrix)


# Finally, we print one example of a failure case where the human classifier got the prediction wrong.
failure_row = test_df[test_df['human_prediction'] != test_df[target_name]].iloc[0]
print("\nFAILURE EXAMPLE")
print(failure_row[['accident', 'surgical_intervention', 'smoking', target_name, 'human_prediction']])


# Print a scatter plot showing correct vs incorrect predictions.
os.makedirs("example/e_ml_model/plots", exist_ok=True)

def make_3d_confusion(x_val, y_val, z_val):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    target_colors = list(map(lambda s: 'g' if s == 'N' else 'r', test_df[target_name]))

    z = np.linspace(0, 1, 100)
    x = z * np.sin(25 * z)
    y = z * np.cos(25 * z) 

    ax.scatter(test_df[x_val], test_df[y_val], test_df[z_val], c= target_colors)
    ax.set_title('3D Accuracy Plot')

    plt.savefig(f'human_algorithm/plots/accuracy_scatter_plot.png')

make_3d_confusion('accident', 'surgical_intervention', 'smoking')