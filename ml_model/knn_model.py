import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from data.fetch_data import get_data

df, target_name = get_data()

# I selected accident, surgical_intervention, and smoking as my variables.
X = df[['accident', 'surgical_intervention', 'smoking']]
y = df[target_name]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# I selected k=1 for the KNN classifier.
k = 1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_train_pred = knn.predict(X_train)

# create confusion matrix
conf_matrix_knn = pd.crosstab(
    y_test,
    y_pred,
    rownames=['Actual'],
    colnames=['Predicted']
)

# compute accuracy on test data
accuracy_knn = (y_pred == y_test).mean()

# display results on test data
print(f"KNN classifier accuracy (k={k}): {accuracy_knn:.2%}\n")
print(conf_matrix_knn)

# Add a 'correct' column for the visualization on test data
test_df = X_test.copy()
test_df[target_name] = y_test
test_df['KNN_prediction'] = y_pred
test_df['correct'] = test_df['KNN_prediction'] == test_df[target_name]

# Add a 'correct' column for the visualization on training data
train_df = X_train.copy()
train_df[target_name] = y_train
train_df['KNN_prediction'] = y_train_pred
train_df['correct'] = train_df['KNN_prediction'] == train_df[target_name]

# Create a visualization of KNN classifier results
os.makedirs("example/e_ml_model/plots", exist_ok=True)

# Create a visualization for training data
# I left this commented out, but feel free to toggle this plot to see training results.
# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#     data=train_df,
#     x='petal length',
#     y='petal width',
#     hue='correct',
#     style='correct',
#     s=100,
#     palette={True: 'green', False: 'red'}
# )

# plt.title('KNN Algorithm (Training Set): Correct vs Incorrect Predictions')
# plt.xlabel('Petal Length (cm)')
# plt.ylabel('Petal Width (cm)')
# plt.legend(title='Prediction Correct')
# plt.grid(True)
# plt.savefig('example/e_ml_model/plots/knn_model_training_results.png', dpi=150)
# plt.close()

# Create a visualization for test data
fig = plt.figure(figsize=(10, 8)) # Adjusted figure size for 3D
ax = fig.add_subplot(111, projection='3d') #

# Define the colors based on the 'correct' column for Matplotlib's scatter function
colors = test_df['correct'].map({True: 'green', False: 'red'})

# Use Matplotlib's scatter function for 3D plotting
# 'risk_score' is used as the z-axis variable
ax.scatter(test_df['accident'], test_df['surgical_intervention'], test_df['smoking'],
           c=colors, s=100, marker='o') #

# Set titles and labels for the axes
ax.set_title('KNN Algorithm: Correct vs Incorrect Predictions in 3D')
ax.set_xlabel('Accident')
ax.set_ylabel('Surgical Intervention')
ax.set_zlabel('Smoking') # Label for the new z-axis

# Create a custom legend as automatic seaborn legends might not work directly in 3D axes
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Incorrect')
green_patch = mpatches.Patch(color='green', label='Correct')
ax.legend(handles=[green_patch, red_patch], title='Prediction Correct')

# Optional: adjust the view angle for a better perspective
ax.view_init(elev=20., azim=-35) #

# The grid option for 3D plots is managed differently, often enabled by default or through ax.grid(True)
ax.grid(True)

# Save the figure
plt.savefig('ml_model/plots/knn_model_test_results_3d.png', dpi=150)

# Display the plot (if you are in an interactive environment)
# plt.show()

plt.close()