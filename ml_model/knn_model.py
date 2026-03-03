import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from example.e_data.fetch_data import load_iris_data

df, target_name = load_iris_data()

# I selected only the petal length and petal width features for classification.
X = df[['petal length', 'petal width']]
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
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=test_df,
    x='petal length',
    y='petal width',
    hue='correct',
    style='correct',
    s=100,
    palette={True: 'green', False: 'red'}
)

plt.title('KNN Algorithm: Correct vs Incorrect Predictions')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Prediction Correct')
plt.grid(True)
plt.savefig('example/e_ml_model/plots/knn_model_test_results.png', dpi=150)
plt.close()