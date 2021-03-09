#!/usr/bin/env python3
# %%
import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel('WARNING')

dataset = pd.read_csv('Churn_Modelling.csv')
print(dataset.describe())

# %%
# Pre-Processing
geo = pd.get_dummies(dataset.Geography, drop_first=True)
gender = pd.get_dummies(dataset.Gender, drop_first=True)
df = pd.concat([geo, gender, dataset], axis=1)
# Remove unneeded features...
df.drop(['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender'], axis=1, inplace=True)
print(df.head())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split into Test & Training Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
np.shape(X_train)

# %%
### Initializing the ANN
ann = tf.keras.models.Sequential()
### Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='tanh'))
### Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='softplus'))
### Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

## Part 3 - Training the ANN
### Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# %%
### Training the ANN on the Training set
h_ann = ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
### Predict for Test Set
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

# Evaluate Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# ROC Metrics
Tp, Fp, Fn, Tn = cm.reshape(1, -1, order='F').squeeze()
# https://ccrma.stanford.edu/workshops/mir2009/references/ROCintro.pdf
accuracy = (Tp+Tn) / (Tp+Fp+Tn+Fn)
precision = (Tp) / (Tp+Fp)
recall = (Tp) / (Tp+Fn)
f1_score = 2 * (precision*recall) / (precision+recall)
metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metrics = [accuracy, precision, recall, f1_score]
session_results = pd.DataFrame(metrics, index=metrics_names)
print(session_results)
