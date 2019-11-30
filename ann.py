# Import Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

# Data Processing
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()


# First Layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 11))

# Second Layer

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# Output Layer 

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'tanh')) 

# Compiling the ANN
classifier.compile(optimizer = 'adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 30)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

"""
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print()
print(cm)
"""

from sklearn.metrics import explained_variance_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

## Accuracy
print('Acurácia', accuracy_score(y_test, y_pred))
## Median
print ('Média(micro):', recall_score(y_test, y_pred, average='micro'))

print ('Média(macro):', recall_score(y_test, y_pred, average='macro'))

## Varince

print('Variância:', "%.3f" % explained_variance_score(y_test, y_pred))
