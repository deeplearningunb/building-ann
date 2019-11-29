# Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

# Evitando Warnings
import warnings
warnings.filterwarnings('ignore')

# Pre Processamento dos Dados
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Dados Categóricos
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


# Adicionando Primeira Layer

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'tanh', input_dim = 11))
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

## A função tanh é mais refinada (suavizada) em relação a função da relu e se da muito bem com valores fortemente positivos e valores fortemente negativos 
## Além disso a função tanh é "semelhante" a função sigmoid o que gerou uma melhor conexão entre as layers aumentando a acuracia em relação aos inputs anteriores

# Segunda Layer

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'tanh'))
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'tanh'))

# Layer De Saída

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) ## Activation Function:  
#classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) ## Activation Function:  


# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 30)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#print()
#print(cm)

## Accuracy e Médias
from sklearn.metrics import classification_report
valores_reais    = y_test
valores_preditos = y_pred
target_names = (['Valores Reais', 'Valores Preditos'])
print()
print('========================Avaliação da Rede========================')
# Média e Acurácia
print(classification_report(valores_reais, valores_preditos, target_names=target_names))

## Variação
from sklearn.metrics import explained_variance_score

print('Variance:', "%.2f" % explained_variance_score(valores_reais, valores_preditos))
print('=================================================================')
