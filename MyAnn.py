import pandas as pd
#importing the dataset
dataset = pd.read_csv('/home/victorh/Git/building-ann/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values             
Y = dataset.iloc[:, 13].values               

#Enconding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenconder_X_1 = LabelEncoder()
X[:, 1] = labelenconder_X_1.fit_transform(X[:,1])
labelenconder_X_2 = LabelEncoder()
X[:, 2] = labelenconder_X_2.fit_transform(X[:, 2])
ohe = OneHotEncoder(categorical_features = [1])
X = ohe.fit_transform(X).toarray()


#spliting the dataset

from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X, Y , test_size = 0.2 , random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))  
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss= 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [20, 25],
              'epochs': [230 , 250, 500],
              'optimizer': ['adam', 'rmsprop', 'sgd']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           n_jobs = -1,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train) 
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print("Best Parameters: {}".format(best_parameters)) #'batch_size': 20, 'epochs': 230, 'optimizer': 'sgd'
print("best accuracy: {}".format(best_accuracy)) #best accuracy: 0.8605
