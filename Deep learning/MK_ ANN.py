# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[: , 3:13].values
y = dataset.iloc[: , -1].values

#Data preprocessing

# taking care of the categorical data and dummy variables
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
country_encoder = LabelEncoder()
X[: , 1] = country_encoder.fit_transform(X[:,1])
gender_encoder = LabelEncoder()
X[: , 2] = gender_encoder.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[: , 1:]



#Split the data into training set and test set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y , test_size = 0.2 , random_state = 0)


#Deep learning requires the data to be Normalised , so we normalise the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Building the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()
#Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6,activation = "relu",kernel_initializer = "uniform",input_dim = 11))

#addinng the second hidden layer relu is the rectifier function
classifier.add(Dense(units = 6,activation = "relu",kernel_initializer = "uniform"))

#adding the output layer , if you have more categoris your output dim will be equal to the number of categories , and the sigmoid equivillent of the class would be softmax
classifier.add(Dense(units = 1,activation = "sigmoid",kernel_initializer = "uniform"))

#compiling the ANN
#Applying the Sotochastic gradient method to the model
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics = ["accuracy"])

#making the prediction using the classifier
classifier.fit(x=X_train ,y = y_train ,batch_size=10,epochs=100)
y_pred =classifier.predict(X_test)
y_pred = (y_pred >0.5)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test , y_pred)

# to predict a new result we need to replicate the same format as out trained x
# this will need to be an array with all the right dummy variables and label encoder
# Tip in numpy to create a horizontal array use Doubel brackets np.array[[1,2,3,4,5]]


#evaluating the ANN model
# we want to use kfold cross validation wrapped in keras for deep learning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#Kerasclassifier expects a function build_classifier
#this function builds an artifitial neural network

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6,activation = "relu",kernel_initializer = "uniform",input_dim = 11))
    classifier.add(Dense(units = 6,activation = "relu",kernel_initializer = "uniform"))
    classifier.add(Dense(units = 1,activation = "sigmoid",kernel_initializer = "uniform"))
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics = ["accuracy"])
    return classifier
if __name__ == "__main__":    
    classifier = KerasClassifier(build_fn = build_classifier, batch_size =10,epochs=100)
    accuracies = cross_val_score(n_jobs = -1,estimator = classifier , X=X_train , y= y_train , cv = 10,scoring = "accuracy")


#improving ANN
