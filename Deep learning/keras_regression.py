# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 22:27:35 2017

@author: Mostafa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import skew,norm
from scipy import stats




loc = "forecast.csv"
#loc = r"J:\NERL\Eng\A&I\Op Analysis\SPC_CTC\Environmental_Analysis\Projects\01 3Di_and_Emissions\Short_term_forecast\HistoricalData\Training_for_forecast_2015_17.csv"
data= pd.read_csv(loc)
data.columns = data.columns.str.lower()




data.dayofweek = data.dayofweek.astype(str)
data.month = data.month.astype(str)
data.season = data.season.astype(str)
data.egll_runway_direction = data.egll_runway_direction.astype(str)


num_cols = data.columns[data.dtypes != object]
cat_cols = data.columns[data.dtypes == object].drop("datetime")

# remove the outliers 
plt.scatter(data["traffic"],data.score_3di)
data = data.drop(data[(data.traffic < 2000) & (data.score_3di < 20)].index).reset_index(drop = True)
plt.scatter(data["traffic"],data.score_3di)


air_temp_columns = data.columns[data.columns.str.contains("airtemp|dewpoint|pressure|perceasterly|windspeed|weatherscore|cloud")]

temp_columns = data.columns[data.columns.str.contains("airtemp")]
dew_columns = data.columns[data.columns.str.contains("dewpoint")]
pres_columns = data.columns[data.columns.str.contains("pressure")]
perc_columns = data.columns[data.columns.str.contains("perceasterly")]
wind_columns = data.columns[data.columns.str.contains("windspeed")]
weather_columns = data.columns[data.columns.str.contains("weatherscore")]
cloud_columns = data.columns[data.columns.str.contains("cloud")]




val_columns = data.columns[~data.columns.isin(air_temp_columns)]

train = data[val_columns]

train["avg_cloudscore"] = data[cloud_columns].mean(1)
train["avg_weatherscore"] = data[weather_columns].mean(1)
train["avg_avgwindspeed"] = data[wind_columns].mean(1)
train["avg_perceasterly"] = data[perc_columns].mean(1)
train["avg_pressure"] = data[pres_columns].mean(1)
#train["avg_dewpoint"] = data[dew_columns].mean(1)
train["avg_airtemp"] = data[temp_columns].mean(1)
train.datetime = pd.to_datetime(train.datetime)




####checking for the skewness of the data   
     
skewness = {}    
    
   
for n,i in enumerate(train.columns[train.dtypes != object].drop("datetime")):
    print(i, skew(train[i]))
    skewness[i] = skew(train[i]) 

skewness = pd.Series(skewness)
skewed_feats = skewness[abs(skewness).sort_values(ascending = False)>0.75]



###applying sqrt to possible skewed features    
for i in train[skewed_feats.index].columns:
    print(i , skew(np.sqrt(train[i])))
    train[i] = np.sqrt(train[i])
    
    
    
train_datetime = train["datetime"]

#we won't be using dateime in our features so let's remove that and the socre_3di which is the target variable
X = train.drop(["datetime","score_3di"],1)
y = train.score_3di

X_columns = X.columns
X_num = X.columns[X.dtypes != object]
X_cat = X.columns[X.dtypes == object]

X_train  ,X_test , y_train , y_test = train_test_split(X, y , test_size = 0.2)

sk = StandardScaler()
X= sk.fit_transform(X)

#X = pd.get_dummies(X,drop_first = True)

for i in X[X_num]:
    print(i ,";", skew(X[i]))    


####************Building the keras regression Model***********************######


# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(14, input_dim=14, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)



##evaluation of the model
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# evaluate model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))





# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(14, input_dim=14, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model



np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))




# define wider model
def wider_model():
	# create model
	 model = Sequential()
	 model.add(Dense(20, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, input_dim=14, kernel_initializer='normal', activation='relu'))
	 model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	 model.compile(loss='mean_squared_error', optimizer='adam')
	 return model
 
def wider_model():
    model = Sequential()
    model.add(Dense(20, input_dim = 14, kernel_initializer = "normal", activation = "relu"))
    model.add(Dense(20, input_dim = 14, kernel_initializer = "normal", activation = "relu"))
    model.add(Dense(20, input_dim = 14, kernel_initializer = "normal", activation = "relu"))
    model.add(Dense(1, kernel_initializer = "normal"))
    model.compile(loss = "mean_squared_error", optimizer = "adam")
    return model
    


np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))

y_pred = estimator.predict(X_test)


