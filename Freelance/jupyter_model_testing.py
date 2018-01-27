import data_preparation as dp
import preprocessing as pp 
import exploring as ex

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.grid_search import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Prediction target labels: TotalFloorArea, ExteriorWall, FinishedBasement, Bathrooms
wall_col = ["LONGITUDE", "LATITUDE", 
               "POSTALCODE", "MUNICIPALITY", "STREETTYPE", "EXTERIORWALL"]
floor_col = ["LONGITUDE", "LATITUDE", 
               "POSTALCODE", "MUNICIPALITY", "STREETTYPE", "TOTALFLOORAREA"]
bath_col = ["LONGITUDE", "LATITUDE", 
               "POSTALCODE", "MUNICIPALITY", "STREETTYPE", "BATHROOMS"]
base_col = ["LONGITUDE", "LATITUDE", 
               "POSTALCODE", "MUNICIPALITY", "STREETTYPE", "FINISHEDBASEMENT"]
test_file5m = 'TestSet5m.csv'
columns = ['FILERECORDID', 'ADDRESSID', 'OAK', 'STREETADDRESS', 'STREETNUMBER','UNIT','STREETNAME',
           'STREETTYPE','MUNICIPALITY','PROVINCE','POSTALCODE','LATITUDE','LONGITUDE',
           'GEOCODE_PRECISION','PROVIDER_IND','YEARBUILT','TOTALFLOORAREA','ARCHITECTURALSTYLE','FOUNDATIONTYPE',
           'BUILDINGSTOREYS','BATHROOMS','KITCHENS','EXTERIORWALL','ROOFTYPE','FINISHEDBASEMENT','ATTACHEDGARAGES']

full_data = pp.get_data(test_file5m)
full_data.columns = columns

data = full_data[["FILERECORDID", "PROVIDER_IND","LONGITUDE", 
                      "LATITUDE", "POSTALCODE", "MUNICIPALITY", 
                      "STREETTYPE", "EXTERIORWALL", "TOTALFLOORAREA",
                      "ARCHITECTURALSTYLE"]]
est, known = pp.split_est(data)
nan, real = pp.split_NaN(known)
encoded = pp.encode(real)
scaled = pp.scale_data(encoded)
scaled = scaled.drop(["FILERECORDID", "PROVIDER_IND"], axis = 1)

scaled = scaled.sample(frac = 1)

floor = scaled[floor_col]
floor_train = floor.iloc[0:400000, :]
floor_test = floor.iloc[400001:, :]

    # train, test
floor_train = floor_train[floor_col]
floor_test = floor_test[floor_col]

    # inputs, labels
floor_train_inputs = floor_train.drop('TOTALFLOORAREA', axis = 1)
floor_train_labels = floor_train.pop('TOTALFLOORAREA')
floor_train_labels = pd.DataFrame(floor_train_labels)

floor_test_inputs = floor_test.drop('TOTALFLOORAREA', axis = 1)
floor_test_labels = floor_test.pop('TOTALFLOORAREA')
floor_test_labels = pd.DataFrame(floor_test_labels)

model_floor_kr = KNeighborsRegressor(n_neighbors = 50, weights = 'distance')
model_floor_kr.fit(floor_train_inputs, floor_train_labels)

predictions1 = model_floor_kr.predict(floor_test_inputs)

MSE_floor_kr = np.mean((predictions1 - floor_test_labels)**2)
MSE_floor_kr

wall = scaled[wall_col]
wall_train = scaled.iloc[0:400000, :]
wall_test = scaled.iloc[400001:, :]





wall_train = wall_train[wall_col]
wall_test = wall_test[wall_col]

wall_train_inputs = wall_train.drop('EXTERIORWALL', axis = 1)
wall_train_labels = wall_train.pop('EXTERIORWALL')
wall_train_labels = pd.DataFrame(wall_train_labels)

wall_test_inputs = wall_test.drop('EXTERIORWALL', axis = 1)
wall_test_labels = wall_test.pop('EXTERIORWALL')
wall_test_labels = pd.DataFrame(wall_test_labels)


model_wall_kr = KNeighborsRegressor()
model_wall_kr.fit(wall_train_inputs, wall_train_labels)

predictions2 = model_wall_kr.predict(wall_test_inputs)

MSE_wall_kr = np.mean((predictions2 - wall_test_labels)**2)
MSE_wall_kr


scaled_train = scaled.iloc[0:400000, :]
scaled_test = scaled.iloc[400001:, :]

train_inputs = scaled_train.drop('EXTERIORWALL', axis = 1)
train_labels = scaled_train.pop('EXTERIORWALL')
train_labels = pd.DataFrame(train_labels)

test_inputs = scaled_test.drop('EXTERIORWALL', axis = 1)
test_labels = scaled_test.pop('EXTERIORWALL')
test_labels = pd.DataFrame(test_labels)

model_wall = KNeighborsRegressor()
model_wall.fit(train_inputs, train_labels)

predictions3 = model_wall.predict(test_inputs)
MSE = np.mean((predictions3 - test_labels)**2)
MSE

