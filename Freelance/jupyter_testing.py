import preprocessing as pp
import exploring as ex

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

file = 'ESTIMATION_MODEL_DATA_20170921.csv'
test_file = 'TestSet.csv'
test_file5m = 'TestSet5m.csv'
columns = ['FILERECORDID', 'ADDRESSID', 'OAK', 'STREETADDRESS', 'STREETNUMBER','UNIT','STREETNAME',
           'STREETTYPE','MUNICIPALITY','PROVINCE','POSTALCODE','LATITUDE','LONGITUDE',
           'GEOCODE_PRECISION','PROVIDER_IND','YEARBUILT','TOTALFLOORAREA','ARCHITECTURALSTYLE','FOUNDATIONTYPE',
           'BUILDINGSTOREYS','BATHROOMS','KITCHENS','EXTERIORWALL','ROOFTYPE','FINISHEDBASEMENT','ATTACHEDGARAGES']

# importing full data
full_data = pp.get_data(test_file5m)
full_data.columns = columns


# splitting data: feature data
feat_data = full_data.drop(["ADDRESSID", "STREETADDRESS", "STREETNUMBER", "UNIT", "STREETTYPE", "GEOCODE_PRECISION", "OAK"], axis = 1)
feat_est, feat_known = pp.split_est(feat_data)
feat_nan, feat_real = pp.split_NaN(feat_known)
feat_real_enc = pp.encode(feat_real)
feat_real_sc = pp.scale_data(feat_real_enc)

# splitting data: location train, test
loc_columns = ["FILERECORDID", "PROVIDER_IND","LONGITUDE", "LATITUDE", 
               "POSTALCODE", "MUNICIPALITY", "STREETTYPE", "EXTERIORWALL",
               "TOTALFLOORAREA", "ARCHITECTURALSTYLE"]
loc_data = full_data[["FILERECORDID", "PROVIDER_IND","LONGITUDE", 
                      "LATITUDE", "POSTALCODE", "MUNICIPALITY", 
                      "STREETTYPE", "EXTERIORWALL", "TOTALFLOORAREA",
                      "ARCHITECTURALSTYLE"]]
loc_est, loc_known = pp.split_est(loc_data)
loc_nan, loc_real = pp.split_NaN(loc_known)
loc_real_enc = pp.encode(loc_real)
loc_real_sc = pp.scale_data(loc_real_enc)
loc_real_enc = loc_real_enc.drop(["FILERECORDID", "PROVIDER_IND"], axis = 1)

loc_train, loc_test = pp.split_train_test(loc_real_sc, 0.75)

loc_train.shape, loc_test.shape, feat_real_sc.shape

# Linear Correlations
ex.correlation_matrix(feat_real_sc)
'''
ExteriorWall: Latitude, PostalCode, Municipality
              FoundationType, TotalFloorArea, YearBuilt, AttachedGarage

TotalFloorArea: PostalCode, Municipality, Latitude
                Bathrooms, Kitchens, RoofType, YearBuilt, FoundationType
                  BuildingStoreys

YearBuilt: Latitude, PostalCode
           TotalFloorArea, FoundationType, Barthrooms, ExteriorWall
              FinishedBasement
'''

# Nonlinear Correlations
tau_df = feat_real_sc.drop(["EXTERIORWALL", "TOTALFLOORAREA", "YEARBUILT", "BATHROOMS", "POSTALCODE", "LATITUDE", "LONGITUDE", "MUNICIPALITY"], axis = 1)
tau_target = feat_real_sc[["EXTERIORWALL", "TOTALFLOORAREA", "YEARBUILT", "BATHROOMS", "POSTALCODE", "LATITUDE", "LONGITUDE", "MUNICIPALITY"]]
ex.kendalltau_corr(tau_df, tau_target, tol = 0.2, overlap = "none") 

scaled_frac = int(0.75*loc_real_sc.shape[0])
    
train = loc_real_sc.iloc[0:scaled_frac, :]

model1 = KNeighborsRegressor()
model1.fit(loc_train.drop("ARCHITECTURALSTYLE", axis = 1), loc_train["ARCHITECTURALSTYLE"] )

predictions = model1.predict(loc_test.drop("ARCHITECTURALSTYLE", axis = 1))
MSE1 = np.mean((loc_test["ARCHITECTURALSTYLE"] - predictions)**2)
print(MSE1) #0.0304

real_train, real_test = pp.split_train_test(loc_real, 0.75)
model2 = KNeighborsClassifier(n_neighbors = 10, weights = 'distance', n_jobs = -1,
                             leaf_size = 10)
model2.fit(real_train.drop("EXTERIORWALL", axis = 1), real_train["EXTERIORWALL"])

predictions2 = model2.predict(real_test.drop("EXTERIORWALL", axis = 1))
accuracy = accuracy_score(real_test["EXTERIORWALL"], predictions2)
print(accuracy) #0.6444













