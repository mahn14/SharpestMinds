import pandas as pd
import numpy as np
import xlrd

import seaborn as sns 
import matplotlib.pyplot as plt

from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.preprocessing import MinMaxScaler

#filepath = "/Users/mahn/Desktop/Opta/Data/TestSet1.xls"

all_columns = ["FILERECORDID", "ADDRESSID", "OAK", "STREETADDRESS", "STREETNUMBER",	"UNIT",	"STREETNAME", "STREETTYPE", "STREETDIRECTION", "MUNICIPALITY", 
	"PROVINCE", "POSTALCODE", "LATITUDE", "LONGITUDE", "GEOCODE_PRECISION", "PROVIDER_IND", "YEARBUILT", "TOTALFLOORAREA", "ARCHITECTURALSTYLE", 
	"FOUNDATIONTYPE", "BUILDINGSTOREYS", "BATHROOMS", "KITCHENS", "EXTERIORWALL", "ROOFTYPE", "FINISHEDBASEMENT", "ATTACHEDGARAGES"]


# Reads in csv beginning with its header, drops STREETDIRECTION (mostly missing values in sample)
def get_data(filepath):
	'''
		@param
			filepath : path and file name of csv 
		@return
			data : Pandas DataFrame without STREETDIRECTION
	'''
	data = pd.read_csv(filepath, sep = "|", na_values = ["", "UNKNOWN"], encoding = "UTF-16")
	data.columns = all_columns
	data = data.drop("STREETDIRECTION", axis = 1)

	return(data)

# Reads in csv and extracts most location based features
def get_location(filepath):
	'''
		@param
			filepath : path and file name of csv
		@return
			data : Pandas DataFrame with only location features
	'''
	data = pd.read_csv(filepath, sep = "|", na_values = ["", "UNKNOWN"], encoding = "UTF-16")
	area = data["LONGITUDE"] * data["LATITUDE"]
	data = data[["FILERECORDID", "PROVIDER_IND", "STREETADDRESS", "STREETNAME", "STREETTYPE", 
		"OAK", "STREETDIRECTION", "MUNICIPALITY", "PROVINCE", "POSTALCODE", "LONGITUDE", "LATITUDE"]]
	data = pd.concat([data, area], axis = 1)

	return(data)


# Splits data into estimated vs non-estimated DataFrames
def split_est(data):
	est_data = data[data['PROVIDER_IND'] == 'EST']
	real_data = data[data['PROVIDER_IND'] != 'EST']

	return(est_data, real_data)


# Splits data into continuous vs discrete DataFrames
def split_cont_disc(data):

	cont_data = data['FILERECORDID']
	disc_data = data['FILERECORDID']

	data = data.drop('FILERECORDID', axis = 1)
	for feature in data.columns:
		if data[feature].dtype == 'object':
			disc_data = pd.concat([disc_data, data[feature]], axis = 1)
		else:
			cont_data = pd.concat([cont_data, data[feature]], axis = 1)

	return(cont_data, disc_data)


# UNKNOWN and blank values only exist in non-numpy discrete data (for this sample)
def split_NaN(data):

	df1 = data[pd.isnull(data).any(axis = 1)]
	df2 = data.iloc[:].dropna()
	return(df1, df2)

# Convert Categorical to unique numeric values
def encode(data):
	for feature in data.columns:
		data[feature] = pd.Categorical(data[feature]).codes
	return(data)

# Normalizes data
def scale_data(data):
	columns = data.columns
	scaler = MinMaxScaler()
	scaled_data = scaler.fit_transform(data)
	scaled_data = pd.DataFrame(scaled_data)
	scaled_data.columns = columns
	return(scaled_data)

# Splits into train and test data
def split_train_test(data, train_frac):
	'''
		@param
			data : dataset to be split
			train_frac : fraction of data to be used as training data
		@return
			train, test
	'''
	data = data.sample(frac = 1)
	scaled_frac = int(train_frac*data.shape[0])
	
	train = data.iloc[0:scaled_frac, :]
	test = data.iloc[train.shape[0]:, :]
	
	return(train, test)





















