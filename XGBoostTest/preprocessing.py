import pandas as pd
import numpy as np

#Load data, assign columns names by index 0 to 31
def load_data(): 
	'''
		@return
			train, test are the training and testing sets, respectively
	'''
	pd.set_option('display.expand_frame_repr', False)

	train = pd.read_csv('train_data.txt', header = None, skiprows = 1, sep = ",")
	test = pd.read_csv('test_data.txt', header = None, skiprows = 1, sep = ",")

	columns = range(32)
	train.columns = columns
	test.columns = columns

	return(train, test)


#Process data by encoding all features from object to numerical dtypes
def process_data():
	'''
		@return
			train, test are the training and testing sets, respectively
	'''
	train, test = load_data()
	combined_set = pd.concat([train, test], axis = 0)

	for feature in combined_set.columns:
		if combined_set[feature].dtype == 'object':
			combined_set[feature] = pd.Categorical(combined_set[feature]).codes

	train = combined_set[:train.shape[0]]
	test = combined_set[train.shape[0]:]

	train_label = train.pop(1)
	test_label = test.pop(1)
	return(train, test, train_label, test_label)