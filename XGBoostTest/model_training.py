# This file will contain your train_and_test.py script.

# Upon running:
#	trains xgb algorithm with training set
#	saves model
#	reloads and applies model to predict labels of testing set
import numpy as np 
import pandas as pd 
import xgboost as xgb 
import seaborn as sns
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
import preprocessing as p 


train, test, train_label, test_label = p.process_data()

# Grid searching for parameter accuracies
def grid_score(xgb_params, gbm_params):
	'''
		xgb_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'objective':'binary:logistic'}
		gbm_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
	'''
	grid_GBM = GridSearchCV(xgb.XGBClassifier(**xgb_params), gbm_params, scoring = 'accuracy')
	grid_GBM.fit(train, train_label)
	print(grid_GBM.grid_scores_)

def grid_search(train, train_label):
	xgdmat = xgb.DMatrix(train, train_label) # Create our DMatrix to make XGBoost more efficient
	our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
	             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1} 
	# Grid Search CV optimized settings
	cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 3000, nfold = 5,
	                metrics = ['error'], # Make sure you enter metrics inside a list or you may encounter issues!
	                early_stopping_rounds = 100) # Look for early stopping that minimizes error
	print(cv_xgb.tail(5))


def train_test(train, train_label, test, test_label):

	# Training
	xgdmat = xgb.DMatrix(train, train_label)
	our_params = {'seed':0, # 'subsample': 0.8, 'colsample_bytree': 0.8, 'eta': 0.1, 
		'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1}
	final_gb = xgb.train(our_params, xgdmat, num_boost_round = 379)

	# Testing
	testdmat = xgb.DMatrix(test)
	y_pred = final_gb.predict(testdmat) # Predict using our testdmat
	print(y_pred)

	y_pred[y_pred > 0.5] = 1
	y_pred[y_pred <= 0.5] = 0

	print("accuracy: " + str(accuracy_score(y_pred, test_label)))
	print("error rate: " + str(1-accuracy_score(y_pred, test_label)))

def testing_my_code():
	xgb_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'objective':'binary:logistic'}
	gbm_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
	grid_score(xgb_params, gbm_params)
	grid_search(train, train_label)
	train_test(train, train_label, test, test_label)

