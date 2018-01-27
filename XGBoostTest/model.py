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

# Grid searching for parameter accuracies
def grid_score(train, train_label, xgb_params, gbm_params):
	'''
		@params
			xgb_params: held constant
			gbm_params: array of values to be tested and compared
		@return
			scoring based on mean, std with the varying gbm_params
	'''
	grid_GBM = GridSearchCV(xgb.XGBClassifier(**xgb_params), gbm_params, scoring = 'accuracy', cv = 5, n_jobs = -1)
	grid_GBM.fit(train, train_label)
	print(grid_GBM.grid_scores_)



# Find iter efficiency
def Dmatrix(train, train_label, xgb_params, num_boost_rounds = 3000, nfold = 5, early_stopping_rounds = 100):
	'''
		@params
			xgb_params: best scoring params from grid_score
			num_boost_rounds: max number of iterations (? i think)
			nfold: number of folds in cv
			early_stopping_rounds: will train until cv error hasn't decreased in this many rounds
		@return
			table of test/train errors for mean/std with the round number
	'''

	xgdmat = xgb.DMatrix(train, train_label) # Create our DMatrix to make XGBoost more efficient

	# Grid Search CV optimized settings
	cv_xgb = xgb.cv(params = xgb_params, dtrain = xgdmat, num_boost_round = num_boost_rounds, nfold = nfold,
	                metrics = ['error'], # Make sure you enter metrics inside a list or you may encounter issues!
	                early_stopping_rounds = early_stopping_rounds) # Look for early stopping that minimizes error
	print(cv_xgb.tail(5))



# Train final model using information from grid_scores, dmatrix
def train_model(train, train_label, xgb_params, num_boostrounds = 500):
	'''
		our_params = {'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'eta': 0.1, 
		'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1}
	'''
	xgdmat = xgb.DMatrix(train, train_label)
	final_gb = xgb.train(xgb_params, xgdmat, num_boost_round = num_boostrounds)

	return(final_gb)


def importance_check(model, display):
	importances = model.get_fscore()

	if display == 'dict':
		print(importances)

'''
	elif display == 'plot:':
		df = pd.DataFrame({'Importance':list(importances.values()), 'Feature':list(importances.keys())})
		df.sort_values(by = 'Importance', inplace = True)
		importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')
'''






'''
# Testing
testdmat = xgb.DMatrix(test)
y_pred = final_gb.predict(testdmat) # Predict using our testdmat
print(y_pred)

y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0


# Results
print("accuracy: " + str(accuracy_score(y_pred, test_label)))
print("error rate: " + str(1-accuracy_score(y_pred, test_label)))
'''
