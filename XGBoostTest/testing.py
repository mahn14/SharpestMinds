import numpy as np
import pandas as pd
import preprocessing as xp 
import model as m 


# importing data
train, test, train_label, test_label = xp.process_data()


##  TRYING OUT TRAINING PARAMETERS
'''
m.grid_score(train, train_label, {}, {})
	# default score
	#	[mean: 0.95374, std: 0.01295, params: {}]
xgb_params1 = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'objective':'binary:logistic'}
gbm_params1 = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}

m.grid_score(train, train_label, xgb_params1, gbm_params1)
	# best score
	#	mean: 0.96256, std: 0.01522, params: {'max_depth': 5, 'min_child_weight': 1}

xgb_params2 = {'n_estimators': 1000, 'seed': 0, 'objective': 'binary:logistic', 'max_depth': 5, 'min_child_weight': 1}
gbm_params2 = {'learning_rate': [0.1, 0.05, 0.01], 'subsample': [0.5, 0.7, 0.9]}

m.grid_score(train, train_label, xgb_params2, gbm_params2)
	# best score (no change)
	#	mean: 0.96256, std: 0.01109, params: {'subsample': 0.5, 'learning_rate': 0.1}

xgb_params3 = {'seed': 0, 'objective': 'binary:logistic', 'max_depth': 5, 'min_child_weight': 1, 'learning_rate':0.1}
gbm_params3 = {'n_estimators': [10, 100, 1000, 2000]}

m.grid_score(train, train_label, xgb_params3, gbm_params3)
	# best score (no change)
	#	mean: 0.96256, std: 0.01522, params: {'n_estimators': 1000}
'''


##	FINDING BOOST ROUNDS USING BEST SCORING PARAMETERS
'''
xgb_params = {'n_estimators': 1000, 'seed': 0, 'objective':'binary:logistic', 
	'max_depth':5, 'min_child_weight':1, 'subsample':0.5, 'learning_rate':0.1} # note that learning_rate is eta for dmatrix

m.Dmatrix(train, train_label, xgb_params)
	# 			  test-mean		  test-std				 train-mean		  train-std  (ERRORS)
	# 221         0.033089        0.015715               0.0              0.0
'''

xgb_params = {'n_estimators': 1000, 'seed': 0, 'objective':'binary:logistic', 
	'max_depth':5, 'min_child_weight':1, 'subsample':0.5, 'eta':0.1} # note that learning_rate is eta for dmatrix

model = m.train_model(train, train_label, xgb_params, 221)

m.importance_check(model, display = 'dict')






'''
params = {'learning_rate': 0.1, 'n_estimators':1000, 'seed':0, 'objective':'binary:logistic', 'max_depth': 3, 'min_child_weight': 1}
m.dmatrix(train, train_label, params, 1000, 5)
'''

'''
params = {'seed':0, 'subsample':0.8, 'colsample_bytree':0.8, 'eta':0.1, 'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1}

final_model = m.train_model(train, train_label, params, num_boostrounds = 100)
m.importance_check(final_model, display = 'dict')
'''






















