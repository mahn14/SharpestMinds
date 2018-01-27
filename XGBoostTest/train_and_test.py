import numpy as np 
import pandas as pd 
import xgboost as xgb 
import seaborn as sns
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
import preprocessing as p 
import pickle

train_i, test_i, train_l, test_l = p.process_data()

def save_model(model):
    f = open('model.pkl', 'wb')
    pickle.dump(model, f)

def load_model():
    f = open('model.pkl')
    opt_model = pickle.load(f)
    return(opt_model)

def build_and_save_model():
	#train_i, test_i, train_l, test_l = p.process_data()
	final_xgdmat = xgb.DMatrix(train_i, train_l)
	our_params = {'seed':0, #'subsample': 0.8, 'colsample_bytree': 0.8, 'eta': 0.1, 
		'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1}
	final_model = xgb.train(our_params, final_xgdmat, num_boost_round = 179)
	save_model(final_model)

def predict(inputs):
    model = load_model()
    return(model.predict(inputs))

if __name__ == '__main__':
 	build_and_save_model()
 	model = load_model()

 	testdmat = xgb.DMatrix(test_i)
 	y_pred = model.predict(testdmat)
 	
 	y_pred[y_pred > 0.5] = 1
	y_pred[y_pred <= 0.5] = 0

	print("accuracy: " + str(accuracy_score(y_pred, test_l)))
		# 0.964 accuracy

