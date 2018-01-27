# Copyright (c) 2017 Yazabi Predictive Inc.

#################################### MIT License ####################################
#                                                                                   #
# Permission is hereby granted, free of charge, to any person obtaining a copy      #
# of this software and associated documentation files (the "Software"), to deal     #
# in the Software without restriction, including without limitation the rights      #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         #
# copies of the Software, and to permit persons to whom the Software is             #
# furnished to do so, subject to the following conditions:                          #
#                                                                                   #
# The above copyright notice and this permission notice shall be included in all    #
# copies or substantial portions of the Software.                                   #
#                                                                                   #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     #
# SOFTWARE.                                                                         #
#                                                                                   #
#####################################################################################

# This is module contains the function signatures for predicton functions that will work
# on the abalone dataset. The code is heavily commented to allow you to follow along
# easily.

# Please report any bugs you find using @yazabi, or email us at contact@yazabi.com.

import pandas as pd
import data_exploration as de
import data_preprocessing as dp
import pickle

def save_model(model):
    """Saves the model provided as input in the current folder, with the
    name 'optimized_model'.

    params:

    :model: the model to be saved. This should be the optimized model that you
    obtain by testing out each algorithm type, parameter set and model_mode
    that you think is likely to work, and will be used for "in-production"
    predictions.

    returns: nothing.

    Hint: to save your model you can use the pickle package, which has already
    been imported for you. See http://scikit-learn.org/stable/modules/model_persistence.html.
    """
    f = open('model.pkl', 'wb')
    pickle.dump(model, f)


def load_model():
    """Loads the saved model for later use in "production mode".

    params: None

    returns: the optimized model, previously saved by the save_model() function.
    """
    f = open('model.pkl')
    opt_model = pickle.load(f)
    return(opt_model)

def build_and_save_model(model_type,model_mode,params, rings):
    """Once you've played around with various model architectures and
    parameter values in data_exploration.py, you'll choose the best performing
    model_type, model_mode and params values, and use them in production. This
    function takes these inputs and saves a fully trained model, ready to
    make predictions.

    params:

    :model_type: a string indicating the model architecture you've found most
    effective. Can be any one of 'naive_bayes', 'knn', 'svm' or 'decision_tree'.

    :model_mode: a string indicating whether you found the problem was best
    treated as a classification problem or a regression problem (respectively
    denoted by the two allowable values 'classification' and 'regression').

    :params: a dict object containing the parameters you found most optimal
    for the model_type and model_mode provided.

    returns: nothing
    """

    # Arguments
    mtype = model_type
    mmode = model_mode
    parameters = params
    train_i1, train_l1, test_i1, test_l1, train_i2, train_l2, test_i2, test_l2, train_i3, train_l3, test_i3, test_l3 = dp.generate_data()

    # Trains Model
    if rings == "lower":
        model = de.build_model(train_inputs = train_i1, train_labels = train_l1, model_type = mtype, model_mode = mmode, model_params = parameters)
    elif rings == "mid":
        model = de.build_model(train_inputs = train_i2, train_labels = train_l2, model_type = mtype, model_mode = mmode, model_params = parameters)
    elif rings == "greater":
        model = de.build_model(train_inputs = train_i3, train_labels = train_l3, model_type = mtype, model_mode = mmode, model_params = parameters)
    else:
        return("rings must be a string of 'lower', 'mid', or 'greater'")
    # Saves Model
    save_model(model)
    

def predict(inputs):
    """Predicts the Rings values for the inputs provided, based on a saved
    and pretrained model.

    params:

    :inputs: a Pandas dataframe containing the inputs whose Rings values are to
    be predicted.

    returns: the predicted Rings values.
    """
    model = load_model()
    return(model.predict(inputs))


if __name__ == '__main__':
    """Loads training inputs and labels using dp.generate_data(),
    then trains and saves a model using build_and_save_model(). Finally,
    Re-loads the model using load_model, and runs evaluate_model()
    to display the corresponding confusion matrix.
    """
    
    train_i1, train_l1, test_i1, test_l1, train_i2, train_l2, test_i2, test_l2, train_i3, train_l3, test_i3, test_l3 = dp.generate_data()

    param = {'n_neighbors': 15, 'weights': 'uniform', 'algorithm': 'kd_tree', 'p': 3}
    build_and_save_model(model_type = 'knn', model_mode = 'regression', params = param, rings = "lower")
    model1 = load_model()
    MSE1, number1 = de.evaluate_model(model = model1, test_inputs = test_i1, test_labels = test_l1, model_mode = 'regression', rings = "lower")


    ## Needs Work...!!! MSE 16...??
    param = {'n_neighbors': 10, 'weights': 'uniform', 'algorithm': 'kd_tree', 'p': 3}
    build_and_save_model(model_type = 'knn', model_mode = 'regression', params = param, rings = "mid")
    model2 = load_model()
    MSE2, number2 = de.evaluate_model(model = model2, test_inputs = test_i2, test_labels = test_l2, model_mode = 'regression', rings = "mid")

    build_and_save_model(model_type = 'svm', model_mode = 'regression', rings = "greater")
    model3 = load_model()
    MSE3, number3 = de.evaluate_model(model = model3, test_inputs = test_i3, test_labels = test_l3, model_mode = 'regression', rings = "greater")

    MSE = (MSE1*number1 + MSE2*number2 + MSE3*number3)/ (number1 + number2 + number3)
    print("MSE = " + str(MSE))









 
    