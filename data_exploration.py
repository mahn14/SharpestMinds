
import pandas as pd
import matplotlib.pyplot as plt
import data_preprocessing as dp
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
import numpy as np

import data_preprocessing as dp

def plot_raw_data(dataset):
    """Generates plot(s) to visualize the value of Rings as a function of
    every column of raw data in dataset. You may use whatever plotting format
    seems most appropriate (scatter plot, dot plot, histogram, etc.). The
    only critical thing is that any correlations between the Rings value and
    the other variables in dataset be made visible. Note: you may use different
    plot styles for different variables. For example, it probably makes sense
    to plot Rings vs Sex as a series of histograms, with Rings on the x-axis
    (this is because Sex is a categorical variable). For other variables,
    a scatter plot might make more sense.

    inputs:

    :dataset: a Pandas dataframe containing raw (unnormalized) data obtained
    from the data-containing CSV file.

    returns: nothing
    """

    #load dataframe
    abalone_data = dp.load_dataset('abalone.data.txt')
    num_features = abalone_data.drop('Sex', axis = 1)
    num_features = num_features.drop('Rings', axis = 1)
    rings = abalone_data['Rings']
    sex = abalone_data['Sex']

        #- Visualizing Distributions
    #plotting each feature's count distribution
    rings_dist = abalone_data['Rings'].hist()
    plt.title('Rings')
    plt.show()
    feature_dist = abalone_data.drop('Rings', axis = 1).hist()
    plt.title('Features')
    plt.show()
    #plotting Rings by gender
    plot1 = abalone_data.loc[abalone_data['Sex'] == 'M']['Rings']
    plot2 = abalone_data.loc[abalone_data['Sex'] == 'F']['Rings']
    plot3 = abalone_data.loc[abalone_data['Sex'] == 'I']['Rings']
    plot_hist = pd.DataFrame({'Male': plot1, 'Female': plot2, 'Infant': plot3})
    plot_hist.plot.hist(alpha = 0.3)
    plt.title('GenderRing Histogram')
    plt.show()

        #- Visualizing Correlations
    
    for feature in num_features.columns:
        plt.scatter(num_features[feature], rings)
        plt.title(feature + ' vs Rings')
        plt.show()
   
    


def dimensionality_reduction(dataset, algorithm, sampled_frac=0.05):
    """Generates 2-D representations of the data provided in dataset,
    using the algorithm specified by the algorithm parameter. This script should
    save the plots it generates as JPEG files named 'mds.jpg', 'tsne.jpg' or
    'isomap.jpg', as appropriate (see below).

    params:

    :dataset: a pandas dataframe obtained by processing the raw data in the
    txt/csv file with the preprocess_dataset() function in the
    data_preprocessing.py module.
    

    :algorithm: a string providing the name of the algorithm to be used
    for dimensionality reduction. Can take on the values 'MDS', 'TSNE' or
    'isomap'. See http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html,
    http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html and
    http://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html.

    :sampled_frac: a float indicating the fraction of samples in the total
    dataset that are to be used for dimensionality reduction and visualization.
    You'll notice that if you try to run TSNE, MDS or Isomap with all samples,
    it will take way too long to run - that's why we use such a small

    d
    returns: nothing

    Hint 1: to save yourself some work, you can just use the split_train_and_test()
    function in the data_preprocessing.py module to obtain a reduced dataset.

    Hint 2: you'll probably want to color the points on your 2-D plots based on
    the category they belong to (i.e. their Rings value). You can do this by
    using plt.scatter(x_values,y_values,c=category_labels.astype(str)), where
    category_labels is a Pandas dataframe containing the Rings values for the
    reduced dataset that you're plotting.
    """
    raw_data = dp.load_dataset(dataset)
    data = dp.preprocess_dataset(raw_data)
    
    train, test = dp.split_train_and_test(data, test_frac = sampled_frac)
    
    if algorithm == 'MDS':
        mds = MDS(n_components = 2)
        embed = mds.fit_transform(test.drop('Rings', axis = 1).drop('Sex', axis = 1))
        plt.scatter(embed[:,0], embed[:,1], c = test['Rings'])
        plt.title('MDS')
        plt.show()

    elif algorithm == 'TSNE':
        tsne = TSNE(n_components = 2, learning_rate = 30.0)
        model = tsne.fit_transform(test.drop('Rings', axis = 1).drop('Sex', axis = 1))
        plt.scatter(model[:,0], model[:,1], c = test['Rings'])
        plt.title('TSNE')
        plt.show()

    elif algorithm == 'isomap':
        iso =  Isomap(n_components = 2)
        model = iso.fit_transform(test.drop('Rings', axis = 1).drop('Sex', axis = 1))
        plt.scatter(model[:,0], model[:,1], c = test['Rings'])
        plt.title('Isomap')
        plt.show()

    else:
        print('try different algorithm parameter')


def build_model(train_inputs,train_labels,model_params,model_mode='classification',
                    model_type='naive_bayes'):
    """Uses the training set to build a machine learning model with the parameters
    specified by the keywords model_params, model_mode and model_type. This function
    should allow the user to train a classifier using K nearest neighbors, a support
    vector machine, a decision tree or a naive Bayes algorithm, or a regressor
    using K nearest neighbors, a support vector machine or a decision tree (it's
    not possible to use naive Bayes for regression, so it can only be deployed
    for classification). For information on each algorithm, see the Python
    curriculum.

    params:

    :train_inputs: a Pandas dataframe obtained by passing appropriately
    preprocessed training data to the split_inputs_and_labels() function in the
    data_preprocessing.py module. Columns represent the features taken as
    input by our learning models.

    :train_labels: a Pandas dataframe likewise obtained from
    split_inputs_and_labels(), corresponding to the training set's Rings values.

    :model_params: a dictionary object containing the parameters to be used to
    train the model (e.g. for a KNeighborsClassifier, we might have
    model_params = {'n_neighbors': 5, 'leaf_size': 30, 'p': 2}).

    :model_mode: either 'classification' or 'regression'. Specifies whether the
    problem should be treated as a classification or regression problem.

    :model_type: 'naive_bayes', 'knn', 'svm' or 'decision_tree'. Indicates which
    model architecture is to be trained.

    returns: the trained model.
    """


    

    # Classification models
    if model_mode == 'classification':

        # SVM
        if model_type == 'svm':
            model = SVC(**model_params)
            return(model.fit(train_inputs, train_labels))

        # KNN
        elif model_type == 'knn':
            model = KNeighborsClassifier(**model_params)
            return(model.fit(train_inputs, train_labels))

        # Decision Tree
        elif model_type == 'decision_tree':
            model = DecisionTreeClassifier(**model_params)
            return(model.fit(train_inputs, train_labels))

        # Naive Bayes
        else:
            model = GaussianNB()
            return(model.fit(train_inputs, train_labels))

    # Regression models
    if model_mode == 'regression':

        # SVM
        if model_type == 'svm':
            model = SVR(**model_params)
            return(model.fit(train_inputs, train_labels))

        # KNN
        elif model_type == 'knn':
            model = KNeighborsRegressor(**model_params)
            return(model.fit(train_inputs, train_labels))
            
        # Decision Tree
        elif model_type == 'decision_tree':
            model = DecisionTreeRegressor(**model_params)
            return(model.fit(train_inputs, train_labels))
            
        else:
            return('must choose "svm", "knn", or "decision_tree" for model_tye')



    pass

def evaluate_model(model,test_inputs,test_labels,model_mode,rings):
    """Evaluates the model passed as input using a test set. This function
    must: 1) print the model accuracy to the terminal (if in 'classification'
    mode) or print the mean standard error to the terminal (if in 'regression' mode)
    and 2) display (but not necessarily save) a plot of the confusion
    matrix obtained from the test set
    (see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html).

    params:

    :model: the model to be evaluated.

    :test_inputs: a Pandas dataframe obtained by passing appropriately
    preprocessed testing data to the split_inputs_and_labels() function in the
    data_preprocessing.py module. Columns represent the features taken as
    input by our model.

    :test_labels: a Pandas dataframe likewise obtained from
    split_inputs_and_labels(), corresponding to the testing set's Rings values.

    :model_mode: either 'classification' or 'regression'. Specifies whether the
    problem should be treated as a classification or regression problem.

    returns: nothing.
    """

    if model_mode == 'classification':

        # Accuracy of prediction
        predictions = model.predict(test_inputs)
        accuracy = accuracy_score(test_labels, predictions)

        # Confusion Matrix
        matrix = confusion_matrix(test_labels, predictions)
        tick_labels = []
        for i in range(1, 30):
            tick_labels.append(str(i))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix)
        fig.colorbar(cax)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        plt.xticks(np.arange(0,len(tick_labels), 1.0))
        plt.yticks(np.arange(0,len(tick_labels), 1.0))
        
        # Evaluations
        if rings == "lower":
            plt.title("Rings [0, 9]")
            plt.show()
            print(accuracy)
        elif rings == "greater":
            plt.title("Rings [10, 29]")
            plt.show()
            print(accuracy)
        else:
            return("rings parameter must be 'lower' or 'greater'")

    elif model_mode == 'regression':

        MSE = np.mean((test_labels - model.predict(test_inputs))**2)
        predictions = model.predict(test_inputs)
        number = len(test_labels)
       
        
        # Confusion Matrix
        matrix = confusion_matrix(test_labels, np.round(predictions))
        tick_labels = []
        for i in range(1, 30):
            tick_labels.append(str(i))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix)
        fig.colorbar(cax)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        plt.xticks(np.arange(0,len(tick_labels), 1.0))
        plt.yticks(np.arange(0,len(tick_labels), 1.0))
        
        # Evaluations
        if rings == "lower":
            plt.title("Rings [0, 9]")
        elif rings == "mid":
            plt.title("Rings [10, 16]")
        elif rings == "greater":
            plt.title("Rings [17, 29]")
        else:
            return("rings parameter must be 'lower' or 'greater'")
        plt.show()
        print("MSE = " + str(MSE) + " count = " + str(number))
        return(MSE, number)

    else:
        return('Choose model_mode as "classification" or "regression"')

