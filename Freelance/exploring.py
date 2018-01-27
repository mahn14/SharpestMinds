import preprocessing as pp
import pandas as pd 
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from scipy import stats
from minepy import MINE
from sklearn.manifold import MDS, Isomap, TSNE
import math



# Pearon Linear Correlation
def pearson_corr(data, label):
	'''
		@param
			data : Pandas DF in which columns will be compared to the label
			label : Pandas DF which only one feature
		@return
			prints pearson correlation score for each feature in data vs label 
	'''
	for feature in data.columns:
		x = stats.pearsonr(data[feature], label)
		print(str(x) + " : " + str(feature))

# Kendal Tau Nonlinear Feature Correlation
def kendalltau_corr(data1, data2, tol, overlap = "none"):
	if type(data1) != type(pd.DataFrame()):
		print("data1 must be type Pandas DataFrame")
		return("")
	if type(data2) != type(pd.DataFrame()):
		print("data2 must be type Pandas DataFrame")
		return("")

	if overlap == "none":
		for i in range(len(data1.columns)):
			for j in range(len(data2.columns)):
				feature1 = str(data1.columns[i])
				feature2 = str(data2.columns[j])
				tau, p = stats.kendalltau(data1[feature1], data2[feature2])
				if tau > tol:
					print("(" + str(tau) + "," + str(p) + ")" + " : " + "(" + str(feature1) + "," + str(feature2) + ")")

	elif overlap == "all":
		for i in range(len(data1.columns)):
			for j in range(len(data2.columns)):
				if i < j:
					feature1 = str(data1.columns[i])
					feature2 = str(data2.columns[j])
					tau, p = stats.kendalltau(data1[feature1], data2[feature2])
					if tau > tol:
						print("(" + str(tau) + "," + str(p) + ")" + " : " + "(" + str(feature1) + "," + str(feature2) + ")")	
	else:
		print("Split overlapping features into separate columns for efficiency")
		kendalltau_corr(data1, data2, tol, overlap = "none")



# Maximal Information Test [0, 1]
def maximal_test(data, label):
	m = MINE()
	for j in label.columns:
		for feature in data.columns:
			m.compute_score(data[feature], label[j])
			print(str(m.mic()) + " : " + str(feature) + " vs " + str(j))


# Scatter Plot
def scatter_df_label(data, label):
	label = pd.DataFrame(label)
	for feature in data.columns:
		plt.scatter(data[feature], label)
		plt.ylabel(str(label.columns[0]))
		plt.xlabel(str(feature))
		plt.show()

	x = len(data.columns)
	coord = []
	for i in range(2):
		coord.append(math.ceil(x/2))

	f, coord = plt.subplots(math.ceil(x/2), 2, sharey = True)

	for i in range(2):
		for j in range(math.ceil(x/2)):
			coord[i][j].scatter(data.iloc[:,i], label.iloc[:,0])
			coord[i][j].set_title(str(x_data.columns[i]) + " vs " + str(label.columns[0]))

	plt.show()


def scatter_x_y(x_data, y_data):
	x = len(x_data.columns)
	y = len(y_data.columns)
	coord = []
	for i in range(x):
		coord.append(range(y))

	f, coord = plt.subplots(x, y, sharey = True)
	
	for i in range(x):
		for j in range(y):
			coord[i][j].scatter(x_data.iloc[:,i], y_data.iloc[:,j])
			coord[i][j].set_title(str(x_data.columns[i]) + " vs " + str(y_data.columns[j]))

	plt.show()


def correlation_matrix(scaled_data):
	alpha = scaled_data.columns
	correlations_matrix = scaled_data.corr()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(correlations_matrix)
	#add a colorbar legend
	fig.colorbar(cax)
	#rotate the x-axis labels so they don't overlap
	ax.set_xticklabels(alpha,rotation=90)
	ax.set_yticklabels(alpha)
	#we have to force the plot to show us all labels, because it won't by default
	plt.xticks(np.arange(0,len(alpha), 1.0))
	plt.yticks(np.arange(0,len(alpha), 1.0))
	plt.show()


# Histogram
def hist(data):
	hist = data.hist(bins = 50)
	plt.show()

def dimensionality_reduction(data, test, algorithm, frac = 0.05):
    
    rand_data = data.sample(frac = 1)
    scaled_test_frac = int(frac * data.shape[0])

    data = rand_data.loc[:scaled_test_frac]

    if algorithm == 'MDS':
        mds = MDS(n_components = 4)
        embed = mds.fit_transform(data.drop(test, axis = 1))
        plt.scatter(embed[:,0], embed[:,1], c = data[test])
        plt.title('MDS')
        plt.show()

    elif algorithm == 'TSNE':
        tsne = TSNE(n_components = 3, learning_rate = 20)
        model = tsne.fit_transform(data.drop(test, axis = 1))
        plt.scatter(model[:,0], model[:,1], c = data[test])
        plt.title('TSNE')
        plt.show()

    elif algorithm == 'isomap':
        iso =  Isomap(n_components = 3, n_neighbors = 10)
        model = iso.fit_transform(data.drop(test, axis = 1))
        plt.scatter(model[:,0], model[:,1], c = data[test])
        plt.title('Isomap')
        plt.show()

    else:
        print('try different algorithm parameter')
















