import glob
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier ## this is for knn classifier
from sklearn.naive_bayes import GaussianNB ## this is for NB classifier
from sklearn.model_selection import train_test_split ## this is for splitting data to measure accuracy
from sklearn import metrics ## this is for measuring accuracy
import joblib

## open the label.csv file and return the label data as an array
def labelRead():
	filename = "label.csv"
	label = []
	with open(filename, 'r') as f:
		csvRead = csv.reader(f)
		for i in csvRead:
			label.append(i[0])
	label = np.array(label)
	return label


## open the trainingImg.csv file and return the label data as an array
def trainingRead():
	trainImg = []
	filename = "trainingImg.csv"
	with open(filename, 'r') as trainingFile:
		read = csv.reader(trainingFile)
		for i in read:
			row = list(map(int, i))
			trainImg.append(row)
	trainImg = np.array(trainImg)
	return trainImg

def trainModel(img, label, classifier, filename):
	## taking 20% of all training data and label for testing the model accuracy 
	x_train, x_test, y_train, y_test = train_test_split(img, label, test_size = 0.2, random_state = 1)
	## fitting the image and label into the respective classifier
	classifier.fit(img, label)
	print("Training model done!")
	print("Model saving initialized")
	## saving the classifier model
	joblib.dump(classifier, filename)
	print("Model saving done!")
	## predicting test images for the accuracy test
	y_pred = classifier.predict(x_test)
	## print the model accuracy
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
	pass

def trainKnnModel(img, label):
	print("~~~~~~~~~~~~~~~~KNN~~~~~~~~~~~~~~~")
	print("training model initialized")
	## declearing the classifier to be KNN classifier using scikit learn
	classifier = KNeighborsClassifier(n_neighbors = 10)
	filename = 'trainingKnnModel.sav'
	## passing arguments for knn classifier in the trainModel() function and save it with this filename
	trainModel(img, label, classifier, filename)
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	pass

def trainNbModel(img, label):
	print("~~~~~~~~~~~~~~~~Nb~~~~~~~~~~~~~~~")
	print("training model initialized")
	## declearing the classifier to be Naive bayes classifier using scikit learn
	classifier = GaussianNB()
	filename = 'trainingNbModel.sav'
	## passing arguments for Nb classifier in the trainModel() function and save it with this filename
	trainModel(img, label, classifier, filename)
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	pass
	
if __name__ == '__main__':
	## calling the trainingRead() and labelRead() function
	img = trainingRead()
	label = labelRead()
	## passing the arguments into the both models
	trainKnnModel(img, label)
	trainNbModel(img, label)
	pass