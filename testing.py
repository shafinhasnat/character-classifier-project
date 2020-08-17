import cv2
import numpy as np
import joblib

## this code is for testing our classifier model
def predict(img, model):
	## read our test image in grayscale
	im = cv2.imread(img, 0)
	## invert the color as before
	im = cv2.bitwise_not(im)
	## resize them in 28x28 resulation
	im = cv2.resize(im, (28, 28))
	## flatten from 2D array to 1D
	im = im.flatten()
	## read the previously saved classifier models
	loadModel = joblib.load(model)
	## classify our image against the trained model
	predict = loadModel.predict([im])
	## return what letter it might be
	return predict[0]

if __name__ == "__main__":
	## we saved our test image in img.png name
	img = "img.png"
	## loading models both knn and Nb
	knnModel = "trainingKnnModel.sav"
	nbModel = "trainingNbModel.sav"
	## pass the parameters in the function above for Knn model
	print("Prediction from knn model: {}".format(predict(img, knnModel)))
	## pass the parameters in the function above for Nb model
	print("Prediction from Nb model: {}".format(predict(img, nbModel)))
