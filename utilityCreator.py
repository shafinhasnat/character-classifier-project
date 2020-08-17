import os
import csv
import string
import glob ## this is for accessing folders and fetch data
import cv2 ## this is for image processing
import numpy as np

def createLabel():
	## create an empty label.csv file
	os.popen("type nul >> label.csv")
	filename = "label.csv"

	## declearing an empty list
	labelList = []

	## append A-Z (upper case letter) each 55 times in the list
	for i in string.ascii_uppercase:
		for j in range(55):
			labelList.append(i)

	## append a-z (lower case letter) each 55 times in the list
	for i in string.ascii_lowercase:
		for j in range(55):
			labelList.append(i)

	## write them into the empty label.csv file		
	with open(filename,'w') as labelFile:
		for i in labelList:
			labelFile.write(i)
			labelFile.write('\n')

def createImgTraining():
	## create an empty trainingImg.csv file
	os.popen("type nul >> trainingImg.csv")
	filename = "trainingImg.csv"
	trainingFolder = []
	imageList = []
	print("Directory reading initialized")
	## read all the directory and image file name of the dataset
	for i in range(11,63):
		n = ".\Hnd\Img\Sample0{}\*".format(i)
		trainingFolder.append(n)
	print("Directory reading done!")
	print("Image manipulation initialized")
	## processing all the images
	for i in trainingFolder:
		for file in glob.glob(i):
			## import each of the image files in grayscale (0 means grayscale)
			im = cv2.imread(file, 0)
			## invert (not operation) the color of the images black on white ----> white on black
			im = cv2.bitwise_not(im)
			## resize the image from 1200x900 to 28x28 in order to reduce load
			im = cv2.resize(im, (28, 28))
			## the images are matrices of 2D array. now flatten them and make them 1D array
			im = im.flatten()
			## put all the flatten 1D array of images in the imageList list
			imageList.append(im)
			pass
	print("Image manipulation done!!")
	print("Creating training file initialized")
	## write a image data in the empty trainingImg.csv file. from now on this will be our training data
	with open(filename, 'w', newline = '') as trainingFile:
		w = csv.writer(trainingFile, delimiter=',')
		for i in imageList:
			w.writerow(i)
	print("Training file creation done!!")

if __name__ == "__main__":
	createLabel()
	createImgTraining()