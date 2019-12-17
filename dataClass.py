import cv2
import numpy as np
import json
import imutils
import glob
import os
from collections import defaultdict
import tensorflow as tf
from pathlib import Path

class Data(object):

	def __init__(self,filePath,jsonPath):
		self.filePath = filePath
		self.jsonPath = jsonPath
		self.trainXLabels = list()
		self.trainYLabels = list()
		self.trainWidthLabels = list()
		self.trainHeightLabels = list()
		self.trainPath = list()
		self.Dictdata = defaultdict(dict)
		self.oneHotEncodedData = None
		self.train_dataset = None
		self.test_dataset = None
		self.val_dataset = None
		self.train_size = None
		self.val_size = None
		self.test_size = None

	def jsonData(self):
		with open(str(self.jsonPath), 'r') as f:
			self.Dictdata =  json.loads(f.read())

	def loadLabels(self):
		files = [file for file in self.filePath]
		for file in files:
			filename, file_extension = os.path.splitext(str(file))
			name = str(os.path.basename(filename));
			my_path = file.absolute().as_posix()
			if (name + ".jpg") in self.Dictdata:
				if "pos" in name and "Bazen" not in name:
					self.trainPath.append(str(my_path))
					boundaryPoints = self.Dictdata[name + ".jpg"]['BoundaryPoints']
					newBoundaryPoints = []
					for item in boundaryPoints:
						newBoundaryPoints.append((int(item[0]*320),int(item[1]*240)))
					x,y,w,h = cv2.boundingRect(np.array(newBoundaryPoints))
					self.trainXLabels.append(round(((x - 5))/320,2))
					self.trainYLabels.append(round(((y - 5))/240,2))
					self.trainWidthLabels.append(round(((x+w + 5))/320,2))
					self.trainHeightLabels.append(round(((y+h + 5))/240,2))
		print (np.array(self.trainXLabels).shape)
		self.trainXLabels = np.array(self.trainXLabels)
		self.trainYLabels = np.array(self.trainYLabels)
		self.trainWidthLabels = np.array(self.trainWidthLabels)
		self.trainHeightLabels = np.array(self.trainHeightLabels)

	def convertToOneHotEncoding(self):
		self.oneHotEncodedData = np.zeros((self.trainXLabels.shape[0], 4))
		for item in range(0, self.trainXLabels.shape[0]):
			self.oneHotEncodedData [item][0] = (self.trainXLabels[item])
			self.oneHotEncodedData [item][1] = (self.trainYLabels[item])
			self.oneHotEncodedData [item][2] = (self.trainWidthLabels[item])
			self.oneHotEncodedData [item][3] = (self.trainHeightLabels[item])
		return self.oneHotEncodedData

	def createTensorflowDatasets(self,trainSize, validationSize, testSize):
		PathDataset = tf.data.Dataset.from_tensor_slices(self.trainPath)
		LabelsDataset = tf.data.Dataset.from_tensor_slices(self.oneHotEncodedData)
		fullDataset = tf.data.Dataset.zip((PathDataset, LabelsDataset))
		self.train_size = int(trainSize*self.oneHotEncodedData.shape[0])
		self.val_size = int(validationSize*self.oneHotEncodedData.shape[0])
		self.test_size  = int(testSize*self.oneHotEncodedData.shape[0])
		fullDataset = fullDataset.shuffle(self.oneHotEncodedData.shape[0])
		self.train_dataset = fullDataset.take(self.train_size)
		test_dataset = fullDataset.skip(self.train_size)
		self.val_dataset = test_dataset.skip(self.val_size)
		self.test_dataset = test_dataset.take(self.test_size)
		return self.train_dataset, self.val_dataset, self.test_dataset

	def createDatasetIterator(self,dataset, datasetSize, batchSize):
		dataset = dataset.shuffle(datasetSize).batch(batchSize)
		datasetIterator = dataset.make_initializable_iterator()
		return datasetIterator

	def getBatchData(self, batch):
		finalData = list()
		for image in (batch[0]):
			image_reader = cv2.imread(image.decode("utf-8"),0)
			normalised_image = image_reader.astype(np.float)/255.0
			normalised_image = np.expand_dims(normalised_image, axis=2)
			finalData.append(normalised_image)
		return finalData

	def getBatchLabels(self,batch):
		finalData = list()
		for label in (batch[1]):
			finalData.append(label)
		return finalData

	def add_variable_summary(self,tf_variable, summary_name):
		with tf.name_scope(summary_name + '_summary'):
			mean = tf.reduce_mean(tf_variable)
			tf.summary.scalar('Mean',mean)
			with tf.name_scope('standard_deviation'):
				standard_deviation = tf.sqrt(tf.reduce_mean(tf.square(tf_variable - mean)))
			tf.summary.scalar('StandardDeviation',standard_deviation)
			tf.summary.scalar('Maximum', tf.reduce_max(tf_variable))
			tf.summary.scalar('Minimum',tf.reduce_min(tf_variable))
			tf.summary.histogram('Histogram',tf_variable)



