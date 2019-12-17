import tensorflow as tf
from layerClass import Layer
from dataClass import Data
from trainClass import train
from pathlib import Path
import numpy as np
from freezeGraphClass import freezeGraph

class Model(Layer,):

    def __init__(self, inputPlaceholder, outputPlaceHolder,numberLayers,numberofFClayers,filterSize,inputFilters, outputFilters,learningRate, stride):
        print ("Initialisation")
        self.initializer = "Xavier"
        Layer.__init__(self,numberLayers,numberofFClayers,filterSize,inputFilters, outputFilters,self.initializer)
        self.input = inputPlaceholder
        self.output = outputPlaceHolder
        self.learningRate = learningRate
        self._prediction = None
        self._optimize = None
        self._loss = None
        self.stride = stride

    def prediction(self,inputShape, outputShape):
        print ("Prediction")
        if not self._prediction:
            self.weight, self.bias = Layer.weights(self)
            self.fcweight, self.fcbias = Layer.fcweights(self,inputShape, outputShape)
            self.cnnOutput =  Layer.createCNNNetwork(self,self.input,self.stride)
            self.fcOutput = Layer.createFCNetwork(self,self.cnnOutput)
            print (self.fcOutput.shape)
            self._prediction = Layer.Output(self, self.fcOutput)

        return self._prediction

    def error(self):
        with tf.name_scope('loss'):
            if not self._loss:
                #self.delta_com = tf.subtract(self.output, self._prediction)
                #self.norm_com = tf.norm(self.delta_com, axis=1)
                #self._loss = tf.reduce_mean(self.norm_com)
                self._loss = tf.losses.mean_squared_error(labels = self.output,predictions = self._prediction)
            return self._loss


    def optimize(self):
        with tf.name_scope('optimiser'):
            if not self._optimize:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
                self._optimize = optimizer.minimize(self._loss)
            return self._optimize


def calculateOutputDimensionOfCNN(width,height, stride, padding, filterSize):
    return int(((width-filterSize+2*padding)/stride) + 1), int(((height-filterSize+2*padding)/stride) + 1)

def populateInputShapeList(width,height,numberofCNNLayers, numberofFClayers, numberofFilters,stride,padding,filterSize):
    inputShapeList = []
    for i in range((numberofCNNLayers)):
        outputDimensionWidth, outputDimensionHeight = calculateOutputDimensionOfCNN(width,height,stride,padding,filterSize)
        width = outputDimensionWidth
        height = outputDimensionHeight
    firstLayershape = width*height*numberofFilters
    inputShapeList.append(firstLayershape)
    return inputShapeList

def getnumberofBatches(Datasize, batchSize):
    return int(Datasize/batchSize)

filePath = Path("C:/Users/soumi/Documents/Labelling/newTrain/").glob('*.jpg')
jsonPath = Path("C:/Users/soumi/Documents/Labelling/newTrain/trainData.txt")
savePath = "C:/Users/soumi/Documents/Labelling/ml/model/"
stride = 2
numberofCNNLayers = 3
numberofFClayers = 3
filterSize = 3
imageWidth = 320
imageHeight = 240
padding = 1
numberofFilters = 16
neurons = 2048
finalLayerNeurons = 4
trainbatchSize = 8
valbatchSize = 8
epochs = 300
learningRate = 0.0001
inputShapeList = populateInputShapeList(imageWidth,imageHeight,numberofCNNLayers, numberofFClayers, numberofFilters,stride,padding,filterSize)
outputTensorName =  "Inference/Output"
inputFiltersList = list()
outputFiltersList = list()
outputShapeList = list()
inputFiltersList.append(1)
outputFiltersList.append(numberofFilters)

for i in range((numberofCNNLayers - 1)) :
    inputFiltersList.append(numberofFilters)
    outputFiltersList.append(numberofFilters)
for i in range((numberofFClayers - 1)):
    inputShapeList.append(neurons)
    outputShapeList.append(neurons)
    neurons = int(neurons/2)

outputShapeList.append(finalLayerNeurons)
data = Data(filePath, jsonPath)
data.jsonData()
data.loadLabels()
oneHotEncoded = data.convertToOneHotEncoding()
print (oneHotEncoded.shape)

print ("Input shape is",inputShapeList)
print ("Output shape is", outputShapeList)

with tf.Graph().as_default() as graph:
    with tf.Session() as sess:
        tf.constant([imageHeight,imageWidth], dtype="float32",name = "imageSize")
        tf.constant([outputTensorName], name = "OutputTensorName")
        X = tf.placeholder(tf.float32, [None, imageHeight,imageWidth,1], name='Input')
        Y = tf.placeholder(tf.float32, [None, finalLayerNeurons])
        m = Model(X,Y,numberofCNNLayers,numberofFClayers,filterSize,inputFiltersList,outputFiltersList, learningRate, stride)
        prediction = m.prediction(inputShapeList, outputShapeList)
        error = m.error()
        data.add_variable_summary(error, "Loss")
        optimizer = m.optimize()
        train_dataset, val_dataset, test_dataset = data.createTensorflowDatasets(0.8,0.1,0.1)
        merged_summary_operation = tf.summary.merge_all()
        modelname = "model_" + "learningRate_" + str(m.learningRate) + "filtersize_" + str(filterSize) + "epochs_" + str(epochs) 
        print (modelname) 
        train_summary_writer = tf.summary.FileWriter(savePath + '/tmp/' + modelname + "train")
        validation_summary_writer = tf.summary.FileWriter(savePath + '/tmp/' + modelname+ "validation")
        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_l)
        print("Initialisation completed")
        trainingClass = train(sess,data,optimizer,error,m,merged_summary_operation)
        for epoch in range(epochs):
            print ("Current epoch is",epoch)
            batches = getnumberofBatches(data.train_size, trainbatchSize)
            trainingError = trainingClass.run(epoch, train_dataset,data.train_size, trainbatchSize,batches,train_summary_writer)
            print ("Training error is ", trainingError)
            batches = getnumberofBatches(data.val_size, valbatchSize)
            #validationError = trainingClass.run(epoch,val_dataset,data.val_size, valbatchSize ,batches,validation_summary_writer)
            validationError = trainingClass.validation(epoch,val_dataset,data.val_size, valbatchSize ,batches,validation_summary_writer)
            print ("Validation Error is ", validationError)
        saver = tf.train.Saver()
        save_path = saver.save(sess, savePath + modelname + "/" + "model.ckpt")
        print("Model saved in path: %s" % save_path)
freezeGraph = freezeGraph(savePath + modelname + "/", outputTensorName)
freezeGraph.freeze_graph()            

            
    