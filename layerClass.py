import tensorflow as tf

class Layer:

    def __init__(self,numberofCNNlayers,numberofFClayers, filterSize,inputFilters, outputFilters,initializer):
        self._numCNNLayers = numberofCNNlayers
        self._numFCLayers = numberofFClayers
        self.filterSize = filterSize
        self.inputFilters = inputFilters
        self.outputFilters = outputFilters
        self.weightsDict = {}
        self.biasDict = {}
        self.weightName = 'wc'
        self.fcName = 'fc'
        self.fcbiasName = 'bfc'
        self.biasName = 'bc'
        if (initializer == "Xavier"):
            self.tf_initializer = tf.contrib.layers.xavier_initializer()
        elif (initializer == "Normal"):
            self.tf_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        elif (initializer == "He"):
            self.tf_initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    
    def weights(self):
        for i in range(self._numCNNLayers):
            self.weightsDict[self.weightName + str(i)] = tf.get_variable(self.weightName + str(i), shape=[self.filterSize, self.filterSize, self.inputFilters[i], self.outputFilters[i]],initializer=self.tf_initializer)                
            self.biasDict[self.biasName + str(i)] = tf.get_variable(self.biasName + str(i), shape=[self.outputFilters[i]],initializer=self.tf_initializer)                
        return self.weightsDict, self.biasDict

    def fcweights(self, inputshape, outputshape):
        for i in range(self._numFCLayers):
            self.weightsDict[self.fcName + str(i)] = tf.get_variable(self.fcName + str(i), shape=[inputshape[i], outputshape[i]],initializer = self.tf_initializer)                
            self.biasDict[self.fcbiasName + str(i)] = tf.get_variable(self.fcbiasName + str(i), shape=[outputshape[i]],initializer=self.tf_initializer)                
        return self.weightsDict, self.biasDict

    
    def conv2d(self,x, W, b, name,strides = 1):
        x = tf.nn.conv2d(input = x, filter = W, strides=[1, strides, strides, 1], padding='SAME', name = name)
        x = tf.nn.bias_add(x, b)
        x = tf.nn.leaky_relu(x)
        return tf.layers.batch_normalization(x) 
     
    def avgpool2d(self,x,name,k):
        return tf.nn.avg_pool(value = x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME', name = name)
    
    def createCNNNetwork(self,x,stride):
        inputdata = x
        for i in range(self._numCNNLayers):
            with tf.name_scope("CNNLayers"):
                name = "Convolution" + str(i)
                namePool = "Pooling" + str(i)
                conv1 = self.conv2d(inputdata, self.weightsDict[self.weightName + str(i)], self.biasDict[self.biasName + str(i)],name)
                print (conv1.shape)
                conv1pool = self.avgpool2d(conv1,namePool,stride)
                inputdata = conv1pool
        return  tf.contrib.layers.flatten(conv1pool)

    def createFCNetwork(self,x):
        inputdata = x
        for i in range(self._numFCLayers-1):
            with tf.name_scope("FCLayers"):
                name = "FullyConnected" + str(i)
                fc1 = tf.nn.bias_add(tf.matmul(inputdata, self.weightsDict[self.fcName + str(i)], name = name),self.biasDict[self.fcbiasName + str(i)])
                fc1Activation = tf.nn.leaky_relu(fc1,name ='finalrelu')
                inputdata = fc1Activation
        return fc1Activation

    def Output(self, x):
        with tf.name_scope("Inference"):
            logits = tf.nn.bias_add(tf.matmul(x,  self.weightsDict[self.fcName + str(self._numFCLayers - 1)]), self.biasDict[self.fcbiasName + str(self._numFCLayers - 1)], name = "Output")
        return logits





            

	