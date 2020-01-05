import tensorflow as tf

class Layer:

    def __init__(self,initializer):
        self.weightsDict = {}
        self.biasDict = {}
        if (initializer == "Xavier"):
            self.tf_initializer = tf.contrib.layers.xavier_initializer()
        elif (initializer == "Normal"):
            self.tf_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        elif (initializer == "He"):
            self.tf_initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)

    
    def conv2d(self,inputFeature,filterSize, inputSize, outputSize, name,strides = 1):
        filter_shape = [filterSize, filterSize, inputSize, outputSize]
        self.weightName = name + "weight"
        self.biasName = name + "bias"
        with tf.variable_scope("variable", reuse=tf.AUTO_REUSE):
            self.weightsDict[self.weightName] = tf.get_variable(self.weightName, shape=filter_shape,initializer=self.tf_initializer)                
            self.biasDict[self.biasName] = tf.get_variable(self.biasName, shape = outputSize, initializer=self.tf_initializer)                
        convOutput = tf.nn.conv2d(input = inputFeature, filter = self.weightsDict[self.weightName], strides=[1, strides, strides, 1], padding='SAME', name = name)
        finalOutput = tf.nn.bias_add(convOutput, self.biasDict[self.biasName])
        return finalOutput
     
    def avgpool2d(self,inputData):
        return tf.nn.avg_pool(value = inputData, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

    def downSamplingBlock(self,x,input_channels,output_channels,down_size,name):
        print (x.shape)
        conv1 =  self.conv2d(x,3,input_channels,output_channels,name+"conv1",strides = down_size)
        print (conv1.shape)
        batchNorm1 = tf.layers.batch_normalization(conv1)
        x1 = tf.nn.leaky_relu(batchNorm1)
        return x1

    def upSamplingBlock(self,currentInput,previousInput,input_channels,output_channels,image_width, image_height,name):
        print ("Upsampling")
        x = tf.image.resize_images(images=currentInput,size=[image_width,image_height], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,align_corners=False,preserve_aspect_ratio=False)
        print (x.shape)
        conv11 = self.conv2d(x, 1,input_channels,output_channels,name + "conv11First")
        x_concat = tf.concat([conv11,previousInput],axis=3)
        print (x_concat.shape)
        conv11 = self.conv2d(x_concat, 1,input_channels,output_channels,name + "conv11")
        batchNorm1 = tf.layers.batch_normalization(conv11)
        x1 = tf.nn.leaky_relu(batchNorm1)
        return x1

    def runBlock(self,inputData,in_channels=1,out_channels=5,channel_size=32):
        self.x1 = self.downSamplingBlock(inputData,input_channels=in_channels,output_channels=0.5*channel_size, down_size=1,name = "DownBlock1")
        self.x2 = self.downSamplingBlock(self.x1,input_channels=0.5*channel_size,output_channels=channel_size, down_size=2,name = "DownBlock2")
        self.x3 = self.downSamplingBlock(self.x2,input_channels=channel_size,output_channels=2*channel_size, down_size=2,name = "DownBlock3")
        self.x4 = self.downSamplingBlock(self.x3, input_channels=2*channel_size,output_channels=4*channel_size, down_size=2,name = "DownBlock4")
        self.x5 = self.downSamplingBlock(self.x4,input_channels=4*channel_size,output_channels=8*channel_size, down_size=2,name = "DownBlock5")
        self.x6 = self.upSamplingBlock(self.x5, self.x4,input_channels=8*channel_size, output_channels=4*channel_size,image_width = 30,image_height = 40,name = "UpBlock1")
        self.x7 = self.upSamplingBlock(self.x6, self.x3,input_channels=4*channel_size, output_channels=2*channel_size,image_width = 60,image_height = 80,name = "UpBlock2")
        self.x8 = self.upSamplingBlock(self.x7, self.x2,input_channels=2*channel_size, output_channels=channel_size,image_width = 120,image_height = 160, name = "UpBlock3")
        self.x9 = self.upSamplingBlock(self.x8, self.x1,input_channels=channel_size, output_channels=0.5*channel_size,image_width = 240,image_height = 320, name = "UpBlock4")
        self.out_conv1 = self.conv2d(self.x9,1,0.5*channel_size,out_channels,name = "Inference/Output")
        return self.out_conv1
        

            

        







            

	