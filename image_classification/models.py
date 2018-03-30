import tensorflow as tf


class CNN(object):
    """docstring for Model"""
    def __init__(self, image_shape, n_classes):
        self.image_shape = image_shape
        self.n_classes = n_classes

        self.inputs = self.neural_net_image_input(image_shape)
        self.label = self.neural_net_label_input(n_classes)
        self.keep_prob = self.neural_net_keep_prob_input()
    def conv_net(self, x, keep_prob):
        """
        Create a convolutional neural network model
        : x: Placeholder tensor that holds image data.
        : keep_prob: Placeholder tensor that hold dropout keep probability.
        : return: Tensor that represents logits
        """
        # Apply 1, 2, or 3 Convolution and Max Pool layers
        #    Play around with different number of outputs, kernel size and stride
        # Function Definition from Above:
        #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
        x = conv2d_maxpool(x, 32, [2, 2], [1, 1], [2, 2], [2, 2])
        
        x = conv2d_maxpool(x, 64, [3, 3], [1, 1], [2, 2], [2, 2])
        x = tf.nn.dropout(x, keep_prob)

        # Apply a Flatten Layer
        # Function Definition from Above:
        #   flatten(x_tensor)
        x = flatten(x)

        # Apply 1, 2, or 3 Fully Connected Layers
        #    Play around with different number of outputs
        # Function Definition from Above:
        #   fully_conn(x_tensor, num_outputs)
        x = fully_conn(x, 1024)
        x = fully_conn(x, 512)
        
        # Apply an Output Layer
        #    Set this to the number of classes
        # Function Definition from Above:
        #   output(x_tensor, num_outputs)
        x = output(x)
        
        # return output
        return x


    def neural_net_image_input(self, image_shape):
        """
        Return a Tensor for a batch of image input
        : image_shape: Shape of the images
        : return: Tensor for image input.
        """
        return tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name='x')


    def neural_net_label_input(self, n_classes):
        """
        Return a Tensor for a batch of label input
        : n_classes: Number of classes
        : return: Tensor for label input.
        """
        return tf.placeholder(tf.float32, [None, n_classes], name='y')


    def neural_net_keep_prob_input(self):
        """
        Return a Tensor for keep probability
        : return: Tensor for keep probability.
        """
        return tf.placeholder(tf.float32, name='keep_prob')

    def conv2d_maxpool(self, x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
        """
        Apply convolution then max pooling to x_tensor
        :param x_tensor: TensorFlow Tensor
        :param conv_num_outputs: Number of outputs for the convolutional layer
        :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
        :param conv_strides: Stride 2-D Tuple for convolution
        :param pool_ksize: kernal size 2-D Tuple for pool
        :param pool_strides: Stride 2-D Tuple for pool
        : return: A tensor that represents convolution and max pooling of x_tensor
        """
        shape = x_tensor.get_shape().as_list()
        weight = tf.Variable(tf.truncated_normal((conv_ksize[0], conv_ksize[1], shape[3], conv_num_outputs)))
        bias = tf.Variable(tf.random_normal([conv_num_outputs]))
        strides = [1, conv_strides[0], conv_strides[1], 1]
        conv_layer = tf.nn.conv2d(x_tensor, weight, strides, padding='VALID')
        conv_layer = tf.nn.bias_add(conv_layer, bias)
        conv_layer = tf.nn.relu(conv_layer)
        ksize = [1, pool_ksize[0], pool_ksize[1], 1]
        strides_pool = [1, pool_strides[0], pool_strides[1], 1]
        conv_layer = tf.nn.max_pool(conv_layer, ksize, strides_pool, padding='VALID')
        return conv_layer

    def fully_conn(self, x_tensor, num_outputs):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        x_tensor = tf.contrib.layers.fully_connected(x_tensor, num_outputs)
        return x_tensor

    def output(self, x_tensor):
        """
        Apply a output layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        x_tensor = tf.layers.dense(x_tensor, self.n_classes)
        return x_tensor 