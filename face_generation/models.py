import tensorflow as tf

class GANs(object):
	"""docstring for GANs"""
	def __init__(self, image_width, image_height, image_channels, z_dim, out_channel_dim, learning_rate, beta1):
		"""
		:param image_width: The input image width
	    :param image_height: The input image height
	    :param image_channels: The number of image channels
	    :param z_dim: The dimension of Z
	    :param out_channel_dim: The number of channels in the output image
	  	:param learning_rate: Learning Rate Placeholder
	    :param beta1: The exponential decay rate for the 1st moment in the optimizer
		"""
		self.image_width = image_width
		self.image_height = image_height
		self.image_channels = image_channels
		self.z_dim = z_dim
		self.out_channel_dim = out_channel_dim
		self.learning_rate = learning_rate
		self.beta1 = beta1

	def model_inputs(self):
	    """
	    Create the model inputs
	    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
	    """
	    real_input = tf.placeholder(tf.float32, (None, self.image_width, self.image_height, self.image_channels))
	    input_z = tf.placeholder(tf.float32, (None, self.z_dim))
	    learning_rate = tf.placeholder(tf.float32)
	    return real_input, input_z, learning_rate

	def discriminator(self, images, reuse=False):
	    """
	    Create the discriminator network
	    :param images: Tensor of input image(s)
	    :param reuse: Boolean if the weights should be reused
	    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
	    """
	    with tf.variable_scope('discriminator', reuse=reuse):
	        # Hidden layer
	        images = tf.contrib.layers.flatten(images)
	        h1 = tf.layers.dense(images, 128, activation=None)
	        # Leaky ReLU
	        h1 = tf.maximum(0.01 * h1, h1)        
	        logits = tf.layers.dense(h1, 1, activation=None)

	        out = tf.sigmoid(logits)
	        return out, logits


	def generator(self, z, is_train=True):
	    """
	    Create the generator network
	    :param z: Input z
	    :param is_train: Boolean if generator is being used for training
	    :return: The tensor output of the generator
	    """
	    if is_train == True:
	        reuse = False
	    else:
	        reuse = True
	    with tf.variable_scope('generator', reuse=reuse):
	        # Hidden layer
	        h1 = tf.layers.dense(z, 128, activation=None)
	        # Leaky ReLU
	        h1 = tf.maximum(0.01 * h1, h1)
	        
	        # Logits and tanh output
	        logits = tf.layers.dense(h1, 784 * self.out_channel_dim, activation=None)
	        out = tf.tanh(logits)
	        out = tf.reshape(out, [-1, 28, 28, self.out_channel_dim])
	        return out

	def model_loss(self, input_real, input_z):
	    """
	    Get the loss for the discriminator and generator
	    :param input_real: Images from the real dataset
	    :param input_z: Z input
	    :param out_channel_dim: The number of channels in the output image
	    :return: A tuple of (discriminator loss, generator loss)
	    """
	    g_model = generator(input_z, self.out_channel_dim)
	    
	    d_model_real, d_logits_real = discriminator(input_real)
	    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)
	    d_loss_real = tf.reduce_mean(
	                  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, 
	                                                          labels=tf.ones_like(d_logits_real)))
	    d_loss_fake = tf.reduce_mean(
	                  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
	                                                          labels=tf.zeros_like(d_logits_real)))
	    d_loss = d_loss_real + d_loss_fake

	    g_loss = tf.reduce_mean(
	                 tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
	                                                     labels=tf.ones_like(d_logits_fake)))
	    return d_loss, g_loss

	def model_opt(d_loss, g_loss):
	    """
	    Get optimization operations
	    :param d_loss: Discriminator loss Tensor
	    :param g_loss: Generator loss Tensor
	    :return: A tuple of (discriminator training operation, generator training operation)
	    """
	    t_vars = tf.trainable_variables()
	    g_vars = [var for var in t_vars if var.name.startswith('generator')]
	    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

	    d_train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(d_loss, var_list=d_vars)
	    g_train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(g_loss, var_list=g_vars)
	    return d_train_opt, g_train_opt