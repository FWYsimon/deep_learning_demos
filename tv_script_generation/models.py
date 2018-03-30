import tensorflow as tf

class LSTM(object):
	"""docstring for LSTM"""
	def __init__(self, batch_size, rnn_size, embed_dim, vocab_size):
		"""
		:param batch_size: Size of batches
		:param rnn_size: Size of RNNs
		:param vocab_size: Number of words in vocabulary.
	    :param embed_dim: Number of embedding dimensions
	    """
		self.batch_size = batch_size
		self.rnn_size = rnn_size
		self.embed_dim = embed_dim
		self.vocab_size = vocab_size

	def get_inputs(self):
	    """
	    Create TF Placeholders for input, targets, and learning rate.
	    :return: Tuple (input, targets, learning rate)
	    """
	    inputs = tf.placeholder(tf.int32, [None, None], name='input')
	    targets = tf.placeholder(tf.int32, [None, None], name='targets')
	    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
	    return inputs, targets, learning_rate

	def get_init_cell(self):
	    """
	    Create an RNN Cell and initialize it.
	    :return: Tuple (cell, initialize state)
	    """
	    def single_cell():
	        return tf.contrib.rnn.BasicLSTMCell(self.rnn_size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
	    
	    cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(2)], state_is_tuple=True)
	    
	    initial_state = cell.zero_state(self.batch_size, tf.float32)
	    
	    initial_state = tf.identity(initial_state, name="initial_state")
	    return cell, initial_state

	def get_embed(self, input_data):
	    """
	    Create embedding for <input_data>.
	    :param input_data: TF placeholder for text input.
	    :return: Embedded input.
	    """
	    embedding = tf.Variable(tf.random_uniform((self.vocab_size, self.embed_dim), -1, 1))
	    embed = tf.nn.embedding_lookup(embedding, input_data)
	    return embed

	def build_rnn(self, cell, inputs):
	    """
	    Create a RNN using a RNN Cell
	    :param cell: RNN Cell
	    :param inputs: Input text data
	    :return: Tuple (Outputs, Final State)
	    """
	    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
	    final_state = tf.identity(final_state, name="final_state")
	    return outputs, final_state

	def build_nn(self, cell, input_data):
	    """
	    Build part of the neural network
	    :param cell: RNN cell
	    :param input_data: Input data
	    :return: Tuple (Logits, FinalState)
	    """
	    embed = get_embed(input_data)
	    outputs, final_state = build_rnn(cell, embed)
	    logits = tf.layers.dense(outputs, self.vocab_size)
	    return logits, final_state
