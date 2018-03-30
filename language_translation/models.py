import tensorflow as tf

class SeqToSeq(object):
	def __init__(self, rnn_size, num_layers, target_vocab_size, source_vocab_size, batch_size, enc_embedding_size, dec_embedding_size, keep_prob):
		"""
		:param rnn_size: RNN Size
		:param rnn_inputs: Inputs for the RNN
		:param target_vocab_size: Size of target vocabulary
		:param source_vocab_size: vocabulary size of source data
		:param batch_size: Batch Size
		:param enc_embedding_size: embedding size of source data
		:param dec_embedding_size: Decoding embedding size
		:param keep_prob: Dropout keep probability
		"""
		self.rnn_size = rnn_size
		self.num_layers = num_layers
		self.target_vocab_size = target_vocab_size
		self.source_vocab_size = source_vocab_size
		self.batch_size = batch_size
		self.enc_embedding_size = enc_embedding_size
		self.dec_embedding_size = dec_embedding_size
		self.keep_prob = keep_prob

	def model_inputs(self):
	    """
	    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
	    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
	    max target sequence length, source sequence length)
	    """
	    inputs = tf.placeholder(tf.int32, [None, None], name="input")
	    targets = tf.placeholder(tf.int32, [None, None])
	    learning_rate = tf.placeholder(tf.float32)
	    probs = tf.placeholder(tf.float32, name="keep_prob")
	    target_sequence_length = tf.placeholder(tf.int32, [None], name="target_sequence_length")
	    max_target_len = tf.reduce_max(target_sequence_length, name="max_target_len")
	    source_sequence_length = tf.placeholder(tf.int32, [None], name="source_sequence_length")
	    return inputs, targets, learning_rate, probs, target_sequence_length, max_target_len, source_sequence_length

	def process_decoder_input(self, target_data, target_vocab_to_int):
	    """
	    Preprocess target data for encoding
	    :param target_data: Target Placehoder
	    :param target_vocab_to_int: Dictionary to go from the target words to an id
	    :return: Preprocessed target data
	    """
	    ending = tf.strided_slice(target_data, [0, 0], [self.batch_size, -1], [1, 1])
	    dec_input = tf.concat([tf.fill([self.batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)
	    return dec_input

	def encoding_layer(self, rnn_inputs, source_sequence_length):
	    """
	    Create encoding layer
	    :param num_layers: Number of layers
	    :param source_sequence_length: a list of the lengths of each sequence in the batch
	    :return: tuple (RNN output, RNN state)
	    """
	    batch_size = rnn_inputs.get_shape().as_list()[0]
	    embed = tf.contrib.layers.embed_sequence(rnn_inputs, vocab_size=self.source_vocab_size, embed_dim=self.enc_embedding_size)
	    def lstm_cell():
	        enc_cell = tf.contrib.rnn.LSTMCell(self.rnn_size,
	                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
	        dec_cell = tf.contrib.rnn.DropoutWrapper(enc_cell,
	                                            input_keep_prob=self.keep_prob)
	        return dec_cell
	    enc_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self.num_layers)])

	    outputs, final_state = tf.nn.dynamic_rnn(enc_cell, embed, sequence_length=source_sequence_length, dtype=tf.float32)
	    return outputs, final_state

	def decoding_layer_train(self, encoder_state, dec_cell, dec_embed_input, 
	                         target_sequence_length, max_summary_length, 
	                         output_layer):
	    """
	    Create a decoding layer for training
	    :param encoder_state: Encoder State
	    :param dec_cell: Decoder RNN Cell
	    :param dec_embed_input: Decoder embedded input
	    :param target_sequence_length: The lengths of each sequence in the target batch
	    :param max_summary_length: The length of the longest sequence in the batch
	    :param output_layer: Function to apply the output layer
	    :return: BasicDecoderOutput containing training logits and sample_id
	    """
	    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)
	    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer=output_layer)
	    output = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
	                                                                       maximum_iterations=max_summary_length)[0]
	    return output

	def decoding_layer_infer(self, encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
	                         end_of_sequence_id, max_target_sequence_length,
	                         output_layer):
	    """
	    Create a decoding layer for inference
	    :param encoder_state: Encoder state
	    :param dec_cell: Decoder RNN Cell
	    :param dec_embeddings: Decoder embeddings
	    :param start_of_sequence_id: GO ID
	    :param end_of_sequence_id: EOS Id
	    :param max_target_sequence_length: Maximum length of target sequences
	    :param output_layer: Function to apply the output layer
	    :return: BasicDecoderOutput containing inference logits and sample_id
	    """
	    start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [self.batch_size], name='start_tokens')
	    
	    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens, end_of_sequence_id)
	    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer=output_layer)
	    output, final_state = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
	                                                            maximum_iterations=max_target_sequence_length)
	    return output

	def decoding_layer(self, dec_input, encoder_state,
	                   target_sequence_length, max_target_sequence_length,
	                   target_vocab_to_int):
	    """
	    Create decoding layer
	    :param dec_input: Decoder input
	    :param encoder_state: Encoder state
	    :param target_sequence_length: The lengths of each sequence in the target batch
	    :param max_target_sequence_length: Maximum length of target sequences
	    :param target_vocab_to_int: Dictionary to go from the target words to an id
	    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
	    """
	    start_of_sequence_id = target_vocab_to_int['<GO>']
	    end_of_sequence_id = target_vocab_to_int['<EOS>']
	    dec_embeddings = tf.Variable(tf.random_uniform([self.target_vocab_size, self.dec_embedding_size]))
	    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
	    def make_cell():
	        dec_cell = tf.contrib.rnn.LSTMCell(self.rnn_size,
	                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
	        dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
	                                            input_keep_prob=keep_prob)
	        return dec_cell
	    dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
	    output_layer = Dense(self.target_vocab_size,
	                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
	    with tf.variable_scope("decode"):
	        training_decoder_output = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
	                                                       target_sequence_length, max_target_sequence_length, output_layer)
	    with tf.variable_scope("decode", reuse=True):
	        inference_decoder_output = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, 
	                                                        end_of_sequence_id, max_target_sequence_length, output_layer)
	    return training_decoder_output, inference_decoder_output

	def seq2seq_model(self, input_data, target_data,
	                  source_sequence_length, target_sequence_length,
	                  max_target_sentence_length,
	                  target_vocab_to_int):
	    """
	    Build the Sequence-to-Sequence part of the neural network
	    :param input_data: Input placeholder
	    :param target_data: Target placeholder
	    :param keep_prob: Dropout keep probability placeholder
	    :param batch_size: Batch Size
	    :param source_sequence_length: Sequence Lengths of source sequences in the batch
	    :param target_sequence_length: Sequence Lengths of target sequences in the batch
	    :param source_vocab_size: Source vocabulary size
	    :param target_vocab_size: Target vocabulary size
	    :param enc_embedding_size: Decoder embedding size
	    :param dec_embedding_size: Encoder embedding size
	    :param rnn_size: RNN Size
	    :param num_layers: Number of layers
	    :param target_vocab_to_int: Dictionary to go from the target words to an id
	    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
	    """
	    _, enc_state = encoding_layer(input_data, source_sequence_length)
	    dec_input = process_decoder_input(target_data, target_vocab_to_int)
	    training_decoder_output, inference_decoder_output = decoding_layer(dec_input,
	                                                                       enc_state, 
	                                                                       target_sequence_length, 
	                                                                       max_target_sentence_length,
	                                                                       target_vocab_to_int)
	    return training_decoder_output, inference_decoder_output