import math
import numpy as np
from collections import Counter
import argparse

from tensorflow.contrib import seq2seq
import tensorflow as tf

import models
import util

parser = argparse.ArgumentParser(description='tv scripts generation')

# Location of data
parser.add_argument('--save_dir', type=str, default='./save',
                    help='model save path')
parser.add_argument('--data_dir', type=str, default='./data/simpsons/moes_tavern_lines.txt',
                    help='simpsons dataset folder path')

# Training options
parser.add_argument('--batch_size', type=int, default=500,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--rnn_size', type=int, default=256,
                    help='upper epoch limit')
parser.add_argument('--embed_dim', type=int, default=256,
                    help='Embedding Dimension Size')
parser.add_argument('--seq_length', type=int, default=20,
                    help='Sequence Length')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--show_every_n_batches', type=int, default=10,
                    help='Show stats for every n number of batches')

args = parser.parse_args()


data_dir = args.data_dir
text = util.load_data(data_dir)

# Ignore notice, since we don't use it for analysing the data
text = text[81:]

# Preprocessing Functions

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

# Implement the function token_lookup to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".
# Create a dictionary for the following symbols where the symbol is the key and value is the token:

# Period ( . )
# Comma ( , )
# Quotation Mark ( " )
# Semicolon ( ; )
# Exclamation mark ( ! )
# Question mark ( ? )
# Left Parentheses ( ( )
# Right Parentheses ( ) )
# Dash ( -- )
# Return ( \n )

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    res = {'.':'||Period||', ',':'||Comma||', '"':'||Quotation_Mark||',';':'||Semicolon||', '!':'||Exclamation_mark||', '?':'||Question_mark||', '(':'||Left_Parentheses||',')':'Right_Parentheses','--':'Dash','\n':'Return'}
    return res

# Preprocess Training, Validation, and Testing Data
util.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

int_text, vocab_to_int, int_to_vocab, token_dict = util.load_preprocess()


# Number of Epochs
num_epochs = args.epochs
# Batch Size
batch_size = args.batch_size
# RNN Size
rnn_size = args.rnn_size
# Embedding Dimension Size
embed_dim = args.embed_dim
# Sequence Length
seq_length = args.seq_length
# Learning Rate
learning_rate = args.learning_rate
# Show stats for every n number of batches
show_every_n_batches = args.show_every_n_batches

save_dir = args.save_dir
# build the graph
vocab_size = len(int_to_vocab)
model = models.LSTM(batch_size, rnn_size, embed_dim, vocab_size)
train_graph = tf.Graph()
with train_graph.as_default():
    input_text, targets, lr = model.get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = model.get_init_cell()
    logits, final_state = model.build_nn(cell, input_text)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    x = batch_size * seq_length
    n_batches = math.floor(len(int_text)//x)
    # 一个batch内的word数量
    n = math.floor(len(int_text)//batch_size)
    nn = batch_size * n_batches * seq_length
    nnn = seq_length * n_batches
    nums = n_batches * seq_length
    res = []
    for num in range(0, nums, seq_length):
        batch_input = []
        batch_target = []
        for idx in range(0, nn, nnn):
            batches = int_text[idx:idx+n+1]
            
            x = batches[num:num+seq_length]
            
            y = batches[num+1:num+seq_length+1]
            if y[seq_length - 1] == nn:
                y[seq_length - 1] = 0
            batch_input.extend(x)
            batch_target.extend(y)
        res.extend(batch_input)
        res.extend(batch_target)
    res = np.array(res)
    res = res.reshape(n_batches, 2, batch_size, seq_length)
    return res

# train the model
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')

util.save_params((seq_length, save_dir))