import numpy as np
import argparse
import tensorflow as tf

import util
import models

parser = argparse.ArgumentParser(description='image classification train')

# Location of data
parser.add_argument('--save_path', type=str, default='checkpoints/dev',
                    help='model save path')
parser.add_argument('--source_path', type=str, default='data/small_vocab_en',
                    help='source text path')
parser.add_argument('--target_path', type=str, default='data/small_vocab_fr',
                    help='target text path')

# Training options
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--rnn_size', type=int, default=256,
                    help='upper epoch limit')
parser.add_argument('--encoding_embedding_size', type=int, default=200,
                    help='Encode Embedding Dimension Size')
parser.add_argument('--decoding_embedding_size', type=int, default=200,
                    help='Decode Embedding Dimension Size')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--display_step', type=int, default=20,
                    help='Show stats for every n number of batches')
parser.add_argument('--num_layers', type=int, default=3,
                    help='Number of Layers')

args = parser.parse_args()

# get the data
source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = util.load_data(source_path)
target_text = util.load_data(target_path)


"""
As you did with other RNNs, you must turn the text into a number so the computer can understand it. 
In the function text_to_ids(), you'll turn source_text and target_text from words to ids.
However, you need to add the <EOS> word id at the end of target_text.
This will help the neural network predict when the sentence should end.
"""
def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    source_id_text = []
    target_id_text = []
    source_text_sentence = source_text.split("\n")
    target_text_sentence = target_text.split("\n")
    for i, val in enumerate(source_text_sentence):
        source_id_temp = []
        source_text_list = val.split()
        for value in source_text_list:
            source_id = source_vocab_to_int[value]
            source_id_temp.append(source_id)
        source_id_text.append(source_id_temp)
    for i, val in enumerate(target_text_sentence):
        target_id_temp = []
        target_text_list = val.split()
        for value in target_text_list:
            target_id = target_vocab_to_int[value]
            target_id_temp.append(target_id)
        target_id_temp.append(target_vocab_to_int['<EOS>'])
        target_id_text.append(target_id_temp)
    return source_id_text, target_id_text

# Preprocess all the data and save it
util.preprocess_and_save_data(source_path, target_path, text_to_ids)


# Number of Epochs
epochs = args.epochs
# Batch Size
batch_size = args.batch_size
# RNN Size
rnn_size = args.rnn_size
# Number of Layers
num_layers = args.num_layers
# Embedding Size
encoding_embedding_size = args.encoding_embedding_size
decoding_embedding_size = args.decoding_embedding_size
# Learning Rate
learning_rate = args.learning_rate
# Dropout Keep Probability
keep_probability = args.keep_prob
display_step = args.display_step


# build the graph
save_path = args.save_path
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = util.load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

model = models.SeqToSeq(rnn_size, num_layers, len(target_vocab_to_int), len(source_vocab_to_int), batch_size, encoding_embedding_size, decoding_embedding_size, keep_prob)

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = model.model_inputs()

    #sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)

    train_logits, inference_logits = model.seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   source_sequence_length,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   target_vocab_to_int)


    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

# Batch and pad the source and target sequences
def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths


# train the model
def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

# Split data to training and validation sets
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source,
                                                                                                             valid_target,
                                                                                                             batch_size,
                                                                                                             source_vocab_to_int['<PAD>'],
                                                                                                             target_vocab_to_int['<PAD>']))                                                                                                  
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths,
                 keep_prob: keep_probability})


            if batch_i % display_step == 0 and batch_i > 0:


                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     source_sequence_length: sources_lengths,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})


                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     source_sequence_length: valid_sources_lengths,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)

                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')

util.save_params(save_path)