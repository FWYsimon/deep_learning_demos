import argparse
import numpy as np
import pickle

import tensorflow as tf

import util
import models

parser = argparse.ArgumentParser(description='image classification train')

# Location of data
parser.add_argument('--save_path', type=str, default='./image_classification',
                    help='model save path')
parser.add_argument('--data_path', type=str, default='cifar-10-batches-py',
                    help='cifar10 dataset folder path')

# Training options
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--keep_prob', type=float, default=0.75,
                    help='upper epoch limit')

args = parser.parse_args()

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    return x / 255

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    one_hot = []
    for x_ in x:
        a = np.zeros(10)
        np.put(a, x_ ,1)
        one_hot.append(a)

    return np.array(one_hot)

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    print(session.run(cost, feed_dict={
                x: feature_batch,
                y: label_batch,
                keep_prob: 1.0
    }))
    print(session.run(accuracy, feed_dict={
                x: valid_features,
                y: valid_labels,
                keep_prob: 1.0
    }))

# data preprocess
cifar10_dataset_folder_path = args.data_path
util.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

# load data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

# build the model
tf.reset_default_graph()

model = models.CNN(image_shape=(32, 32, 3),
				n_classes=10)

# Inputs
x = model.inputs
y = model.label
keep_prob = model.keep_prob

# Model
logits = model.conv_net(x, keep_prob)
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

# hyperparameter
epochs = args.epochs
batch_size = args.batch_size
keep_probability = args.keep_prob

save_model_path = args.save_path

# train the model
print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in util.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)