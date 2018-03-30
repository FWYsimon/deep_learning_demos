import numpy as np
import tensorflow as tf
import argparse
import os
from glob import glob
from matplotlib import pyplot


import util
import models

parser = argparse.ArgumentParser(description='face generation')

# Location of data
parser.add_argument('--data_dir', type=str, default='./data',
                    help='dataset folder path')

# Training options
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--z_dim', type=int, default=100,
                    help='The dimension of Z')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='The exponential decay rate for the 1st moment in the optimizer')

args = parser.parse_args()

data_dir = args.data_dir

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = util.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()

def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """

    if (data_image_mode == "RGB"):
        image_channels = 3
    if (data_image_mode == "L"):
        image_channels = 1
    model = models.GANs(28, 28, image_channels, z_dim, image_channels, learning_rate, beta1)
    input_real, input_z, _ = model.model_inputs()
    d_loss, g_loss = model.model_loss(input_real, input_z)
    d_train_opt, g_train_opt = model.model_opt(d_loss, g_loss)
    
    samples = []
    losses = []
    steps = 0
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                # TODO: Train Model
                steps += 1
                batch_z = np.random.uniform(-1, 1, size=[batch_size, z_dim])
                batch_images = batch_images*2
                
                _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(g_train_opt, feed_dict={input_z: batch_z})
                if steps % 100 == 0:
                    show_generator_output(sess, 25, input_z, image_channels, data_image_mode)
            train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
            train_loss_g = g_loss.eval({input_z: batch_z})
            print("Epoch {}/{}...".format(epoch_i+1, epochs),
              "Discriminator Loss: {:.4f}...".format(train_loss_d),
              "Generator Loss: {:.4f}".format(train_loss_g))  
            losses.append((train_loss_d, train_loss_g))
            saver.save(sess, './checkpoints/generator.ckpt')

# hyperparameter
batch_size = args.batch_size
z_dim = args.z_dim
learning_rate = args.learning_rate
beta1 = args.beta1

epochs = args.epochs

celeba_dataset = util.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)