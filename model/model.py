import tensorflow as tf
import numpy as np


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def inference(images):
    """This is essentailly the tensorflow graph for the Sketch-A-Net network

    Args:
        images: the input images of shape (N, H, W, C) for the network returned from the data layer

    Returns:
        Logits for the softmax loss
    """
    # Layer 1
    weights1 = weight_variable([15, 15, 6, 64])
    biases1 = bias_variable([64])
    conv1 = tf.nn.conv2d(images, weights1, [1, 3, 3, 1], padding='VALID')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases1))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2
    weights2 = weight_variable([5, 5, 64, 128])
    biases2 = bias_variable([128])
    conv2 = tf.nn.conv2d(pool1, weights2, [1, 1, 1, 1], padding='VALID')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases2))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], stides=[1, 2, 2, 1], padding='VALID')

    # Layer 3
    weights3 = weight_variable([3, 3, 128, 256])
    biases3 = bias_variable([256])
    conv3 = tf.nn.conv2d(pool2, weights3, [1, 1, 1, 1], padding='SAME')
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, biases3))

    # Layer 4
    weights4 = weight_variable([3, 3, 256, 256])
    biases4 = bias_variable([256])
    conv4 = tf.nn.conv2d(relu3, weights4, [1, 1, 1, 1], padding='SAME')
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, biases4))

    # Layer 5
    weights5 = weight_variable([3, 3, 256, 256])
    biases5 = bias_variable([256])
    conv5 = tf.nn.conv2d(relu4, weights5, [1, 1, 1, 1], padding='SAME')
    relu5 = tf.nn.relu(tf.nn.bias_add(conv5, biases5))
    pool5 = tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 6
    weights6 = weight_variable([7, 7, 256, 512])
    biases6 = bias_variable([512])
    fc6 = tf.nn.conv2d(pool5, weights6, [1, 1, 1, 1], padding='VALID')
    relu6 = tf.nn.relu(tf.nn.bias_add(fc6, biases6))
    dropout6 = tf.nn.dropout(relu6, 0.5)

    # Layer 7
    weights7 = weight_variable([1, 1, 512, 512])
    biases7 = bias_variable([512])
    fc7 = tf.nn.conv2d(dropout6, weights7, [1, 1, 1, 1], padding='VALID')
    relu7 = tf.nn.relu(tf.nn.bias_add(fc7, biases7))
    dropout7 = tf.nn.dropout(relu7, 0.5)

    # Layer 8
    weights8 = weight_variable([1, 1, 512, 250])
    biases8 = bias_variable([250])
    fc8 = tf.nn.conv2d(dropout7, weights8, [1, 1, 1, 1], padding='VALID')

