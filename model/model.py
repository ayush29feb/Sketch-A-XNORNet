import tensorflow as tf
import numpy as np

def weight_variable(shape, pretrained=None, layer=None):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    # if pretrained:
    #     initial = tf.constant_initializer(value=pretrained['conv' + str(layer)], dtype=tf.float32)
    return tf.Variable(pretrained['conv' + str(layer)], name='weights')

def bias_variable(shape, pretrained=None, layer=None):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    # if pretrained:
    #     initial = tf.constant_initializer(value=pretrained['conv' + str(layer)], dtype=tf.float32)
    return tf.Variable(pretrained['conv' + str(layer)], name='biases')

def inference(images, pretrained=None):
    """This is essentailly the tensorflow graph for the Sketch-A-Net network

    Args:
        images: the input images of shape (N, H, W, C) for the network returned from the data layer

    Returns:
        Logits for the softmax loss
    """
    # Layer 1
    with tf.name_scope('L1') as scope:
        weights1 = weight_variable([15, 15, 6, 64], pretrained[0], 1)
        biases1 = bias_variable([64], pretrained[1], 1)
        conv1 = tf.nn.conv2d(images, weights1, [1, 3, 3, 1], padding='VALID', name='conv1')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases1), name='relu1')
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    # Layer 2
    with tf.name_scope('L2') as scope:
        weights2 = weight_variable([5, 5, 64, 128], pretrained[0], 2)
        biases2 = bias_variable([128], pretrained[1], 2)
        conv2 = tf.nn.conv2d(pool1, weights2, [1, 1, 1, 1], padding='VALID', name='conv2')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases2), name='relu2')
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    # Layer 3
    with tf.name_scope('L3') as scope:
        weights3 = weight_variable([3, 3, 128, 256], pretrained[0], 3)
        biases3 = bias_variable([256], pretrained[1], 3)
        conv3 = tf.nn.conv2d(pool2, weights3, [1, 1, 1, 1], padding='SAME', name='conv3')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, biases3), name='relu3')

    # Layer 4
    with tf.name_scope('L4') as scope:
        weights4 = weight_variable([3, 3, 256, 256], pretrained[0], 4)
        biases4 = bias_variable([256], pretrained[1], 4)
        conv4 = tf.nn.conv2d(relu3, weights4, [1, 1, 1, 1], padding='SAME', name='conv4')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, biases4), name='relu4')

    # Layer 5
    with tf.name_scope('L5') as scope:
        weights5 = weight_variable([3, 3, 256, 256], pretrained[0], 5)
        biases5 = bias_variable([256], pretrained[1], 5)
        conv5 = tf.nn.conv2d(relu4, weights5, [1, 1, 1, 1], padding='SAME', name='conv5')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, biases5), name='relu5')
        pool5 = tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

    # Layer 6
    with tf.name_scope('L6') as scope:
        weights6 = weight_variable([7, 7, 256, 512], pretrained[0], 6)
        biases6 = bias_variable([512], pretrained[1], 6)
        fc6 = tf.nn.conv2d(pool5, weights6, [1, 1, 1, 1], padding='VALID', name='fc6')
        relu6 = tf.nn.relu(tf.nn.bias_add(fc6, biases6), name='relu6')
        dropout6 = tf.nn.dropout(relu6, 1.0, name='dropout6')

    # Layer 7
    with tf.name_scope('L7') as scope:
        weights7 = weight_variable([1, 1, 512, 512], pretrained[0], 7)
        biases7 = bias_variable([512], pretrained[1], 7)
        fc7 = tf.nn.conv2d(dropout6, weights7, [1, 1, 1, 1], padding='VALID', name='fc7')
        relu7 = tf.nn.relu(tf.nn.bias_add(fc7, biases7), name='relu7')
        dropout7 = tf.nn.dropout(relu7, 1.0, name='dropout7')

    # Layer 8
    with tf.name_scope('L8') as scope:
        weights8 = weight_variable([1, 1, 512, 250], pretrained[0], 8)
        biases8 = bias_variable([250], pretrained[1], 8)
        fc8 = tf.nn.conv2d(dropout7, weights8, [1, 1, 1, 1], padding='VALID', name='fc8')

    logits = tf.reshape(fc8, [-1, 250])
    return logits

def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    y_ = tf.argmax(logits, axis=1)
    tf.summary.histogram('logits', y_)
    correct = tf.nn.in_top_k(logits, labels, 5)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
