import tensorflow as tf
import numpy as np

def weight_variable(shape, weights=None):
    """Initializes the weights variable for a required layer using the
    pretrained model if provided or else initialize using a normal distribution.

    Args:
        shape: the shape of the bias variable to be created
        weights: pretrained weights

    Returns:
        tf.Variable with the respectives weights
    """
    if weights is not None and weights.shape != shape:
        raise ValueError('The pretrained shapes don\'t match with the layer shapes')
    initial = tf.truncated_normal(shape, stddev=0.1) if weights is None else weights
    return tf.Variable(initial, name='weights')

def bias_variable(shape, biases=None):
    """Initializes the biases variable for a required layer using the
    pretrained model if provided or else initialize using a normal distribution.

    Args:
        shape: the shape of the bias variable to be created
        biases: pretrained biases

    Returns:
        tf.Variable with the respectives biases
    """
    if biases is not None and biases.shape != shape:
        raise ValueError('The pretrained shapes don\'t match with the layer shapes')
    initial = tf.truncated_normal(shape, stddev=0.1) if biases is None else biases
    return tf.Variable(initial, name='biases')

def inference(images, dropout_prob=1.0, pretrained=(None, None)):
    """This prepares the tensorflow graph for the vanilla Sketch-A-Net network
    and returns the tensorflow Op from the last fully connected layer

    Args:
        images: the input images of shape (N, H, W, C) for the network returned from the data layer

    Returns:
        Logits for the softmax loss
    """
    weights, biases = pretrained

    # Layer 1
    with tf.name_scope('L1') as scope:
        weights1 = weight_variable((15, 15, 6, 64), weights['conv1'])
        biases1 = bias_variable((64,), biases['conv1'])
        conv1 = tf.nn.conv2d(images, weights1, [1, 3, 3, 1], padding='VALID', name='conv1')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases1), name='relu1')
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    # Layer 2
    with tf.name_scope('L2') as scope:
        weights2 = weight_variable((5, 5, 64, 128), weights['conv2'])
        biases2 = bias_variable((128,), biases['conv2'])
        conv2 = tf.nn.conv2d(pool1, weights2, [1, 1, 1, 1], padding='VALID', name='conv2')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases2), name='relu2')
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    # Layer 3
    with tf.name_scope('L3') as scope:
        weights3 = weight_variable((3, 3, 128, 256), weights['conv3'])
        biases3 = bias_variable((256,), biases['conv3'])
        conv3 = tf.nn.conv2d(pool2, weights3, [1, 1, 1, 1], padding='SAME', name='conv3')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, biases3), name='relu3')

    # Layer 4
    with tf.name_scope('L4') as scope:
        weights4 = weight_variable((3, 3, 256, 256), weights['conv4'])
        biases4 = bias_variable((256,), biases['conv4'])
        conv4 = tf.nn.conv2d(relu3, weights4, [1, 1, 1, 1], padding='SAME', name='conv4')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, biases4), name='relu4')

    # Layer 5
    with tf.name_scope('L5') as scope:
        weights5 = weight_variable((3, 3, 256, 256), weights['conv5'])
        biases5 = bias_variable((256,), biases['conv5'])
        conv5 = tf.nn.conv2d(relu4, weights5, [1, 1, 1, 1], padding='SAME', name='conv5')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, biases5), name='relu5')
        pool5 = tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

    # Layer 6
    with tf.name_scope('L6') as scope:
        weights6 = weight_variable((7, 7, 256, 512), weights['conv6'])
        biases6 = bias_variable((512,), biases['conv6'])
        fc6 = tf.nn.conv2d(pool5, weights6, [1, 1, 1, 1], padding='VALID', name='fc6')
        relu6 = tf.nn.relu(tf.nn.bias_add(fc6, biases6), name='relu6')
        dropout6 = tf.nn.dropout(relu6, keep_prob=dropout_prob, name='dropout6')

    # Layer 7
    with tf.name_scope('L7') as scope:
        weights7 = weight_variable((1, 1, 512, 512), weights['conv7'])
        biases7 = bias_variable((512,), biases['conv7'])
        fc7 = tf.nn.conv2d(dropout6, weights7, [1, 1, 1, 1], padding='VALID', name='fc7')
        relu7 = tf.nn.relu(tf.nn.bias_add(fc7, biases7), name='relu7')
        dropout7 = tf.nn.dropout(relu7, keep_prob=dropout_prob, name='dropout7')

    # Layer 8
    with tf.name_scope('L8') as scope:
        weights8 = weight_variable((1, 1, 512, 250), weights['conv8'])
        biases8 = bias_variable((250,), biases['conv8'])
        fc8 = tf.nn.conv2d(dropout7, weights8, [1, 1, 1, 1], padding='VALID', name='fc8')

    logits = tf.reshape(fc8, [-1, 250])
    return logits

def loss(logits, labels):
    """Applies the softmax loss to given logits

    Args:
        logits: the logits obtained from the inference graph
        labels: the ground truth labels for the respective images

    Returns:
        The loss value obtained form the softmax loss applied
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate=0.001):
    """Returns the training Op for the loss function using the AdamOptimizer

    Args:
        learning_rate: the initial learning_rate

    Returns:
        train_op: the tensorflow's trainig Op
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels, validation):
    """Evaluates the number of correct predictions for the given logits and labels

    Args:
        logits: the logits obtained from the inference graph
        labels: the ground truth labels
    
    Return:
        Returns the number of correct predictions
    """
    y_ = tf.argmax(logits, axis=1)
    if tf.assert_equal(validation, tf.constant(True)):
        y_ = tf.reduce_max(tf.reshape(y_, [10, -1]), axis=0)
    correct = tf.equal(tf.cast(y_, tf.float32), labels[:tf.size(y_)])
    return tf.reduce_sum(tf.cast(correct, tf.int32))
