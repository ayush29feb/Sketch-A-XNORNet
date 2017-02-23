"""Trains the Sketch-A-Net Network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time

import tensorflow as tf
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin

from data_layer import DataLayer, load_pretrained_model
import sketchnet as sn

FLAGS = None

def do_eval(sess, 
            eval_correct, 
            images_placeholder,
            labels_placeholder,
            validation_set_placeholder,
            dataset,
            is_val=True):
    
    num_examples = DataLayer.NUM_ITEMS_PER_CLASS * DataLayer.NUM_CLASSES
    steps_per_epoch = num_examples // dataset.batch_size
    true_count = 0
    for step in xrange(steps_per_epoch):
        images, labels = dataset.next_batch_test() if is_val else dataset.next_batch_train()
        true_count += sess.run(eval_correct, feed_dict={
            images_placeholder: images, 
            labels_placeholder: labels, 
            validation_set_placeholder: is_val})
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def run_training():
    """Trains and evaluates the Sketch-A-Net network
    """
    # Instantiate the data layer to have easy access of the dataset
    dataset = DataLayer(FLAGS.data_path, batch_size=FLAGS.batch_size)
    
    # Load the pretrained models
    pretrained = load_pretrained_model(FLAGS.pretrain_path)

    # Tell tensorflow that the model will be built into the default graph
    with tf.Graph().as_default():
        ############### Create all the placeholders ###############
        # Instantiate the required placeholders for images, labels, dropout, learning_rate
        images_placeholder = tf.placeholder(tf.float32, name='images_pl')       
        labels_placeholder = tf.placeholder(tf.float32, name='labels_pl')

        dropout_rate_placeholder = tf.placeholder_with_default(FLAGS.dropout_rate, shape=(), name='learning_rate_pl')
        learning_rate_placeholder = tf.placeholder_with_default(FLAGS.learning_rate, shape=(), name='learning_rate_pl')
        is_val_placeholder = tf.placeholder(tf.bool, shape=(), name='is_val_pl')

        ############### Declare all the Ops for the graph ###############
        # Build a graph that computes predictions from the inference model
        logits = sn.inference(images_placeholder, dropout_rate_placeholder, pretrained=pretrained)

        # Add the loss Op to the graph
        loss = sn.loss(logits, labels_placeholder)

        # Add the Op to calculate and apply gradient to the graph
        train_op = sn.training(loss, learning_rate_placeholder)

        eval_correct = sn.evaluation(logits, labels_placeholder, is_val_placeholder)

        # Add the variable initializer Op to the graph
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints
        saver = tf.train.Saver()

        # Create a session for running the Ops on the graph
        sess = tf.Session()

        # create a summary writer
        summary_writter = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        ############### Start Running the Ops ###############
        # Run the Op to initialize variables
        sess.run(init)

        # the training loop
        max_steps = (int) (FLAGS.epoch * DataLayer.NUM_CLASSES * DataLayer.NUM_TRAIN_ITEMS_PER_CLASS / FLAGS.batch_size)
        epoch_size = DataLayer.NUM_CLASSES * DataLayer.NUM_TRAIN_ITEMS_PER_CLASS / FLAGS.batch_size
        for step in xrange(max_steps):
            start_time = time.time()
            
            # fill the feed_dict and evalute the loss and train_op
            images, labels = dataset.next_batch_train()
            _, loss_value = sess.run([train_op, loss], feed_dict={images_placeholder: images, labels_placeholder: labels})
            
            duration = time.time() - start_time

            # print the status every 10 steps
            if step % 10 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            
            # save and evalutae the model every 10 epochs
            if step % (10 * epoch_size) == 0:
                checkpoint_file = os.path.join(FLAGS.ckpt_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                
                # Do evaluation of the validation set
                do_eval(sess, 
                        eval_correct, 
                        images_placeholder,
                        labels_placeholder,
                        is_val_placeholder,
                        dataset,
                        is_val=True)
                
                # Do evaluation of the training set
                do_eval(sess, 
                        eval_correct, 
                        images_placeholder,
                        labels_placeholder,
                        is_val_placeholder,
                        dataset,
                        is_val=False)

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    if tf.gfile.Exists(FLAGS.ckpt_dir):
        tf.gfile.DeleteRecursively(FLAGS.ckpt_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.ckpt_dir)
    if not tf.gfile.Exists(FLAGS.data_path):
        raise IOError('The file at' + FLAGS.data_path + ' does not exsits.')
    print('Starting the training...')
    run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tensorflow/sketch-a-net/logs/training',
        help='Directory to save the logs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='The initial learning rate for the optimizer'
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.5,
        help='The probability to keep a neuron in dropout'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=135,
        help='The batch_size'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=500,
        help='epoch size, the number of times the trainer should use the dataset'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='dataset/dataset_with_order_info_224.mat',
        help='The .mat file with the dataset downloaded from http://www.eecs.qmul.ac.uk/~tmh/downloads.html'
    )
    parser.add_argument(
        '--pretrain_path',
        type=str,
        default='dataset/model_with_order_info_224.mat',
        help='The .mat file with the pretrained weights downloaded from http://www.eecs.qmul.ac.uk/~tmh/downloads.html'
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='/tmp/tensorflow/sketch-a-net/logs/ckpts'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
