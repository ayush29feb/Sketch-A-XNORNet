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
            dataset,
            is_val):
    
    num_examples = (DataLayer.NUM_TEST_ITEMS_PER_CLASS if is_val else DataLayer.NUM_TRAIN_ITEMS_PER_CLASS) * DataLayer.NUM_CLASSES
    steps_per_epoch = num_examples // dataset.batch_size
    if is_val:
        steps_per_epoch *= 10
    true_count = 0
    total_duration = 0
    for step in xrange(steps_per_epoch):
        start_time = time.time()
        images, labels = dataset.next_batch_test() if is_val else dataset.next_batch_train()
        count = sess.run(eval_correct, feed_dict={
            images_placeholder: images, 
            labels_placeholder: labels})
        true_count += count
        duration = time.time() - start_time
        total_duration += duration 
        # print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f (%.3f sec)' % (dataset.batch_size, count, float(count) / dataset.batch_size, duration))
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f (%.3f sec)' %
        (num_examples, true_count, precision, total_duration))

def run_training():
    """Trains and evaluates the Sketch-A-Net network
    """
    # Instantiate the data layer to have easy access of the dataset
    dataset = DataLayer(FLAGS.data_path, batch_size=FLAGS.batch_size)
    
    # Load the pretrained models
    pretrained = load_pretrained_model(FLAGS.pretrain_path if FLAGS.pretrain else None)

    # Tell tensorflow that the model will be built into the default graph
    with tf.Graph().as_default():
        ############### Create all the placeholders ###############
        # Instantiate the required placeholders for images, labels, dropout, learning_rate
        images_placeholder = tf.placeholder(tf.float32, name='images_pl')       
        labels_placeholder = tf.placeholder(tf.float32, name='labels_pl')

        dropout_rate_placeholder = tf.placeholder_with_default(1.0, shape=(), name='learning_rate_pl')
        learning_rate_placeholder = tf.placeholder_with_default(FLAGS.learning_rate, shape=(), name='learning_rate_pl')

        ############### Declare all the Ops for the graph ###############
        # Build a graph that computes predictions from the inference model
        logits = sn.inference(images_placeholder, dropout_rate_placeholder, pretrained=pretrained)

        # Add the loss Op to the graph
        loss = sn.loss(logits, labels_placeholder)

        # Add the Op to calculate and apply gradient to the graph
        train_op = sn.training(loss, learning_rate_placeholder)

        # Evaluation
        eval_correct_train = sn.evaluation(logits, labels_placeholder, False)
        eval_correct_test = sn.evaluation(logits, labels_placeholder, True)

        # Add the variable initializer Op to the graph
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints
        saver = tf.train.Saver()

        # Create a session for running the Ops on the graph
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            # Restore the variables
            latest_ckpt_file = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
            if latest_ckpt_file is not None:
                saver.restore(sess, latest_ckpt_file)
                print('Model Restored')

            # create a summary writer
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            summary_merged = tf.summary.merge_all()

            ############### Start Running the Ops ###############
            # Run the Op to initialize variables
            sess.run(init)

            # the training loop
            if not FLAGS.eval_only:
                max_steps = (int) (FLAGS.epoch * DataLayer.NUM_CLASSES * DataLayer.NUM_TRAIN_ITEMS_PER_CLASS / FLAGS.batch_size)
                epoch_size = DataLayer.NUM_CLASSES * DataLayer.NUM_TRAIN_ITEMS_PER_CLASS / FLAGS.batch_size
                for step in xrange(max_steps):
                    start_time = time.time()
                    
                    # fill the feed_dict and evalute the loss and train_op
                    images, labels = dataset.next_batch_train()
                    feed_dict = {
                        images_placeholder: images,
                        labels_placeholder: labels,
                        dropout_rate_placeholder: FLAGS.dropout_rate,
                        learning_rate_placeholder: FLAGS.learning_rate
                    }
                    _, loss_value, summary_str = sess.run([train_op, loss, summary_merged], feed_dict=feed_dict)
                    
                    duration = time.time() - start_time

                    # save every 100 steps
                    if step % 100 == 0:
                        print('Saving Checkpoint...')
                        checkpoint_file = os.path.join(FLAGS.ckpt_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_file, global_step=step)
                        print('Checkpoint Saved!')

                    # print the status every 10 steps
                    if step % 10 == 0:
                        summary_writer.add_summary(summary_str, step)
                        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

                    # evalutae the model every 10 epochs
                    if (step + 1) % (10 * epoch_size) == 0:
                        # Do evaluation of the validation set
                        do_eval(sess, 
                                eval_correct_test, 
                                images_placeholder,
                                labels_placeholder,
                                dataset,
                                is_val=True)
                        
                        # Do evaluation of the training set
                        do_eval(sess, 
                                eval_correct_train, 
                                images_placeholder,
                                labels_placeholder,
                                dataset,
                                is_val=False)
            # Final Evaluation
            do_eval(sess, 
                    eval_correct_test, 
                    images_placeholder,
                    labels_placeholder,
                    dataset,
                    is_val=True)

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    if not tf.gfile.Exists(FLAGS.ckpt_dir):
        tf.gfile.MakeDirs(FLAGS.ckpt_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    
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
        default='dataset/dataset_with_order_info_256.mat',
        help='The .mat file with the dataset downloaded from http://www.eecs.qmul.ac.uk/~tmh/downloads.html'
    )
    parser.add_argument(
        '--pretrain_path',
        type=str,
        default='dataset/model_with_order_info_256.mat',
        help='The .mat file with the pretrained weights downloaded from http://www.eecs.qmul.ac.uk/~tmh/downloads.html'
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='/tmp/tensorflow/sketch-a-net/logs/ckpts'
    )
    parser.add_argument(
        '--eval_only',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--pretrain',
        type=bool,
        default=True
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
