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
import bwsketchnet as snbw
import sketchxnornet as snxnor

FLAGS = None

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, dataset, is_train):
    """Evaluation Step
    """
    print('Running Evaluation on ' + ('train' if is_train else 'test') + 'set')
    # calculate variable for looping over the entire test set once
    num_examples = (DataLayer.NUM_TRAIN_ITEMS_PER_CLASS if is_train else DataLayer.NUM_TEST_ITEMS_PER_CLASS) * DataLayer.NUM_CLASSES
    steps_per_epoch = num_examples // (dataset.train_batch_size if is_train else dataset.test_batch_size)
    last_step_size = num_examples % (dataset.train_batch_size if is_train else dataset.test_batch_size)
    batch_size = (dataset.train_batch_size if is_train else dataset.test_batch_size)
    # runnning stats
    true_count = 0
    start_time = time.time()

    # eval loop
    for step in xrange(steps_per_epoch):
        images, labels = dataset.next_batch_train() if is_train else dataset.next_batch_test()
        count = sess.run(eval_correct, feed_dict={
            images_placeholder: images,
            labels_placeholder: labels
        })
        true_count += count
        # precision = float(count) / batch_size
        # print ('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        #    (batch_size, count, precision))
    
    # run remaining examples
    if last_step_size > 0:
        images, labels = dataset.next_batch_train(last_step_size) if is_train else dataset.next_batch_test(last_step_size)
        true_count += sess.run(eval_correct, feed_dict={
            images_placeholder: images,
            labels_placeholder: labels
        })
    
    # print logs
    duration = time.time() - start_time
    precision = float(true_count) / num_examples
    tf.summary.scalar('precision', precision)
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f (%.3f sec)' %
        (num_examples, true_count, precision, duration))

def run_training(net=sn):
    """Trains and evaluates the Sketch-A-Net network
    """
    # Instantiate the data layer to have easy access of the dataset
    dataset = DataLayer(FLAGS.data_path, batch_size=FLAGS.batch_size)
    
    # Load the pretrained models
    pretrained = load_pretrained_model(FLAGS.model_path if FLAGS.no_pretrain else None)
    
    # Tell tensorflow that the model will be built into the default graph
    with tf.Graph().as_default():
        ############### Create all the placeholders ###############
        # Instantiate the required placeholders for images, labels, dropout, learning rate
        images_placeholder = tf.placeholder(tf.float32, name='images_pl')       
        labels_placeholder = tf.placeholder(tf.float32, name='labels_pl')

        dr_placeholder = tf.placeholder_with_default(1.0, shape=(), name='dr_pl')

        ############### Declare all the Ops for the graph ###############
        # Build a graph that computes predictions from the inference model
        logits = net.inference(images_placeholder, dr_placeholder, pretrained=pretrained)

        # Add the loss Op to the graph
        loss = net.loss(logits, labels_placeholder)

        # create global_step variable
        global_step = tf.Variable(10000 if FLAGS.no_pretrain else 0, name='global_step', trainable=False)

        # Add the Op to calculate and apply gradient to the graph
        train_op = net.training(loss, FLAGS.lr, global_step, decay_steps=FLAGS.decay_step, decay_rate=FLAGS.decay_rate)

        # Evaluation
        eval_correct_train = net.evaluation(logits, labels_placeholder, k=FLAGS.topk, is_train=True)
        eval_correct_test = net.evaluation(logits, labels_placeholder, k=FLAGS.topk, is_train=False)

        # Add the variable initializer Op to the graph
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints
        saver = tf.train.Saver()

        # Create a session for running the Ops on the graph
        with tf.Session() as sess:
            # create a summary writer
            summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'log', str(np.random.randint(0, 1E+8))), sess.graph)
            summary_merged = tf.summary.merge_all()

            # Restore the variables or Run the Op to initialize variables
            latest_ckpt_file = tf.train.latest_checkpoint(os.path.join(FLAGS.logdir, 'ckpt'))
            if latest_ckpt_file is not None:
                saver.restore(sess, latest_ckpt_file)
                print('Model Restored')
            else:
                sess.run(init)

            # Reset Global Step
            if FLAGS.reset_step:
                sess.run(tf.assign(global_step, 0))

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
                        dr_placeholder: FLAGS.dr
                    }
                    _, loss_value, summary_str, g_step = sess.run([train_op, loss, summary_merged, global_step], feed_dict=feed_dict)
                    
                    duration = time.time() - start_time

                    # save and print the status every 10 steps
                    if step % epoch_size == 0:
                        summary_writer.add_summary(summary_str, g_step)
                    
                    if step % 10 == 0:
                        print('Step %d: loss = %.2f (%.3f sec)' % (g_step, loss_value, duration))

                    # save model every 5 epochs
                    if (step + 1) % (5 * epoch_size) == 0:
                        # Save Model
                        checkpoint_file = os.path.join(FLAGS.logdir, 'ckpt', 'model.ckpt')
                        saver.save(sess, checkpoint_file, global_step=global_step)
                        print('Checkpoint Saved!')

                    # evalutae the model every 10 epochs
                    if (step + 1) % (10 * epoch_size) == 0:
                        # Do evaluation of the validation set
                        do_eval(sess, 
                                eval_correct_test, 
                                images_placeholder,
                                labels_placeholder,
                                dataset,
                                is_train=False)
                    
                        # Do evaluation of the training set
                        do_eval(sess, 
                                eval_correct_train, 
                                images_placeholder,
                                labels_placeholder,
                                dataset,
                                is_train=True)
            # Final Evaluation
            do_eval(sess, 
                    eval_correct_test, 
                    images_placeholder,
                    labels_placeholder,
                    dataset,
                    is_train=False)

def main(_):
    if not tf.gfile.Exists(os.path.join(FLAGS.logdir, 'log')):
        tf.gfile.MakeDirs(os.path.join(FLAGS.logdir, 'log'))
    if not tf.gfile.Exists(os.path.join(FLAGS.logdir, 'ckpt')):
        tf.gfile.MakeDirs(os.path.join(FLAGS.logdir, 'ckpt'))
    
    if not tf.gfile.Exists(FLAGS.data_path):
        raise IOError('The file at' + FLAGS.data_path + ' does not exsits.')
    print('Starting the training...')
    net = sn
    if FLAGS.net == 'snbw':
        net = snbw
    elif FLAGS.net == 'snxnor':
        net = snxnor
    run_training(net=net)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'net',
        type=str,
        default='sn',
        help='The Network to be run, sn, snbw, snxnor'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        default='/tmp/tensorflow/sketch-a-net/',
        help='Directory to save the logs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='The initial learning rate for the optimizer'
    )
    parser.add_argument(
        '--decay_step',
        type=float,
        default=250,
        help='The decay step for exponential decay learning rate'
    )
    parser.add_argument(
        '--decay_rate',
        type=float,
        default=0.90,
        help='The decay rate for exponential decay learning rate'
    )
    parser.add_argument(
        '--dr',
        type=float,
        default=0.5,
        help='The probability to keep a neuron in dropout'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=54,
        help='The batch_size'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=500,
        help='epoch size, the number of times the trainer should use the dataset'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=1,
        help='top-k accuracy'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='dataset/dataset_with_order_info_256.mat',
        help='The .mat file with the dataset downloaded from http://www.eecs.qmul.ac.uk/~tmh/downloads.html'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='dataset/model_with_order_info_256.mat',
        help='The .mat file with the pretrained weights downloaded from http://www.eecs.qmul.ac.uk/~tmh/downloads.html'
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--no_pretrain',
        action='store_false',
        default=True
    )
    parser.add_argument(
        '--reset_step',
        action='store_true',
        default=False
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
