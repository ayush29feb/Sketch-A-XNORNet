import tensorflow as tf
import model as cnn
from data import DataLayer
from pretrain import load_weights_biases

import time

BATCH_SIZE = 54
IMAGE_HEIGHT = 225
IMAGE_WIDTH = 225
IMAGE_CHANNELS = 6

LEARNING_RATE = 0.1
MAX_STEPS = 2000
LOG_DIR = '/tmp/sketchnet/'
DATA_PATH = '../dataset/dataset_with_order_info_224.mat'

data_layer = DataLayer(DATA_PATH)

def fill_feed_dict(images_pl, labels_pl):
    print 'Preparing next training batch'
    images_feed, labels_feed = data_layer.next_batch()
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    print 'Next training batch is ready'
    return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def run_training():
    with tf.Graph().as_default():
        # placeholders for images and labels
        images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        labels_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE))

        # Load pretrained model
        pretrained = load_weights_biases('../dataset/model_with_order_info_224.mat')

        # Get the logits from the inference graph a.k.a. network
        logits = cnn.inference(images_placeholder, pretrained)
        
        # Get the loss
        loss = cnn.loss(logits, labels_placeholder)
        
        # Add the Ops that calculate and apply the gradients to the parameters
        train_op = cnn.training(loss, LEARNING_RATE)

        # Add the Op to compare the logits to the labels during evaluation
        eval_correct = cnn.evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        sess = tf.Session()
        
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        sess.run(init)

        for step in xrange(MAX_STEPS):
            print 'Step ' + str(step)
            start_time = time.time()
            
            feed_dict = fill_feed_dict(images_placeholder, labels_placeholder)
            _, loss_value, true_count = sess.run([train_op, loss, eval_correct], feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                print('Step %d: accuracy = %.2f (%.3f sec)' % (step, true_count, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

def main():
    run_training()

if __name__ == '__main__':
    main()