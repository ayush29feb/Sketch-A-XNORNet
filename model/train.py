import tensorflow as tf
import model as cnn
from data import placeholder_data

import time

BATCH_SIZE = 100
IMAGE_HEIGHT = 225
IMAGE_WIDTH = 225
IMAGE_CHANNELS = 6

LEARNING_RATE = 0.1
MAX_STEPS = 2000
LOG_DIR = '/tmp/sketchnet/'

def fill_feed_dict(images_pl, labels_pl):
  images_feed, labels_feed = placeholder_data()
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

def run_training():
    with tf.Graph().as_default():
        # placeholders for images and labels
        images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        labels_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE))

        # Get the logits from the inference graph a.k.a. network
        logits = cnn.inference(images_placeholder)
        
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
            start_time = time.time()
            
            feed_dict = fill_feed_dict(images_placeholder, labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

def main():
    run_training()

if __name__ == '__main__':
    main()