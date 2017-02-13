import tensorflow as tf
import model as cnn

BATCH_SIZE = 100
IMAGE_HEIGHT = 225
IMAGE_WIDTH = 225
IMAGE_CHANNELS = 6

LEARNING_RATE = 0.1
MAX_STEPS = 100

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

        init = tf.global_variable_initializer()
        sess = tf.Session()
        sess.run(init)

        for step in xrange(MAX_STEPS):
            feed_dict = None # ???
            
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

def main():
    run_training()

if __name__ == '__main__':
    main()