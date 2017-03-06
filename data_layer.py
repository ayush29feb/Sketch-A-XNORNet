import numpy as np
import h5py
from scipy.ndimage.interpolation import rotate
import scipy.io as sio
import os.path
import logging

class DataLayer:
    """
    This is an abstraction of the data layer for the Sketch-A-Net Dataset.
    It allows us to feed the network data easily in batches with allows
    the nauances of the data-augmentation like random crops, flips, etc.
    """

    NUM_CLASSES = 250
    NUM_ITEMS_PER_CLASS = 80
    NUM_TRAIN_ITEMS_PER_CLASS = 54
    NUM_TEST_ITEMS_PER_CLASS = 26
    NUM_VALIDATION_COPIES = 10
    
    INPUT_SIZE = 256
    OUTPUT_SIZE = 225
    NUM_CHANNELS = 6

    def __init__(self, filepath, batch_size=135):
        """
        Initializes the variables and loads the data file
        """
        # initialize the cursors to keep track where we are in the Dataset
        self.train_cursor = 0
        self.test_cursor = 0
        self.train_batch_size = batch_size
        self.test_batch_size = batch_size // 10

        # initialize the idx arrays
        a_train_ = np.tile(np.arange(self.NUM_TRAIN_ITEMS_PER_CLASS), self.NUM_CLASSES).reshape(self.NUM_CLASSES, self.NUM_TRAIN_ITEMS_PER_CLASS)
        b_train_ = np.tile(np.arange(self.NUM_CLASSES) * self.NUM_ITEMS_PER_CLASS, self.NUM_TRAIN_ITEMS_PER_CLASS).reshape(self.NUM_TRAIN_ITEMS_PER_CLASS, self.NUM_CLASSES).T
        self.train_idxs = (a_train_ + b_train_).reshape(-1)

        a_test_ = np.tile(np.arange(self.NUM_TEST_ITEMS_PER_CLASS), self.NUM_CLASSES).reshape(self.NUM_CLASSES, self.NUM_TEST_ITEMS_PER_CLASS)
        b_test_ = np.tile(np.arange(self.NUM_CLASSES) * self.NUM_ITEMS_PER_CLASS, self.NUM_TEST_ITEMS_PER_CLASS).reshape(self.NUM_TEST_ITEMS_PER_CLASS, self.NUM_CLASSES).T
        self.test_idxs = (a_test_ + b_test_ + self.NUM_TRAIN_ITEMS_PER_CLASS).reshape(-1)

        # load the .mat file containing the dataset
        print('Loading the dataset...')
        data = h5py.File(filepath)
        self.dataset_images = data['imdb']['images']['data']
        self.dataset_labels = data['imdb']['images']['labels']
        print('Dataset loaded!')

    def get_images_shape():
        """Returns the shape of images returned by next_batch_train
        """
        return (self.batch_size, self.OUTPUT_SIZE, self.OUTPUT_SIZE, self.NUM_CHANNELS)
    
    def next_batch_train(self, batch_size=None):
        """
        Returns the next batch for the training data with the requested batch_size
        or the current default. This function takes care of all the data augmentation
        techniques.

        Args:
            batch_size: the number of items requested
        
        Returns:
            images: an ndarray of shape (batch_size, OUTPUT_SIZE, OUTPUT_SIZE, NUM_CHANNELS)
            labels: an ndarray of shape (batch_size)
        """

        # set the batch_size and output_size to class default
        if batch_size is None:
            batch_size = self.train_batch_size
        output_size = self.OUTPUT_SIZE
        input_size = self.INPUT_SIZE

        # create an array of indicies to retrieve
        idxs = self.train_idxs[self.train_cursor:self.train_cursor+batch_size]
        if self.train_cursor+batch_size >= self.train_idxs.size:
            idxs = np.append(idxs, self.train_idxs[:(self.train_cursor+batch_size - self.train_idxs.size)])

        # retrieve the images and labels
        labels = self.dataset_labels[idxs, :].reshape(-1)
        images_raw = self.dataset_images[idxs, :, :, :].swapaxes(1, 3)

        # apply data augmentation
        images = np.zeros((batch_size, output_size, output_size, images_raw.shape[3]))
        x = np.random.randint(input_size - output_size, size=batch_size)
        y = np.random.randint(input_size - output_size, size=batch_size)
        flip = np.random.rand(batch_size) > 0.45
        degs = (np.random.rand(batch_size) > 0.45) * (np.random.randint(11, size=batch_size) - 5.0)

        # TODO: vectorize data augmentation
        for i in xrange(batch_size):
            images[i, :, :, :] = images_raw[i, x[i]:x[i]+output_size, y[i]:y[i]+output_size, :]
            if flip[i]:
                images[i, :, :, :] = np.fliplr(images[i, :, :, :])
            if degs[i] != 0:
                images[i, :, :, :] = rotate(images[i, :, :, :], degs[i], cval=255.0, reshape=False)

        # move the cursors
        self.train_cursor = (self.train_cursor + batch_size) % (self.NUM_TRAIN_ITEMS_PER_CLASS * self.NUM_CLASSES)

        return (255 - images, labels - 1)

    def next_batch_test(self, batch_size=None):
        """
        Returns the next batch for the test data with the requested batch_size
        or the current default. This function takes care of all the data augmentation
        techniques.

        Args:
            batch_size: the number of items requested
        
        Returns:
            images: an ndarray of shape (batch_size * 10, OUTPUT_SIZE, OUTPUT_SIZE, NUM_CHANNELS)
            labels: an ndarray of shape (batch_size * 10) ranging from [0, 249]
        """

        # set the batch_size and output_size to class default
        if batch_size is None:
            batch_size = self.test_batch_size
        output_size = self.OUTPUT_SIZE
        input_size = self.INPUT_SIZE

         # create an array of indicies to retrieve
        idxs = self.test_idxs[self.test_cursor:self.test_cursor+batch_size]
        if self.test_cursor+batch_size >= self.test_idxs.size:
            idxs = np.append(idxs, self.test_idxs[:(self.test_cursor+batch_size - self.test_idxs.size)])

        # retrieve the images and labels & apply data augmentation
        labels = np.tile(self.dataset_labels[idxs, :].reshape(-1), 10)
        images_raw = self.dataset_images[idxs, :, :, :].swapaxes(1, 3)
        images = np.concatenate((images_raw[:, 0:output_size, 0:output_size, :],
                            images_raw[:, input_size-output_size:input_size+1, 0:output_size, :],
                            images_raw[:, 0:output_size, input_size-output_size:input_size+1, :],
                            images_raw[:, input_size-output_size:input_size+1, input_size-output_size:input_size+1, :],
                            images_raw[:, (input_size-output_size+1)/2:input_size - (input_size - output_size + 1) / 2 + 1,
                                        (input_size-output_size+1)/2:input_size - (input_size - output_size + 1) / 2 + 1, :]), 
                            axis=0)
        images = np.concatenate((images, np.fliplr(images)), axis=0)

        # move the cursors
        self.test_cursor = (self.test_cursor + batch_size) % (self.NUM_TEST_ITEMS_PER_CLASS * self.NUM_CLASSES)

        return (255.0 - images, labels - 1)

def load_pretrained_model(filepath):
    """
    Loads the pretrained weights and biases from the pretrained model available
    on http://www.eecs.qmul.ac.uk/~tmh/downloads.html

    Args:
        Takes in the filepath for the pretrained .mat filepath
    
    Returns:
        Returns the dictionary with all the weights and biases for respective layers
    """
    if filepath is None or not os.path.isfile(filepath):
        print 'Pretrained Model Not Available!'
        return None, None

    data = sio.loadmat(filepath)
    weights = {}
    biases = {}
    conv_idxs = [0, 3, 6, 8, 10, 13, 16, 19]
    for i, idx in enumerate(conv_idxs):
        weights['conv' + str(i + 1)] = data['net']['layers'][0][0][0][idx]['filters'][0][0]
        biases['conv' + str(i + 1)] = data['net']['layers'][0][0][0][idx]['biases'][0][0].reshape(-1)
    
    print('Pretrained Model Loaded!')
    return (weights, biases)
