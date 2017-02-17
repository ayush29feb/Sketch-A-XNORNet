import numpy as np
import tensorflow as tf
import h5py
from random import randint, uniform
from scipy.misc import imrotate

class DataLayer:
    """
    This is an abstraction of the data layer for the Sketch-A-Net.
    It allows us to feed the network data easily in batches with allows
    the nauances of the data-augmentation like random crops, flips, etc.
    """

    def __init__(self, filepath, batch_size=135):
        """
        Loads the mat file into images and labels.
        """
        print 'Loading the dataset'
        self.batch_size = batch_size
        self.train_cursor = 0

        data = h5py.File(filepath)
        sets = data['imdb']['images']['set'][()].reshape(-1)
        self.X_train = data['imdb']['images']['data'][sets == 1, :, :, :].swapaxes(1, 3)
        self.y_train = data['imdb']['images']['labels'][sets == 1, :].reshape(-1)
        self.X_test = data['imdb']['images']['data'][sets == 3, :, :, :].swapaxes(1, 3)
        self.y_test = data['imdb']['images']['labels'][sets == 3, :].reshape(-1)

        print 'Data has been loaded'
        
    def next_batch(batch_size=None):
        """
        Returns the next batch for the training data with the requested batch_size
        or the current default. This function takes care of all the data augmentation
        techniques.

        Args:
            batch_size: the number of items requested
        
        Returns:
            images: an ndarray of shape (batch_size, 225, 225, 6)
        """
        # declare index and shape variables
        if batch_size is None:
            batch_size = self.batch_size

        N = self.X_train.shape[0]
        input_size = self.X_train.shape[1]
        cursor = self.train_cursor % N
        start_idx = cursor
        end_idx = cursor + batch_size
        output_size = 225 # output size

        # create the current batch in X, y
        X, y = None, None
        if end_idx < N:
            X = self.X_train[start_idx:end_idx, :, :, :]
            y = self.y_train[start_idx:end_idx]
        else:
            end_idx = end_idx % N
            X = np.concatenate((self.X_train[start_idx:, :, :, :], self.X_train[:end_idx, :, :, :]), axis=0)
            y = np.concatenate((self.y_train[start_idx:], self.y_train[:end_idx]), axis=0)
        
        # Apply the data augmentation for each data point
        for i in range(batch_size):
            # randomly crop the image to be the output size
            x_ = randint(0, input_size - output_size + 1)
            y_ = randint(0, input_size - output_size + 1)
            X[i, :, :, :] = X[i, x_:x_ + output_size - 1, y_:y_ + output_size - 1, :]

            # randomly rotate the image
            if uniform(0, 1) > 0.45:
                X[i, :, :, :] = rotate(X[i, :, :, :], 5, cval=255.0)

            # randomly flip the image horizontaly
            if uniform(0, 1) > 0.45:
                X[i, :, :, :] = np.fliplr(X[i, :, :, :])
        
        # Move the train cursor
        train_cursor += batch_size
        train_cursor = train_cursor % N

        return (X, y)

    def test_set():
        """
        Returns the entire test set with after applying data augmentation

        Returns:
            A tuple with the (X, y) containing the testing images and labels
        """
        X = X_test
        X = np.concatenate((X[:, 1:225, 1:225, :], X[:, 32:256, 1:225, :], X[:, 1:225, 32:256, :], X[:, 32:256, 32:256, :], X[:, 16:240, 16:240, :]), axis=0)
        X = np.concatenate((X, np.fliplr(X)), axis=0)
        y = np.tile(y_test, 10)

        return (X, y)