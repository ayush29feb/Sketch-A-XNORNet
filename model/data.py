import numpy as np
import tensorflow as tf
import h5py
from random import randint, uniform
from scipy.ndimage.interpolation import rotate

class DataLayer:
    """
    This is an abstraction of the data layer for the Sketch-A-Net.
    It allows us to feed the network data easily in batches with allows
    the nauances of the data-augmentation like random crops, flips, etc.
    """

    def __init__(self, filepath):
        """
        Loads the mat file into images and labels.
        """
        print 'Loading the dataset'
        self.train_cursor = 0
        self.test_cursor = 0
        self.batch_size = 54

        data = h5py.File(filepath)
        self.sets = data['imdb']['images']['set'][()].reshape(-1)
        self.X_dataset = data['imdb']['images']['data'] # [sets == 1, :, :, :].swapaxes(1, 3)
        self.y_dataset = data['imdb']['images']['labels'] # [sets == 1, :].reshape(-1)
        # self.X_test = data['imdb']['images']['data'] # [sets == 3, :, :, :].swapaxes(1, 3)
        # self.y_test = data['imdb']['images']['labels'] # [sets == 3, :].reshape(-1)

        print 'Data has been loaded'
        
    def next_train_batch(self, batch_size=None):
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

        N = self.X_dataset.shape[0]
        input_size = self.X_dataset.shape[2]
        cursor = self.train_cursor % N
        start_idx = cursor
        end_idx = cursor + 54
        output_size = 225 # output size

        # create the current batch in X, y
        X, y = None, None
        if end_idx < N:
            X = self.X_dataset[start_idx:end_idx, :, :, :].swapaxes(1, 3) # self.X_train[start_idx:end_idx, :, :, :]
            y = self.y_dataset[start_idx:end_idx, :].reshape(-1) # self.y_train[start_idx:end_idx]
        else:
            end_idx = end_idx % N
            X = np.concatenate((self.X_dataset[start_idx:, :, :, :].swapaxes(1, 3), self.X_dataset[:end_idx, :, :, :].swapaxes(1, 3)), axis=0)
            y = np.concatenate((self.y_dataset[start_idx:].reshape(-1), self.y_dataset[:end_idx].reshape(-1)), axis=0)
        
        X_train, y_train = np.zeros((54, output_size, output_size, 6)), y
        # Apply the data augmentation for each data point
        for i in range(batch_size):
            # randomly crop the image to be the output size
            x_ = randint(0, input_size - output_size)
            y_ = randint(0, input_size - output_size)
            X_train[i, :, :, :] = X[i, x_:x_ + output_size, y_:y_ + output_size, :]

            # randomly rotate the image
            if uniform(0, 1) > 0.45:
                X_train[i, :, :, :] = rotate(X_train[i, :, :, :], 5, cval=255.0, reshape=False)

            # randomly flip the image horizontaly
            if uniform(0, 1) > 0.45:
                X_train[i, :, :, :] = np.fliplr(X_train[i, :, :, :])
        
        # Move the train cursor
        self.train_cursor += 80
        self.train_cursor = self.train_cursor % N

        return (255 - X_train, y_train - 1)

    def next_test_batch(self):
        """
        Returns the entire test set with after applying data augmentation

        Returns:
            A tuple with the (X, y) containing the testing images and labels
        """
        N = self.X_dataset.shape[0]
        start_idx = self.test_cursor + 54
        end_idx = self.test_cursor + 80

        X = self.X_dataset[start_idx:end_idx, :, :, :].swapaxes(1, 3)
        X = np.concatenate((X[:, 0:225, 0:225, :], X[:, 31:257, 0:225, :], X[:, 0:225, 31:257, :], X[:, 31:257, 31:257, :], X[:, 16:241, 16:241, :]), axis=0)
        X = np.concatenate((X, np.fliplr(X)), axis=0)
        y = np.tile(self.y_dataset[start_idx:end_idx, :].reshape(-1), 10)

        self.test_cursor = end_idx % N

        return (255 - X, y - 1)