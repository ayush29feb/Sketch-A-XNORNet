import numpy as np
from scipy import ndimage
from scipy.misc import imread, imresize
import cPickle as pickle
import os
import scipy.io
import random

# packs up all the pngs in the category into a single file
# Preconditions:
# - there are only images inside of all the files
# Postconditions:
# - all images will be flattened out, and should be reshaped
#   by the client to (1111,1111)
def prep_data_cat(path, cat, size=(64, 64)):
    fnames = os.listdir(os.path.join(path, cat))

    X = np.empty(tuple([0] + list(size)))
    y = 1 * (np.array(sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])) == cat)
    y = np.tile(y, (len(fnames), 1))
    for fname in fnames:
        img = ndimage.imread(os.path.join(path, cat, fname))
        img = imresize(img, size)
        X = np.vstack((X, img.reshape(tuple([1] + list(img.shape)))))

    return (X, y)

# packs up all the pngs into multiple batch files
def prep_data(path, train=0.8, test=0.1):    
    cats = sorted(os.listdir(path))

    Xs_train = []
    ys_train = []
    Xs_val = []
    ys_val = []
    Xs_test = []
    ys_test = []
    for cat in cats:
        X_cat, y_cat = prep_data_cat(path, cat)
        train_idx = (int) (X_cat.shape[0] * train)
        test_idx = (int) (X_cat.shape[0] * (1 - test))
        
        Xs_train.append(X_cat[:train_idx, :, :])
        ys_train.append(y_cat[:train_idx, :])
        Xs_val.append(X_cat[train_idx:test_idx, :, :])
        ys_val.append(y_cat[train_idx:test_idx, :])
        Xs_test.append(X_cat[test_idx:, :, :])
        ys_test.append(y_cat[test_idx:, :])

    X_train, y_train = np.concatenate(Xs_train), np.concatenate(ys_train)
    X_val, y_val = np.concatenate(Xs_val), np.concatenate(ys_val)
    X_test, y_test = np.concatenate(Xs_test), np.concatenate(ys_test)

    return (X_train, y_train, X_val, y_val, X_test, y_test)

def load_data(filename):
    cats = sorted(os.listdir(path))
    X, y = None, None
    with open(filename, 'rb') as f:
        X, y = pickle.load(f)
    return (X, y)
