import numpy as np
import scipy.io as sio

def load_weights_biases(filepath, optparams=False):
    """
    Loads the pretrained weights and biases from the pretrained model available
    on http://www.eecs.qmul.ac.uk/~tmh/downloads.html

    Args:
        Takes in the filepath for the pretrained .mat filepath
    
    Returns:
        Returns the dictionary with all the weights and biases for respective layers
    """
    print 'Loading the pretrained model'
    data = sio.loadmat(filepath)
    weights = {}
    biases = {}
    # weightsMomentum = {}
    # biasesMomentum = {}
    conv_idxs = [0, 3, 6, 8, 10, 13, 16, 19]
    for i, idx in enumerate(conv_idxs):
        weights['conv' + str(i + 1)] = data['net']['layers'][0][0][0][idx]['filters'][0][0]
        biases['conv' + str(i + 1)] = data['net']['layers'][0][0][0][idx]['biases'][0][0].reshape(-1)
        # weightsMomentum['conv' + str(i + 1)] = data['net']['layers'][0][0][0][idx]['weightsMomentum'][0,0]
        # biasesMomentum['conv' + str(i + 1)] = data['net']['layers'][0][0][0][idx]['biasesMomentum'][0,0].reshape(-1)
    
    print 'Weights have been loaded'
    return (weights, biases)

# Weights Learning Rate = 1
# Biases Learning Rate = 2
# filtersWeightDecay = 1
# biasesWeightDecay = 0