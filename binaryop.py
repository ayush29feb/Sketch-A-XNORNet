import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def binarize_weights(x, name=None):
    """Creates the binarize_weights Op with f as forward pass
    and df as the gradient for the backward pass

    Args:
        x: The input Tensor
        name: the name for the Op
    
    Returns:
        The output tensor
    """
    def f(x):
        alpha = np.abs(x).sum(0).sum(0).sum(0) / x[:,:,:,0].size
        y = np.sign(x)
        y[y == 0] = 1
        return y * alpha

    def df(op, grad):
        x = op.inputs[0]
        n = tf.reduce_prod(tf.shape(x[:,:,:,0])[:3])
        alpha = tf.div(tf.reduce_sum(tf.abs(x), [0, 1, 2]), tf.cast(n, tf.float32))
        ds = tf.multiply(x, tf.cast(tf.less_equal(tf.abs(x), 1), tf.float32))
        return tf.multiply(grad, tf.add(tf.cast(1/n, tf.float32), tf.multiply(alpha, ds)))
        
    with ops.name_scope(name, 'BinarizeWeights', [x]) as name:
        fx = py_func(f, [x], [tf.float32], name=name, grad=df)
        return fx[0]

def binary_activation(x, name=None):
    """Creates the binary_activation Op with f as forward pass
    and fd as the gradient for the backward pass

    Args:
        x: The input Tensor
        name: the name for the Op

    Returns:
        The output tensor
    """
    def f(x):
        y = np.sign(x)
        y[y == 0] = 1
        return y
    
    def df(op, grad):
        x = op.inputs[0]
        alpha = tf.cast(tf.less_equal(tf.abs(x), 1), tf.float32) # alpha = (|x| <= 1) * 1
        return tf.multiply(grad, alpha) # grad * alpha
    
    with ops.name_scope(name, 'BinarizeInputs', [x]) as name:
        fx = py_func(f, [x], [tf.float32], name=name, grad=df)
        return fx[0]
