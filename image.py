# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tensorflow as tf
import image_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'image_train',
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = image_input.IMAGE_SIZE
NUM_CLASSES = image_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = image_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = image_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

TOWER_NAME = 'tower'


def _activation_summary(x):
    """ Output a Summary protocol buffer for output from per layer of the network.
    """
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """ if variable has been initialized, return it. Or else, initialize the variable.
    go to tf.get_variable for details.
    """

    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """if weights has been initialized, return it. Or else, initialize the weights. if wd is true,
    the loss function will include weight decay.

    Args:
        name: the name of weights' variable.
        shape: weight's shape.
        stddev: the standard deviation of the truncated normal distribution.
        wd: whether or not to include weight decay in the loss function.

    Returns:
        weight variable  correspond to the name.
    """
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'image-batches-bin')
    return image_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)


def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'image_test')
    return image_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=FLAGS.batch_size)


def inference(images):
    """ A convolutional neural network composites a few convolutional layers followed
    by a few full connected layers.

    Args:
        images: images used to train.

    Returns:
        output from final layer of the network.

    """
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 96],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 96, 96],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])

        weights = _variable_with_weight_decay('weights', shape=[dim, 768],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [768], tf.constant_initializer(0.1))
        local3 = tf.nn.sigmoid(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[768, 300],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [300], tf.constant_initializer(0.1))

        local4 = tf.matmul(local3, weights) + biases
        _activation_summary(local4)

    return local4


def loss(logits, labels):
    """ The loss function is used to optimize the neural network.
    Euclidean distance is used here.

    Args:
        logits: output from the network for a batch of images.
        labels: labels for a batch of images.

    Returns:
        The loss for the batch of images.

    """
    # Calculate the average cross entropy loss across the batch.

    loss_euclidean_distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(logits, labels)), 1))
    loss_mean = tf.reduce_mean(loss_euclidean_distance, name='loss_eunlidean_distance')
    tf.add_to_collection('losses', loss_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
