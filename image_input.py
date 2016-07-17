from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

IMAGE_SIZE = 64

NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 40000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1770


def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()
    label_dim = 300
    label_bytes = label_dim * 4
    result.height = 64
    result.width = 64
    result.depth = 3
    image_bytes = result.height * result.width * result.depth

    record_bytes = label_bytes + image_bytes

    # reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # result.key, value = reader.read(filename_queue)
    #
    #
    # record_bytes = tf.decode_raw(value, tf.uint8)
    # #record_bytes = value
    #
    # # label = tf.cast(
    # #     tf.slice(record_bytes, [0], [label_bytes]), tf.float32)
    # #
    # # result.label = tf.reshape(label, [label_dim])
    # #num = tf.constant([48])
    #
    # #result.label = tf.sub(result.label, num)
    #
    # c = (value.tolist())[:label_bytes]
    # label = np.fromstring(value[:label_bytes], dtype=np.float32).reshape(300)
    # result.label = tf.constant(label)

    reader = tf.TFRecordReader()
    _, serializded_example = reader.read(filename_queue)
    features = tf.parse_single_example(serializded_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'image_raw': tf.FixedLenFeature([], tf.string)
                                       })

    image = tf.decode_raw(features['image_raw'], tf.uint8)

    depth_major = tf.reshape(image,
                             [result.height, result.width, result.depth])

    result.uint8image = depth_major  # tf.transpose(depth_major, [1, 2, 0])

    label_raw = tf.decode_raw(features['label'], tf.float32)

    result.label = tf.reshape(label_raw,
                             [label_dim])
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    #return images, tf.reshape(label_batch, [batch_size, 300])
    return images, label_batch


def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'image_%d.tfrecords' % i)
                 for i in xrange(0, 1)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = 50
    width = 50

    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    distorted_image = tf.image.random_flip_left_right(distorted_image)

    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    float_image = tf.image.per_image_whitening(distorted_image)

    min_fraction_of_examples_in_queue = 0.03
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


