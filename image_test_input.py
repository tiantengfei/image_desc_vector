import sys
import tensorflow as tf
import os

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 332145
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1770

FLAGS = tf.app.flags.FLAGS


def read_record(file_queue):
    class ImageRecord:
        pass

    result = ImageRecord()
    result.height = 128
    result.width = 64
    result.depth = 3
    label_bytes = 1
    image_name_bytes = 4
    image_bytes = result.height * result.width * result.depth
    record_bytes = image_bytes + image_name_bytes * 2 + label_bytes

    reader = tf.FixedLengthRecordReader(record_bytes)
    result.key, value = reader.read(file_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    num = tf.constant([48])

    result.label = tf.sub(result.label, num)
    result.image_name_1 = tf.cast(
        tf.slice(record_bytes, [label_bytes], [image_name_bytes]),
        tf.uint8
    )

    result.image_name_2 = tf.cast(
        tf.slice(record_bytes, [label_bytes + image_name_bytes], [image_name_bytes]),
        tf.uint8
    )

    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes + image_name_bytes * 2], [image_bytes]),
                             [result.height, result.width, result.depth])

    result.uint8image = depth_major  # tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, name_1, name_2, min_queue_examples,
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
        images, label_batch, names_1, names_2 = tf.train.batch(
            [image, label, name_1, name_2],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size]), \
           tf.reshape(names_1, [batch_size, 4]), tf.reshape(names_2, [batch_size, 4])


def inputs_new(eval_data, data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'image_test.bin')]
    file_queue = tf.train.string_input_producer(filenames)

    read_input = read_record(file_queue)

    width = 112
    height = 56

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)
    float_image = tf.image.per_image_whitening(resized_image)
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           read_input.image_name_1, read_input.image_name_2,
                                           min_queue_examples, batch_size,
                                           shuffle=False)


def inputs(eval_data):
    #if n
    # ot FLAGS.data_dir:
        #raise ValueError('Please supply a data_dir')
    data_dir = os.path.join('image_train', 'image_test')
    return inputs_new(eval_data=eval_data, data_dir=data_dir, batch_size=128)
                      #batch_size=FLAGS.batch_size)
