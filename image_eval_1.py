from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import image
import image_test_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/ttf/Desktop/after_train/image_25_eval_64',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/ttf/Desktop/after_train/image_25_train_64',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 10,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 1770,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op, logits, labels, names_1, names_2):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            num = 0
            error_num_label0 = 0
            error_num_label1 = 0
            label_0_wrong_list = []
            label_1_wrong_list = []
            while step < num_iter and not coord.should_stop():
                predictions, logits_new, labels_new, names_1_new, names_2_new = \
                    sess.run([top_k_op, logits, labels, names_1, names_2])
                true_count += np.sum(predictions)
                step += 1

                for pre, logit, label, name_1, name_2 in zip(predictions, logits_new, labels_new,
                                                             names_1_new, names_2_new):
                    if not pre:
                        num += 1
                        if label == 0:
                            error_num_label0 += 1
                            label_0_wrong_list.append((name_1.tostring(), name_2.tostring()))
                        else:
                            error_num_label1 += 1
                            label_1_wrong_list.append((name_1.tostring(), name_2.tostring()))
                        print('error: pre is {}, prediction is {}, but label is {}. images are {}'
                              'and {}'.format(
                            pre, logit, label, name_1.tostring(), name_2.tostring()
                        ))

            all_wrong_msg = {'label_0_wrong': label_0_wrong_list, 'label_1_wrong': label_1_wrong_list}
            with open('all_wrong_msg.json', 'wb') as f:
                json.dump(all_wrong_msg, f)
            print('total error number is {}'.format(num))
            print('error label_0 number is {}'.format(error_num_label0))
            print('error label_1 number is {}'.format(error_num_label1))
            print('total true number is {}'.format(true_count))
            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    with tf.Graph().as_default():

        eval_data = FLAGS.eval_data == 'test'
        images, labels, names_1, names_2 = image_test_input.inputs(eval_data=eval_data)

        logits = image.inference(images)

        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        variable_averages = tf.train.ExponentialMovingAverage(
            image.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        summary_op = tf.merge_all_summaries()

        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                                graph_def=graph_def)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op, logits, labels, names_1, names_2)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument

    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
