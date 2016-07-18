"""A library to train Inception using multiple replicas with synchronous update.
Please see accompanying README.md for details and instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import image

import numpy as np
import tensorflow as tf
import image_input
import slim

IMAGE_SIZE = image_input.IMAGE_SIZE
NUM_CLASSES = image_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = image_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = image_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the sync_replicas_optimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 180,
                            'Save summaries interval seconds.')

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# Learning rate decay factor selected from https://arxiv.org/abs/1604.00981
tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                          'Learning rate decay factor.')

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.


def train():
    assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')

    tf.logging.info('PS hosts are %s ' % ps_hosts)
    tf.logging.info('Worker hosts are %s ' % worker_hosts)
    print('PS hosts are %s ' % ps_hosts)
    print('Worker hosts are %s ' % worker_hosts)

    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
                                         'worker': worker_hosts})

    server = tf.train.Server(cluster_spec, job_name=FLAGS.job_name,
                             task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        server.join()
    else:

        """Train Inception on a dataset for a number of steps."""
        # Number of workers and parameter servers are infered from the workers and ps
        # hosts string.
        num_workers = len(cluster_spec.as_dict()['worker'])
        num_parameter_servers = len(cluster_spec.as_dict()['ps'])
        # If no value is given, num_replicas_to_aggregate defaults to be the number of
        # workers.
        if FLAGS.num_replicas_to_aggregate == -1:
            num_replicas_to_aggregate = num_workers
        else:
            num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

            # Both should be greater than 0 in a distributed training.
            assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                                   'num_parameter_servers'
                                                                   ' must be > 0.')
        # Choose worker 0 as the chief. Note that any worker could be the chief
        # but there should be only one chief.
        is_chief = (FLAGS.task_id == 0)

        # Ops are assigned to worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d' % FLAGS.task_id,
                                                      cluster=cluster_spec)):
            # Variables and its related init/assign ops are assigned to ps.
            # with slim.scopes.arg_scope(
            # [slim.variables.variable, slim.variables.global_step],
            # device=slim.variables.VariableDeviceChooser(num_parameter_servers)):
            # Create a variable to count the number of train() calls. This equals the
            # number of updates applied to the variables.
            # global_step = slim.variables.global_step()
            global_step = tf.Variable(0, name='global_step', trainable=False)
            num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
            decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                            global_step,
                                            decay_steps,
                                            LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            tf.scalar_summary('learning_rate', lr)
            opt = tf.train.GradientDescentOptimizer(lr)

            images, labels = image.distorted_inputs()
            logits = image.inference(images)
            total_loss = image.loss(logits, labels)

            # train_op = image.train(loss, global_step)

            if is_chief:
                loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
                losses = tf.get_collection('losses')
                loss_averages_op = loss_averages.apply(losses + [total_loss])

                for l in losses + [total_loss]:
                    # Name each loss as '(raw)' and name the moving average version of the loss
                    # as the original loss name.
                    tf.scalar_summary(l.op.name + ' (raw)', l)
                    tf.scalar_summary(l.op.name, loss_averages.average(l))
                with tf.control_dependencies([loss_averages_op]):
                    total_loss = tf.identity(total_loss)

            variable_averages = tf.train.ExponentialMovingAverage(
                MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = (tf.trainable_variables() + tf.moving_average_variables())

            for var in variables_averages_op:
                tf.histogram_summary(var.op.name, var)

            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=num_replicas_to_aggregate,
                replica_id=FLAGS.task_id,
                total_num_replicas=num_workers,
                variable_averages=variable_averages,
                variables_to_average=variables_averages_op)

            # batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
            # assert batchnorm_updates, 'Batchnorm updates are missing'
            # batchnorm_updates_op = tf.group(*batchnorm_updates)
            ## Add dependency to compute batchnorm_updates.
            # with tf.control_dependencies([batchnorm_updates_op]):
            #   total_loss = tf.identity(total_loss)

            # Compute gradients with respect to the loss.
            grads = opt.compute_gradients(total_loss)

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    tf.histogram_summary(var.op.name + '/gradients', grad)

            apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

            with tf.control_dependencies([apply_gradients_op]):
                train_op = tf.identity(total_loss, name='train_op')

            # Get chief queue_runners, init_tokens and clean_up_op, which is used to
            # synchronize replicas.
            # More details can be found in sync_replicas_optimizer.
            chief_queue_runners = [opt.get_chief_queue_runner()]
            init_tokens_op = opt.get_init_tokens_op()
            clean_up_op = opt.get_clean_up_op()

            # Create a saver.
            saver = tf.train.Saver()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()

            # Build an initialization operation to run below.
            init_op = tf.initialize_all_variables()

            # We run the summaries in the same thread as the training operations by
            # passing in None for summary_op to avoid a summary_thread being started.
            # Running summaries and training operations in parallel could run out of
            # GPU memory.
            sv = tf.train.Supervisor(is_chief=is_chief,
                                     logdir=FLAGS.train_dir,
                                     init_op=init_op,
                                     summary_op=None,
                                     global_step=global_step,
                                     saver=saver,
                                     save_model_secs=FLAGS.save_interval_secs)

            tf.logging.info('%s Supervisor' % datetime.now())

            sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement)

            # Get a session.
            sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

            # Start the queue runners.
            queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
            sv.start_queue_runners(sess, queue_runners)
            tf.logging.info('Started %d queues for processing input data.',
                            len(queue_runners))
            print('Started %d queues for processing input data.'%
                  len(queue_runners))
            if is_chief:
                sv.start_queue_runners(sess, chief_queue_runners)
                sess.run(init_tokens_op)

            # Train, checking for Nans. Concurrently run the summary operation at a
            # specified interval. Note that the summary_op and train_op never run
            # simultaneously in order to prevent running out of GPU memory.
            next_summary_time = time.time() + FLAGS.save_summaries_secs
            while not sv.should_stop():
                try:
                    start_time = time.time()
                    loss_value, step, logits_data, labels_data = sess.run([train_op, global_step, logits, labels])
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    if step > FLAGS.max_steps:
                        break
                    duration = time.time() - start_time

                    if step % 2 == 0:
                        examples_per_sec = FLAGS.batch_size / float(duration)
                        format_str = ('Worker %d: %s: step %d, loss = %.2f'
                                      '(%.1f examples/sec; %.3f  sec/batch)')
                        print('logits is {}, \n lables is {} '.format(logits_data[1], labels_data[1]))
                        tf.logging.info(format_str %
                                        (FLAGS.task_id, datetime.now(), step, loss_value,
                                         examples_per_sec, duration))
                        print(format_str %
                                        (FLAGS.task_id, datetime.now(), step, loss_value,
                                         examples_per_sec, duration))

                    # Determine if the summary_op should be run on the chief worker.
                    if is_chief and next_summary_time < time.time():
                        tf.logging.info('Running Summary operation on the chief.')
                        print('Running Summary operation on the chief.')
                        summary_str = sess.run(summary_op)
                        sv.summary_computed(sess, summary_str)
                        tf.logging.info('Finished running Summary operation.')
                        print('Finished running Summary operation.')

                        # Determine the next time for running the summary.
                        next_summary_time += FLAGS.save_summaries_secs
                except:
                    if is_chief:
                        tf.logging.info('About to execute sync_clean_up_op!')
                        print('About to execute sync_clean_up_op!')
                        sess.run(clean_up_op)
                    raise

            # Stop the supervisor.  This also waits for service threads to finish.
            sv.stop()

            # Save after the training ends.
            if is_chief:
                saver.save(sess,
                           os.path.join(FLAGS.train_dir, 'model.ckpt'),
                           global_step=global_step)
            print("end")


def main(_):  # pylint: disable=unused-argument

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
