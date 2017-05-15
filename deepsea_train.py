from datetime import datetime
import time

import tensorflow as tf
import deepsea_input
import deepsea_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.app.flags.DEFINE_string(
    'train_dir', 'deepsea_train', 'directory to store event logs and checkpoints')
tf.app.flags.DEFINE_integer('seed', 0, 'random seed')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size')
tf.app.flags.DEFINE_float('lr', 1e-2, 'learning rate')
tf.app.flags.DEFINE_float('lr_decay', 8e-7, 'learning rate decay')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum')
tf.app.flags.DEFINE_float('lambda3', 0.9, 'max kernel norm')
tf.app.flags.DEFINE_integer('log_frequency', 10, 'how often to log results to the console')
tf.app.flags.DEFINE_integer('max_epochs', 100, 'max number of training epochs')
tf.app.flags.DEFINE_integer('max_to_keep', 100, 'number of recent checkpoints to keep')


def _build_train_op(total_loss, global_step):
    lr = tf.train.inverse_time_decay(FLAGS.lr, global_step, 1, FLAGS.lr_decay)
    tf.summary.scalar('learning_rate', lr)

    losses = tf.get_collection('losses')
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name, l)

    opt = tf.train.MomentumOptimizer(FLAGS.lr, FLAGS.momentum)
    grads = opt.compute_gradients(total_loss)
    train_op = opt.apply_gradients(grads, global_step=global_step)

    if FLAGS.lambda3 > 0:
        with tf.variable_scope('max_norm'):
            with tf.control_dependencies([train_op]):
                with tf.control_dependencies(
                    [deepsea_model.apply_max_norm(w, FLAGS.lambda3)
                     for w in tf.get_collection('weights')]):
                    max_norm_op = tf.no_op()
    else:
        max_norm_op = train_op

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    for grad, var in grads:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    return max_norm_op


def train():
    with tf.Graph().as_default():
        tf.set_random_seed(FLAGS.seed)
        global_step = tf.contrib.framework.get_or_create_global_step()
        seqs, labels = deepsea_input.inputs('train', FLAGS.data_dir, FLAGS.batch_size)
        logits = deepsea_model.build_model(seqs, FLAGS.batch_size)
        total_loss = deepsea_model.loss(logits, labels)
        train_op = _build_train_op(total_loss, global_step)
        num_batches_per_epoch = deepsea_input.NUM_TRAIN_EXAMPLES / FLAGS.batch_size

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._start_time = time.time()

            def before_run(self, run_context):
                return tf.train.SessionRunArgs([total_loss, global_step])

            def after_run(self, run_context, run_values):
                total_loss_value, global_step_value = run_values.results
                if global_step_value % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    fmt_str = '%s: step %d, total loss = %.3f (%.1f examples/sec; %.3f sec/batch)'
                    print(fmt_str % (datetime.now(), global_step_value,
                                     total_loss_value, examples_per_sec, sec_per_batch))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            config=config,
            save_checkpoint_secs=None,
            save_summaries_steps=1000,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_epochs * num_batches_per_epoch),
                   tf.train.NanTensorHook(total_loss),
                   tf.train.CheckpointSaverHook(FLAGS.train_dir,
                                                save_steps=10000,
                                                saver=saver),
                   _LoggerHook()]) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(_):
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
