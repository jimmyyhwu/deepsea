from datetime import datetime
import os
import sys
import time

import numpy as np
import tensorflow as tf
import deepsea_input
import deepsea_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.app.flags.DEFINE_string('train_dir', 'deepsea_train', 'directory to read model checkpoints')
tf.app.flags.DEFINE_string('eval_dir', 'deepsea_eval', 'directory to write event logs')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size')
tf.app.flags.DEFINE_integer('global_step', -1, 'global step of model to evaluate')
tf.app.flags.DEFINE_string('split', 'val', 'eval data split: val or test')
tf.app.flags.DEFINE_integer('eval_interval_secs', 1000, 'how often to run eval')
tf.app.flags.DEFINE_boolean('run_once', False, 'whether to run eval only once')
tf.app.flags.DEFINE_boolean('report_progress', False, 'whether to report progress during eval')
tf.app.flags.DEFINE_boolean('save_predictions', False, 'whether to save predictions to npy file')


def _progress(curr, end, message):
    sys.stdout.write('\r>> %s %.1f%%' % (message, float(curr) / float(end) * 100.0))
    sys.stdout.flush()


def _eval(saver, summary_writer, summary_op, logits, mean, auc, report_progress=False):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if FLAGS.global_step > 0:
            ckpt_path = os.path.join(FLAGS.train_dir, 'model.ckpt-%d' % FLAGS.global_step)
            if len(tf.gfile.Glob(ckpt_path + '*')) > 0:
                saver.restore(sess, ckpt_path)
                global_step = FLAGS.global_step
            else:
                print('Checkpoint for step %d not found' % FLAGS.global_step)
                return
        else:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoints found')
                return

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_examples = deepsea_input.NUM_VAL_EXAMPLES
            if FLAGS.split == 'test':
                num_examples = deepsea_input.NUM_TEST_EXAMPLES
            num_steps = num_examples / FLAGS.batch_size

            if FLAGS.save_predictions:
                predictions = tf.nn.sigmoid(logits)
                all_predictions = np.zeros(
                    (num_examples, deepsea_model.NUM_OUTPUTS), dtype=np.float32)

            step = 0
            update_ops = tf.get_collection('update_ops')
            sess.run(tf.local_variables_initializer())
            while step < num_steps and not coord.should_stop():
                if FLAGS.save_predictions:
                    pred_vals, _ = sess.run([predictions, update_ops])
                    start, end = FLAGS.batch_size*step, FLAGS.batch_size*(step + 1)
                    all_predictions[start:end, :] = pred_vals
                else:
                    sess.run(update_ops)
                step += 1
                if report_progress:
                    _progress(step, num_steps, 'Running eval on %s data' % FLAGS.split)
            if report_progress:
                print

            mean_cross_entropy_loss = sess.run(mean)
            overall_auc = sess.run(auc)
            fmt_str = ('%s: eval on %s data, checkpoint at step %s, '
                       'mean cross entropy loss = %.3f, overall auc = %.3f')
            print(fmt_str % (
                datetime.now(), FLAGS.split, global_step, mean_cross_entropy_loss, overall_auc))

            if not FLAGS.run_once:
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(
                    tag='mean_cross_entropy_loss', simple_value=mean_cross_entropy_loss)
                summary.value.add(tag='overall_auc', simple_value=overall_auc)
                summary_writer.add_summary(summary, global_step)

            if FLAGS.save_predictions:
                np.save(
                    os.path.join(FLAGS.eval_dir, '%s-predictions-%s' % (FLAGS.split, global_step)),
                    all_predictions)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    assert FLAGS.split == 'val' or FLAGS.split == 'test', 'split must be either val or test'
    with tf.Graph().as_default() as g:
        seqs, labels = deepsea_input.inputs(
            FLAGS.split, FLAGS.data_dir, FLAGS.batch_size, shuffle=False)
        logits = deepsea_model.build_model(seqs, FLAGS.batch_size, training=False)
        with tf.variable_scope('metrics'):
            mean = deepsea_model.mean_cross_entropy_loss(logits, labels)
            auc = deepsea_model.auc(logits, labels)

        saver = tf.train.Saver()
        summary_op = None
        summary_writer = None
        if not FLAGS.run_once:
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
        while True:
            _eval(saver, summary_writer, summary_op,
                  logits, mean, auc, report_progress=FLAGS.report_progress)
            if FLAGS.run_once:
                return
            time.sleep(FLAGS.eval_interval_secs)


def main(_):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
