import os

import tensorflow as tf

HEIGHT = 1
WIDTH = 1000
DEPTH = 4
NUM_OUTPUTS = 919
NUM_TRAIN_EXAMPLES = 4400000
NUM_VAL_EXAMPLES = 8000
NUM_TEST_EXAMPLES = 455024


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'seq_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string)
        })
    seq = tf.decode_raw(features['seq_raw'], tf.uint8)
    label = tf.decode_raw(features['label_raw'], tf.uint8)
    seq.set_shape([HEIGHT*WIDTH*DEPTH])
    seq = tf.reshape(seq, [HEIGHT, WIDTH, DEPTH])
    label.set_shape([NUM_OUTPUTS])
    seq = tf.cast(seq, tf.float32)
    label = tf.cast(label, tf.float32)
    return seq, label


def inputs(split, data_dir, batch_size, shuffle=True):
    filenames = sorted(tf.gfile.Glob(os.path.join(data_dir, split + '.tfrecord*')))
    num_train_shards = len(filenames)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
        seq, label = read_and_decode(filename_queue)
        if shuffle:
            min_queue_examples = NUM_TRAIN_EXAMPLES / num_train_shards
            seqs, labels = tf.train.shuffle_batch(
                [seq, label], batch_size=batch_size, num_threads=1,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            seqs, labels = tf.train.batch(
                [seq, label], batch_size=batch_size, num_threads=1,
                capacity=3 * batch_size)
    return seqs, labels
