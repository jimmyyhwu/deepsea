import math
import os
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import h5py
import scipy.io

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.app.flags.DEFINE_integer('num_train_shards', 1000, 'number of train data tfrecord shards')

DATA_URL = 'http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz'


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _progress(curr, end, message):
    sys.stdout.write('\r>> %s %.1f%%' % (message, float(curr) / float(end) * 100.0))
    sys.stdout.flush()


def convert_train(train_seqs, train_labels):
    _, _, num_examples = train_seqs.shape
    assert(num_examples == train_labels.shape[1])

    filename = 'train.tfrecord'
    num_examples_per_shard = int(math.ceil(float(num_examples) / FLAGS.num_train_shards))
    for shard_id in xrange(FLAGS.num_train_shards):
        output_filename = '%s-%.5d-of-%.5d' % (filename, shard_id, FLAGS.num_train_shards)
        output_file = os.path.join(FLAGS.data_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        start_idx = shard_id * num_examples_per_shard
        for i in xrange(start_idx, min(num_examples, start_idx + num_examples_per_shard)):
            seq = train_seqs[:, :, i]
            label = train_labels[:, i]
            example = tf.train.Example(features=tf.train.Features(feature={
                'seq_raw': _bytes_feature(seq.tostring()),
                'label_raw': _bytes_feature(label.tostring())}))
            writer.write(example.SerializeToString())
            _progress(i + 1, num_examples, 'Writing %s' % filename)
        writer.close()
    print


def convert_val_test(seqs, labels, split):
    num_examples, _, _ = seqs.shape
    assert(num_examples == labels.shape[0])

    filename = os.path.join(FLAGS.data_dir, split + '.tfrecord')
    writer = tf.python_io.TFRecordWriter(filename)
    for i in xrange(num_examples):
        seq = seqs[i, :, :].T
        label = labels[i, :]
        example = tf.train.Example(features=tf.train.Features(feature={
            'seq_raw': _bytes_feature(seq.tostring()),
            'label_raw': _bytes_feature(label.tostring())}))
        writer.write(example.SerializeToString())
        _progress(i + 1, num_examples, filename)
    writer.close()

    np.save(os.path.join(FLAGS.data_dir, split + '.npy'), labels)
    print


def maybe_download_and_extract():
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def reporthook(count, block_size, total_size):
            _progress(count*block_size, total_size, 'Downloading %s' % filename)
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, reporthook)
        print
    extracted_dir_path = os.path.join(dest_directory, 'deepsea_train')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    maybe_download_and_extract()
    h5f = h5py.File(os.path.join(FLAGS.data_dir, 'deepsea_train/train.mat'), 'r')
    convert_train(h5f['trainxdata'], h5f['traindata'])
    vmat = scipy.io.loadmat(os.path.join(FLAGS.data_dir, 'deepsea_train/valid.mat'))
    convert_val_test(vmat['validxdata'], vmat['validdata'], 'val')
    tmat = scipy.io.loadmat(os.path.join(FLAGS.data_dir, 'deepsea_train/test.mat'))
    convert_val_test(tmat['testxdata'], tmat['testdata'], 'test')


if __name__ == '__main__':
    tf.app.run()
