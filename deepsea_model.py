import math

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('stdv', 0, 'stddev for weight initialization')
tf.app.flags.DEFINE_float('lambda1', 5e-7, 'l2 regularization coefficient')
tf.app.flags.DEFINE_float('lambda2', 1e-8, 'l1 sparsity coefficient')

CONV1_OUT_CHANNELS = 320
CONV2_OUT_CHANNELS = 480
CONV3_OUT_CHANNELS = 960
NUM_OUTPUTS = 919


def _activation_summary(x):
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def _weight_variable_with_l2_loss(name, shape, wd):
    if FLAGS.stdv == 0:
        initializer = tf.contrib.layers.variance_scaling_initializer()
    else:
        initializer = tf.random_uniform_initializer(
            -math.sqrt(3)*FLAGS.stdv, math.sqrt(3)*FLAGS.stdv)
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    l2_loss = tf.multiply(tf.nn.l2_loss(var), wd, name='l2_loss')
    tf.add_to_collection('losses', l2_loss)
    return var


def _bias_variable(name, shape):
    if FLAGS.stdv == 0:
        initializer = tf.constant_initializer(0.0)
    else:
        initializer = tf.random_uniform_initializer(
            -math.sqrt(3)*FLAGS.stdv, math.sqrt(3)*FLAGS.stdv)
    return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)


def build_model(seqs, batch_size, training=True):
    with tf.variable_scope('conv1') as scope:
        kernel = _weight_variable_with_l2_loss(
            'weights', [1, 8, 4, CONV1_OUT_CHANNELS], FLAGS.lambda1)
        tf.add_to_collection('weights', kernel)
        conv = tf.nn.conv2d(seqs, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _bias_variable('biases', [CONV1_OUT_CHANNELS])
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        _activation_summary(conv1)

    pool1 = tf.nn.max_pool(
        conv1, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='VALID', name='pool1')
    dropout1 = tf.nn.dropout(pool1, 0.2 if training else 1, name='dropout1')

    with tf.variable_scope('conv2') as scope:
        kernel = _weight_variable_with_l2_loss(
            'weights', [1, 8, CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS], FLAGS.lambda1)
        tf.add_to_collection('weights', kernel)
        conv = tf.nn.conv2d(dropout1, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _bias_variable('biases', [CONV2_OUT_CHANNELS])
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        _activation_summary(conv2)

    pool2 = tf.nn.max_pool(
        conv2, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='VALID', name='pool2')
    dropout2 = tf.nn.dropout(pool2, 0.2 if training else 1, name='dropout2')

    with tf.variable_scope('conv3') as scope:
        kernel = _weight_variable_with_l2_loss(
            'weights', [1, 8, CONV2_OUT_CHANNELS, CONV3_OUT_CHANNELS], FLAGS.lambda1)
        tf.add_to_collection('weights', kernel)
        conv = tf.nn.conv2d(dropout2, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _bias_variable('biases', [CONV3_OUT_CHANNELS])
        conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        _activation_summary(conv3)

    dropout3 = tf.nn.dropout(conv3, 0.5 if training else 1, name='dropout3')

    with tf.variable_scope('fc4') as scope:
        reshape = tf.reshape(dropout3, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _weight_variable_with_l2_loss('weights', [dim, NUM_OUTPUTS], FLAGS.lambda1)
        tf.add_to_collection('weights', weights)
        biases = _bias_variable('biases', [NUM_OUTPUTS])
        fc4 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
        _activation_summary(fc4)

    with tf.variable_scope('sigmoid_linear') as scope:
        weights = _weight_variable_with_l2_loss(
            'weights', [NUM_OUTPUTS, NUM_OUTPUTS], FLAGS.lambda1)
        tf.add_to_collection('weights', weights)
        biases = _bias_variable('biases', [NUM_OUTPUTS])
        sigmoid_linear = tf.nn.xw_plus_b(fc4, weights, biases, name=scope.name)
        l1_loss = tf.multiply(tf.reduce_sum(tf.abs(sigmoid_linear)), FLAGS.lambda2, name='l1_loss')
        tf.add_to_collection('losses', l1_loss)
        _activation_summary(sigmoid_linear)

    return sigmoid_linear


def apply_max_norm(weights, max_norm):
    num_outputs = weights.get_shape()[-1].value
    transposed = tf.transpose(weights)
    transposed_shape = transposed.get_shape()
    reshaped = tf.reshape(transposed, [num_outputs, -1])
    clipped = tf.clip_by_norm(reshaped, FLAGS.lambda3, axes=[1])
    max_norm = tf.transpose(tf.reshape(clipped, transposed_shape))
    return tf.assign(weights, max_norm)


def cross_entropy_loss(logits, labels):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


def loss(logits, labels):
    cross_entropy_mean = cross_entropy_loss(logits, labels)
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def mean_cross_entropy_loss(logits, labels):
    with tf.variable_scope('mean_cross_entropy_loss'):
        mean, _ = tf.metrics.mean(
            cross_entropy_loss(logits, labels), updates_collections='update_ops')
    return mean


def auc(logits, labels):
    with tf.variable_scope('auc'):
        predictions = tf.nn.sigmoid(logits)
        _auc, _ = tf.metrics.auc(labels, predictions, updates_collections='update_ops')
    return _auc
