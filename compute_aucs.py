import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


def _progress(curr, end, message):
    sys.stdout.write('\r>> %s %.1f%%' % (message, float(curr) / float(end) * 100.0))
    sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='data directory')
    parser.add_argument(
        '--eval_dir', default='deepsea_eval', help='directory to read predictions from')
    parser.add_argument('--split', default='test', help='val or test')
    parser.add_argument(
        '--global_step', default='1', help='global step of model that generated predictions')
    args = parser.parse_args()

    labels = np.load(os.path.join(args.data_dir, args.split + '.npy'))
    predictions = np.load(
        os.path.join(args.eval_dir, '%s-predictions-%s.npy' % (args.split, args.global_step)))
    num_outputs = labels.shape[1]
    assert(len(labels) == len(predictions))
    assert(num_outputs == 919)

    aucs_file = os.path.join(args.eval_dir, '%s-aucs-%s.txt' % (args.split, args.global_step))
    with open(aucs_file, 'w') as f:
        for i in xrange(num_outputs):
            try:
                auc = roc_auc_score(labels[:, i], predictions[:, i])
                f.write('%.9f\n' % auc)
            except ValueError:
                f.write('NA (No positive in Test region)\n')
            _progress(i + 1, num_outputs, 'Computing AUCs')
        print
    print('Wrote AUCs to %s' % aucs_file)
