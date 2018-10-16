import argparse
from os.path import join, basename, dirname, exists

import numpy as np
import sys
import tensorflow as tf
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.lib.io.file_io import FileIO

from melbaseline import groundtruth
from melbaseline.cloudml_utils import create_local_copy, save_model
from melbaseline.generator import DataGenerator
from melbaseline.history import print_history
from melbaseline.network import create_model


def create_mel_sample_loader(dataset):
    """
    Create sample loading function.

    :param dataset: dataset, dict that maps a ``key`` to a sample
    :return: function that returns a sample for a key (or sample id)
    """
    def mel_sample_loader(key):
        return dataset[key]

    return mel_sample_loader


def train_and_predict(train_files, valid_files, test_files, feature_files, job_dir, balance=False):

    if tf.test.gpu_device_name():
        print('Default GPU: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Failed to find default GPU.")
        #sys.exit(1)

    print('Creating local copies, if necessary...')

    train_files = [create_local_copy(train_file) for train_file in train_files.split(',')]
    valid_files = [create_local_copy(val_file) for val_file in valid_files.split(',')]
    test_files = [create_local_copy(test_file) for test_file in test_files.split(',')]
    feature_files = [create_local_copy(feature_file) for feature_file in feature_files.split(',')]

    print('Loading data...')

    train_ground_truth = groundtruth.open_ground_truth(train_files)
    print('Loaded {} training annotations from {}.'.format(len(train_ground_truth.key_index_labels), train_files))

    train_valid_file = valid_files[0]
    valid_ground_truth = groundtruth.open_ground_truth(train_valid_file)
    print('Loaded {} validation annotations from {}.'.format(len(valid_ground_truth.key_index_labels),
                                                             train_valid_file))

    # ensure the same indices are used in training and validation
    valid_ground_truth.use_indices_from(train_ground_truth)

    train_valid_features = {}
    for feature_file in feature_files:
        train_valid_features.update(joblib.load(feature_file))

    # create generator
    batch_size = 1000
    lr = 0.001
    epochs = 500
    dropout = 0.2
    filters = 64
    checkpoint_model_file = 'checkpoint_model.h5'
    model_file = join(job_dir, 'model.h5')
    input_shape = (40, 9)  # mel feature space
    classes_count = len(train_ground_truth.classes())
    print('Number of classes: {}'.format(classes_count))
    with FileIO(join(job_dir, 'classes.txt'), mode='w') as output_f:
        for i, name in enumerate(train_ground_truth.index_to_label):
            output_f.write("{}: {}\n".format(i, name).encode('utf-8'))

    print('Creating generators...')
    classes_to_balance = None
    if balance:
        # balance based on main genres
        classes_to_balance = train_ground_truth.main_classes()

    train_generator = DataGenerator(train_ground_truth.key_index_labels, classes_count, classes_to_balance,
                                    sample_loader=create_mel_sample_loader(train_valid_features),
                                    batch_size=batch_size,
                                    sample_shape=input_shape, shuffle=True)
    valid_generator = DataGenerator(valid_ground_truth.key_index_labels, classes_count, None,
                                    sample_loader=create_mel_sample_loader(train_valid_features),
                                    batch_size=batch_size,
                                    sample_shape=input_shape, shuffle=False)

    print('Creating model...')
    model = create_model(input_shape=input_shape, output_dim=classes_count, filters=filters, dropout=dropout)

    model.compile(loss='binary_crossentropy', optimizer=(Adam(lr=lr)), metrics=['binary_accuracy'])
    print('Number of model parameters: {}'.format(model.count_params()))
    print(model.summary())

    print('Training...')
    callbacks = [EarlyStopping(monitor='val_loss', patience=50, verbose=1),
                 ModelCheckpoint(checkpoint_model_file, monitor='val_loss')]
    history = model.fit_generator(train_generator, epochs=epochs, callbacks=callbacks,
                                  validation_data=valid_generator)

    print(model.summary())
    print('Setup: batch_size={}, epochs={}, dropout={}, filters={}'.format(batch_size, epochs, dropout, filters))
    print_history(history.history)

    print('Loading best model before early stopping...')
    model = load_model(checkpoint_model_file)
    # and save to job_dir
    save_model(model, model_file)

    for train_file, valid_file, test_file in zip(train_files, valid_files, test_files):
        thresholds = predict_validation_labels(model, valid_file, train_ground_truth, train_valid_features,
                                               job_dir, input_shape)
        predict_test_labels(model, test_file, train_file, train_ground_truth, thresholds,
                            job_dir, input_shape)


def predict_validation_labels(model, valid_file, train_ground_truth, features, job_dir, input_shape, batch_size=1000):
    print('Validation prediction for {} ...'.format(valid_file))
    valid_ground_truth = groundtruth.open_ground_truth(valid_file)
    # ensure the same indices are used in training and validation
    valid_ground_truth.use_indices_from(train_ground_truth)
    valid_keys = list(valid_ground_truth.key_index_labels.keys())
    X_valid = to_array(valid_keys, input_shape, create_mel_sample_loader(features))
    valid_preds = model.predict(X_valid, batch_size=batch_size)
    joblib.dump(valid_preds, 'valid_preds_' + to_filename(valid_file, '.joblib'))
    thresholds = evaluate_predictions(valid_preds, valid_keys, valid_ground_truth)
    output_file = join(job_dir, 'valid_preds_' + to_filename(valid_file, '.tsv'))
    valid_preds_tuples = distribution_to_tuples(valid_preds, thresholds)
    write_predictions(output_file, valid_preds_tuples, valid_keys, valid_ground_truth)
    treshold_file = join(job_dir, 'thresholds_' + to_filename(valid_file, '.txt'))
    write_thresholds(treshold_file, thresholds)
    return thresholds


def write_thresholds(file, thresholds):
    with FileIO(file, mode='w') as output_f:
        for i in range(thresholds.shape[0]):
            output_f.write("{}: {}\n".format(i, thresholds[i]).encode('utf-8'))


def predict_test_labels(model, test_file, train_file, train_ground_truth, thresholds, job_dir, input_shape,
                        batch_size=1000):
    print('Test prediction for {} ...'.format(test_file))
    test_features = joblib.load(test_file)
    test_keys = list(test_features.keys())
    X_test = to_array(test_keys, input_shape, create_mel_sample_loader(test_features))
    test_preds = model.predict(X_test, batch_size=batch_size)
    joblib.dump(test_preds, 'test_preds_' + to_filename(test_file, '.joblib'))
    output_file = join(job_dir, 'test_preds_' + to_filename(test_file, '.tsv'))
    train_ground_truth.set_target_ground_truth(train_file)
    test_preds_tuples = distribution_to_tuples(test_preds, thresholds)
    write_predictions(output_file, test_preds_tuples, test_keys, train_ground_truth)


def write_predictions(output_file, tuple_predictions, keys, ground_truth):
    """
    Write predictions to a TSV file.

    :param output_file: output file
    :param tuple_predictions: predictions stored in a tuple per sample
    :param keys: iterable of sample keys (UUIDs)
    :param ground_truth: ground-truth object that knows the index->label conversion
    """
    if exists(output_file):
        print('WARNING: Overwriting {}'.format(output_file))

    with FileIO(output_file, mode='w') as f:
        # convert indices to label names using index_to_label
        for key, indices in zip(keys, tuple_predictions):
            line = key + ground_truth.to_labels(indices)
            f.write(line + '\n')


def to_array(keys, input_shape, loader):
    """
    Create an in-memory feature tensor X based on the given ids, input_shape and loader.

    :param keys: keys (sample ids/UUIDs)
    :param input_shape: input shape (shape of features for one sample)
    :param loader: loader function
    :return: X
    """
    samples = len(keys)
    X = np.empty((samples, *input_shape))
    for i, key in enumerate(keys):
        X[i] = loader(key)
    return X


def to_filename(file, extension):
    """
    Create filename with a given extension that is (hopefully) unique.

    :param file: file
    :param extension: new extension
    :return: filename (without path)
    """
    s = basename(file)
    if 'tagtraum' in s or 'lastfm' in s or 'discogs' in s or 'allmusic' in s:
        pass
    else:
        # take folder name into account and hope it's different
        # from other foldernames
        s = basename(dirname(file)) + '_' + s
    return s.replace('.tsv.bz2', '').replace('.joblib', '') + extension


def evaluate_predictions(predictions, keys, ground_truth):
    """
    Evaluate the given predictions against the given ground-truth.
    The ``predictions`` may refer to *more* classes than covered by the ``ground_truth``.
    This scenario may happen, when we train on the union of multiple datasets with different
    labels, but evaluate on just one of these datasets (with fewer labels).
    During evaluation, only the intersection between prediction and ground-truth labels
    are considered.

    :param predictions: predictions (same order as ``keys``)
    :param keys: prediction keys (UUIDs), to be used with ``ground_truth``
    :param ground_truth: ground-truth to evaluate against
    """

    # get the number of classes the prediction is for
    prediction_classes = predictions.shape[1]
    # get the number of classes actually covered by the ground truth
    ground_truth_classes = len(ground_truth.classes())

    binarizer = MultiLabelBinarizer(classes=[c for c in range(prediction_classes)])
    binarizer.fit([[c] for c in range(prediction_classes)])
    y = binarizer.transform([ground_truth.key_index_labels[k] for k in keys])

    ir_metrics = calculate_ir_metrics(predictions, y)
    max_f_score_threshold = ir_metrics['max_f_score_threshold']
    max_f_score = ir_metrics['max_f_score']
    average_precision = ir_metrics['average_precision']

    max_sum = 0.
    max_main_sum = 0.
    ap_sum = 0.
    ap_main_sum = 0.
    main_labels = 0
    for i in ground_truth.classes():
        # this only considers classes/labels that actually occur in the ground-truth
        max_sum += max_f_score[i]
        ap_sum += average_precision[i]
        if ground_truth.index_to_main_label[i]:
            max_main_sum += max_f_score[i]
            ap_main_sum += average_precision[i]
            main_labels += 1

    print('Avg max f_score: {}'.format(max_sum / ground_truth_classes))
    print('Avg max f_score (main labels): {}'.format(max_main_sum / main_labels))
    print('mAP: {}'.format(ap_sum / ground_truth_classes))
    print('mAP (main labels): {}'.format(ap_main_sum / main_labels))

    thresholds = np.empty(prediction_classes)
    for i in range(prediction_classes):
        thresholds[i] = -max_f_score_threshold[i]
    for i in ground_truth.classes():
        thresholds[i] = -thresholds[i]

    return thresholds


def distribution_to_tuples(predictions, threshold, at_least_one_hot=True):
    """
    Convert a distribution to k-hot tuples using the given threshold.

    :param predictions: predictions, 2-dim numpy matrix (samples, distribution)
    :param threshold: per-class thresholds for k-hot mapping (binarization)
    :param at_least_one_hot: ensure that the relatively highest prediction is chosen, if no prediction is greater
    than its threshold
    :return: list of tuples, each tuple contains class indices
    """
    threshold_normalized_predictions = predictions / threshold
    k_hot_predictions = np.where(threshold_normalized_predictions >= 1., 1., 0.)

    classes = predictions.shape[1]
    binarizer = MultiLabelBinarizer(classes=[c for c in range(classes)])
    binarizer.fit([[c] for c in range(classes)])
    binarized_prediction_tuples = binarizer.inverse_transform(k_hot_predictions)

    # make sure we have at least one prediction
    if at_least_one_hot:
        ensure_at_least_one_prediction(binarized_prediction_tuples, threshold_normalized_predictions)

    return binarized_prediction_tuples


def ensure_at_least_one_prediction(binarized_prediction_tuples, threshold_normalized_predictions):
    """
    Ensure that ``binarized_prediction_tuples`` does not contain empty tuples
    by picking the class with the maximum threshold-normalized prediction.
    Note that the original list is manipulated!

    :param threshold_normalized_predictions: numpy matrix (sample, class) with the threshold-normalized predictions
    :param binarized_prediction_tuples: list of tuples, with tuples consisting of class indices
    """
    highest_preds = np.argmax(threshold_normalized_predictions, axis=1)
    for i, tuple_prediction in enumerate(binarized_prediction_tuples):
        if len(tuple_prediction) == 0:
            binarized_prediction_tuples[i] = (highest_preds[i],)


def calculate_ir_metrics(predictions, groundtruth):
    """
    Calculate several information retrieval (IR) metrics, like precision, recall, f_score, etc.

    :param predictions: predictions, 2-dim numpy matrix (samples, distribution)
    :param groundtruth: ground-truth
    :return: dictionary of IR-metrics
    """
    classes = predictions.shape[1]
    # For each class
    f_score = {}
    precision = {}
    recall = {}
    threshold = {}
    average_precision = {}
    max_f_score = {}
    max_f_score_threshold = {}

    for i in range(classes):
        precision[i], recall[i], threshold[i] = precision_recall_curve(groundtruth[:, i], predictions[:, i])
        average_precision[i] = average_precision_score(groundtruth[:, i], predictions[:, i])
        f_score[i] = np.nan_to_num((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))
        f_max_index = np.argmax(f_score[i])
        max_f_score[i] = f_score[i][f_max_index]
        max_f_score_threshold[i] = threshold[i][f_max_index]

    return {'precision': precision,
            'recall': recall,
            'f_score': f_score,
            'threshold': threshold,
            'max_f_score': max_f_score,
            'max_f_score_threshold': max_f_score_threshold,
            'average_precision': average_precision}


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The program 'trainandpredict' takes train, validation and test datasets
    as arguments, trains a neural network and predicts labels for both the
    validation and the test dataset.
    ''')
    # Input Arguments
    parser.add_argument(
        '--train-files',
        help='One or more comma-separated .tsv.bz2 ground-truth files provided by the organizers.',
        required=True
    )
    parser.add_argument(
        '--valid-files',
        help='One or more comma-separated .tsv.bz2 ground-truth files provided by the organizers. Must be in same '
             'order as the train-files, if different label spaces are used.',
        required=True
    )
    parser.add_argument(
        '--test-files',
        help='One or more comma-separated joblib file created by \'extractmelfeatures\' that contain both UUIDs '
             'and mel features for the test set. Must be in same order as the train-files, if different label '
             'spaces are used.',
        required=True
    )
    parser.add_argument(
        '--feature-files',
        help='One or more comma-separated mel feature files generated by \'extractmelfeatures\' which provide features'
             ' for both the validation and the training set.',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='output directory',
        required=True
    )
    parser.add_argument(
        '--balance',
        help='If present, the training dataset is balanced w.r.t. the main genre labels.',
        dest='balance',
        action='store_true',
        required=False
    )
    parser.set_defaults(balance=False)

    args = parser.parse_args()
    arguments = args.__dict__
    return arguments


def main():
    arguments = parse_arguments()
    train_and_predict(**arguments)


if __name__ == '__main__':
    main()
