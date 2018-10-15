from itertools import cycle

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """
    Key-based Keras sequence to be used as generator during fitting and/or predicting.
    This sequence can balance samples for a given set of classes.
    """

    def __init__(self, labels, classes, classes_to_balance, sample_loader, batch_size=32, sample_shape=(32, 32, 32),
                 shuffle=True):
        """
        Initialization.

        :param labels: labels as dictionary with unique keys and class-tuples as values
        :param classes: number of classes
        :param classes_to_balance: the classes used during balancing, may be ``None``, if no balancing is desired
        :param sample_loader: function used to load a sample (features) when given a sample key (e.g. a UUID)
        :param batch_size: batch size
        :param sample_shape: shape of a single feature sample
        :param shuffle: shuffle samples
        """
        self.sample_shape = sample_shape
        self.batch_size = batch_size
        self.labels = labels
        if classes_to_balance is None:
            print('DataGenerator does NOT balance.')
            self.sample_ids = list(labels.keys())
        else:
            print('DataGenerator WILL balance samples based on main genre labels.')
            self.sample_ids = self.balance(labels, set(classes_to_balance))
        label_set = set()
        for label_list in labels.values():
            label_set |= set(label_list)
        self.binarizer = MultiLabelBinarizer(classes=[c for c in range(classes)])
        self.binarizer.fit([[c] for c in range(classes)])
        self.shuffle = shuffle
        self.sample_loader = sample_loader
        self.indexes = None
        self.on_epoch_end()

    @staticmethod
    def balance(labels, classes_to_balance):
        """
        Create a list of label keys, so that their corresponding classes are balanced
        (through oversampling). Only consider keys that are associated with one of the
        classes contained in ``classes_to_balance``.

        :param classes_to_balance: set of classes to consider during balancing
        :param labels: dictionary of keys and class tuples
        :return: list of label keys with the same number of samples per class. Keys in this list
        are **not** unique (oversampling).
        """
        keys = labels.keys()
        print('Number of classes to balance: {} {}'.format(len(classes_to_balance), classes_to_balance))
        # create per class id lists
        per_class_sample_ids = {}
        for klass in classes_to_balance:
            per_class_sample_ids[klass] = []

        max_number_of_samples = 0
        for sample_id in keys:
            for klass in labels[sample_id]:
                if klass in classes_to_balance:
                    per_class_sample_ids[klass].append(sample_id)
                    max_number_of_samples = max(max_number_of_samples, len(per_class_sample_ids[klass]))
        print('Max number of samples per balanced class: {}'.format(max_number_of_samples))
        per_class_endless_iterators = [cycle(l) for l in per_class_sample_ids.values()]

        balanced_keys = []
        for _ in range(max_number_of_samples):
            for iter in per_class_endless_iterators:
                balanced_keys.append(next(iter))
        print('Original number of samples: {}'.format(len(keys)))
        oversample_factor = len(balanced_keys) / float(len(keys))
        print('Balanced number of samples: {} (factor {:0.2f})'.format(len(balanced_keys), oversample_factor))
        return balanced_keys

    def __len__(self):
        """
        Number of batches per epoch.

        :return: batches per epoch
        """
        return int(np.floor(len(self.sample_ids) / self.batch_size))

    def __getitem__(self, batch_index):
        """
        Generate a batch for the given batch index.

        :param batch_index: batch index
        :return: one batch of data
        """
        indexes = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        temp_keys = [self.sample_ids[k] for k in indexes]
        X, y = self.__data_generation(temp_keys)
        return X, y

    def on_epoch_end(self):
        """
        Re-shuffle, if necessary after each epoch.
        """
        self.indexes = np.arange(len(self.sample_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, keys):
        """
        Creates a batch of data.

        :param keys: ids for the samples that should make up the batch
        :return: batch of data
        """
        X = np.empty((self.batch_size, *self.sample_shape))
        for i, key in enumerate(keys):
            X[i,] = self.sample_loader(key)

        y = self.binarizer.transform([self.labels[key] for key in keys])

        return X, y
