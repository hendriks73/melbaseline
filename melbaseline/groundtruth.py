import bz2
import csv
import io


def open_ground_truth(files, target_file=None):
    """
    Opens and reads one or multiple ``...tsv.bz2`` ground-truth files
    as given by the MediaEval 2018 AcousticBrainz task.

    :param files: one or multiple files.
    :param target_file: the ground-truth to use as target "namespace", if ``None`` the first of the given files is
    used.
    :return: ground truth object
    """
    if isinstance(files, str):
        return GroundTruth(files)
    elif len(files) > 1:
        return CombinedGroundTruth(files, target_file)
    else:
        return GroundTruth(files[0])


class AbstractGroundTruth:
    """
    Base class for ground-truth encapsulation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.index_to_label = []
        self.norm_to_label = {}
        self.index_to_main_label = {}

    def classes(self):
        return [i for i,l in enumerate(self.index_to_label) if l is not None]

    def main_classes(self):
        return [i for i, v in enumerate(self.index_to_main_label) if v]

    def get_label(self, index):
        if index < 0 or index > len(self.index_to_label):
            return None
        normed_label = self.normalize_label(self.index_to_label[index])
        if normed_label is not None and normed_label in self.norm_to_label:
            return self.norm_to_label[normed_label]
        else:
            return None

    def to_labels(self, indices):
        line = ''
        for ind in indices:
            label = self.get_label(ind)
            if label is not None:
                line = line + '\t' + label
        return line

    def _read_label_files(self, files):
        labels = {}
        empty_string_set = {''}
        normalize = len(files) > 1
        for file in files:
            with open(file, mode='rb') as compressed_file:
                with bz2.BZ2File(compressed_file, mode='r') as binary_file:
                    text_file = io.TextIOWrapper(binary_file, 'utf-8', None, None)
                    reader = csv.reader(text_file, delimiter='\t')
                    for row in reader:
                        if row[0] == 'recordingmbid':
                            continue
                        id = row[0]
                        genres = set(row[2:]).difference(empty_string_set)
                        if normalize:
                            genres = set([self.normalize_label(g) for g in genres])
                        if id in labels:
                            labels[id] |= genres
                        else:
                            labels[id] = genres
        return labels

    def _create_key_index_labels(self, labels, label_to_index):
        key_index_labels = {}
        for key, value in labels.items():
            indices = []
            for l in value:
                if l in label_to_index:
                    indices.append(label_to_index[l])
            key_index_labels[key] = indices
        return key_index_labels

    def _index_conversions(self, labels):
        all_labels = set()
        for l in labels.values():
            all_labels |= l
        index_to_label = list(all_labels)
        index_to_label.sort()
        label_to_index = {}
        norm_to_label = {}
        for i, label in enumerate(index_to_label):
            label_to_index[label] = i
            norm_to_label[self.normalize_label(label)] = label
        return index_to_label, label_to_index, norm_to_label

    def set_target_ground_truth(self, target_ground_truth):
        """
        Sets the ground truth to use for ``get_label`` calls.

        :param target_ground_truth: ground truth to use in ``get_label`` calls.
        """
        if isinstance(target_ground_truth, str):
            target_ground_truth = GroundTruth(target_ground_truth)
        # use the norm to label mapping from the target groundtruth
        self.norm_to_label = target_ground_truth.norm_to_label

    @staticmethod
    def is_main_label(label):
        return '---' not in label

    @staticmethod
    def normalize_label(label):
        if label is None:
            return None
        splits = label.split('---')
        n = ''
        for s in splits:
            lower = s.lower()
            norm = ''
            for c in lower:
                if c.isalpha():
                    norm += str(c)
            if len(n) > 0:
                n += '---'
            n += norm
        return n


class GroundTruth(AbstractGroundTruth):
    """
    Ground truth class for a single ground truth file.
    """

    def __init__(self, file):
        super().__init__()
        self.files = [file]
        self.labels = self._read_label_files(self.files)
        self.index_to_label, self.label_to_index, self.norm_to_label = self._index_conversions(self.labels)
        self.index_to_main_label = [self.is_main_label(label) for label in self.index_to_label]
        self.key_index_labels = self._create_key_index_labels(self.labels, self.label_to_index)

    def use_indices_from(self, ground_truth):
        """
        Adjust indices, labels etc. to match the given ground-truth.

        :param ground_truth: target ground truth
        """

        # the other ground truth is normed, so we have to "un"norm it

        def translate(other_label):
            norm_other = self.normalize_label(other_label)
            if norm_other in self.norm_to_label:
                return self.norm_to_label[norm_other]
            else:
                return None

        self.index_to_label = [translate(l) for l in ground_truth.index_to_label]
        self.label_to_index = {translate(l): i for l, i in ground_truth.label_to_index.items() if translate(l) is not None}
        self.index_to_main_label = ground_truth.index_to_main_label
        self.key_index_labels = self._create_key_index_labels(self.labels, self.label_to_index)


class CombinedGroundTruth(AbstractGroundTruth):
    """
    Ground truth class that combines multiple ground truths into one object.
    """

    def __init__(self, files, target_ground_truth=None):
        super().__init__()
        if target_ground_truth is None:
            target_ground_truth = files[0]
        self.files = files
        self.labels = self._read_label_files(self.files)
        self.index_to_label, self.label_to_index, _ = self._index_conversions(self.labels)
        self.index_to_main_label = [self.is_main_label(label) for label in self.index_to_label]
        self.key_index_labels = self._create_key_index_labels(self.labels, self.label_to_index)
        # use the norm to label mapping from the target groundtruth
        self.set_target_ground_truth(target_ground_truth)
