import json
from os import walk
from os.path import join

import numpy as np
from sklearn.externals import joblib


def read_feature_file(json_file):
    """
    Reads an AcousticBrainz low-level feature file and extracts
    most Mel features. Features are max1 normalized individually.
    I.e. ``mel_min`` values are max1 normalized, independently from
    ``mel_max`` values.

    :param json_file: AcousticBrainz low-level feature file
    :return: partially scaled values as dictionary
    """
    with open(json_file, mode='r') as f:
        data = json.load(f)
    # we're only interested in melbands features
    melbands = data['lowlevel']['melbands']
    mel_min = max1_scale(melbands['min']) # 40 dims each
    mel_mean = max1_scale(melbands['mean'])
    mel_median = max1_scale(melbands['median'])
    mel_max = max1_scale(melbands['max'])
    mel_var = max1_scale(melbands['var'])
    mel_dmean = max1_scale(melbands['dmean'])
    mel_dvar = max1_scale(melbands['dvar'])
    mel_dmean2 = max1_scale(melbands['dmean2'])
    mel_dvar2 = max1_scale(melbands['dvar2'])

    mel_features = np.vstack([mel_min, mel_mean, mel_median, mel_max, mel_var,
                              mel_dmean, mel_dvar, mel_dmean2, mel_dvar2]).T
    mel_features = mel_features.reshape(mel_features.shape[0], mel_features.shape[1])
    return mel_features


def read_feature_folder(base_folder):
    """
    Reads a folder and its subfolders, parses all json files and stores
    the result in a dictionary using the filenames (minus the '.json') as keys.

    :param base_folder: folder with AcousticBrainz ``.json`` files
    :return: dictionary with AcousticBrainz UUIDs as keys
    """
    feature_dataset = {}
    for (dirpath, _, filenames) in walk(base_folder):
        for file in [f for f in filenames if f.endswith('.json')]:
            key = file.replace('.json', '')
            features = read_feature_file(join(dirpath, file))
            feature_dataset[key] = features
    return feature_dataset


def convert_json_folder_to_joblib(base_folder, output_file):
    """
    Read all json features files in the given folder and its subfolders,
    store all mel features under keys equivalent to the file names (minus extension),
    store the resulting dict in ``output_file`` and return the dict.

    :param base_folder: base folder for json files
    :param output_file: joblib file
    :return: dict of keys and features
    """
    dataset = read_feature_folder(base_folder)
    joblib.dump(dataset, output_file)
    return dataset


def max1_scale(numbers):
    """
    Scale the given array of positive numbers so that the max is ``1``.
    Note that this does not shift the distribution.

    :param numbers: array of positive numbers
    :return: max1 scaled numpy array
    """
    scaled = np.array(numbers)
    maximum = np.max(scaled)
    if maximum != 0.:
        scaled = scaled / maximum
    return scaled.astype(np.float16)

