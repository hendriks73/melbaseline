.. image:: https://img.shields.io/badge/License-AGPL%20v3-blue.svg
    :target: https://www.gnu.org/licenses/agpl-3.0

=============================================================
Mel Baseline for the MediaEval 2018 AcousticBrainz Genre Task
=============================================================

This package is intended to create a relatively *simple baseline* for
the `MediaEval 2018 AcousticBrainz Genre Task <http://multimediaeval.org/mediaeval2018/acousticbrainz/>`_ .
Participants are asked to automatically classify tracks into genres based on pre-computed content-based features.
Multiple ground-truths from several sources and with different, but overlapping label namespaces are provided.

The task consists of two subtasks:

1) Single-source Classification
2) Multi-source Classification

For more details please see `this page <https://multimediaeval.github.io/2018-AcousticBrainz-Genre-Task/>`_.

Method
------

Instead of trying to learn genres from all features available in AcousticBrainz JSON
files (`sample <https://acousticbrainz.org/ad73ef2a-a4ff-4970-82f5-fe8901033d7c/low-level/view?n=0>`_),
we *only* pay attention to the low-level melband features. Each of the 9 different melband features
consists of 40 values, representing information about a specific `Mel <https://en.wikipedia.org/wiki/Mel_scale>`_
frequency band. Mel-based features have a long history in automatic genre recognition and are usually associated
with `timbre <https://en.wikipedia.org/wiki/Timbre>`_. E.g. G. Tzanetakis used
`MFCC <https://en.wikipedia.org/wiki/Mel-frequency_cepstrum>`_ as one of the timbral features in [1].

We treat the 9 melband features as a one-dimensional image with 9 channels. I.e. ``mean`` with its 40 values
is one channel, ``dmean`` another and so forth. This preserves both the spatial relationship between
bands as well as different kinds of features. We present these features to our neural network
in the shape of a ``(40, 9)``-tensor (size, channel).

Because our feature tensor has spatial relationships, it makes sense to use a
`convolutional neural network (CNN) <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_
architecture. In fact, we are using a fully convolutional network (FCN), i.e. we do not employ fully connected
layers for classification, but global pooling.

The used network is specified in `network.py <melbaseline/network.py>`_. Because the genre classification task
is multi-class and multi-label, we are using *sigmoids* as final activation functions and *binary crossentropy*
as loss function. For regularization we use both
`dropout <http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf>`_ and
`early stopping <https://en.wikipedia.org/wiki/Early_stopping>`_.
For optimization during learning we use `Adam <https://arxiv.org/abs/1412.6980>`_.

After training, the model is capable of predicting real values between 0 and 1 for each label. In order to map each of
of these values to ``true`` or ``false``, we have to define thresholds. To ensure a performance that
neither emphasizes `precision nor recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_, we calculate
the `F1 score <https://en.wikipedia.org/wiki/F1_score>`_ (harmonic average of precision and recall) for each possible
threshold for predictions on the validation set and pick the threshold for ``max(F1)``. This is done for each label
individually. The found thresholds are then used when predicting labels for the test set. The used F1 maximization
procedure is also known under the term *plug-in rule approach* (as opposed to *structured loss minimization*).
Note that should no class-prediction be greater than its threshold, we normalize predictions using their thresholds and
then simply pick the largest one. This is done to ensure at least one prediction per track as required by the provided
``check.R`` evaluation script.


Subtask 1
---------

The system described above works nicely when training on a single ground-truth, a validation dataset and a testing
dataset that all share the same label space (all datasets refer to genres in the same way).


Subtask 2
---------

Subtask 2 allows training on multiple datasets (each with their own label space). At prediction time, it is required
to "speak the language" of only *one* of the training datasets (the target ground-truth). I.e. predict for a specific
label space (set of genre/label names).

In order to exploit some overlaps, we normalize all genre names before creating the combined training ground-truth.
The applied normalization is very simple—it merely removes any non-letters and whitespace from the genre names.

Because we calculate thresholds based on validation data, we require validation datasets for each of the training
datasets.

At prediction time we simply drop any labels that do not occur in the desired target ground-truth and undo the
normalization.


Installation
============

1) Create a *clean* Python 3.5 environment (e.g. using `miniconda <https://conda.io/miniconda.html>`_)
2) (install dependencies)
3) Clone this repo and run ``setup.py install``:

.. code-block:: console

    git clone https://github.com/hendriks73/melbaseline.git
    cd melbaseline
    python setup.py install


Usage
=====

After installation, you should be able to run ``extractmelfeatures`` from the command line in order
to extract the melband features from your JSON files:

.. code-block:: console
    extractmelfeatures -i INPUT_BASE_FOLDER -o mel_features.joblib

Obviously you need to replace ``INPUT_BASE_FOLDER`` with the folder into which you extracted the
`AcousticBrainz <https://acousticbrainz.org>`_ JSON feature files.

Once you have create ``mel_features.joblib``-files for all dataset parts, you can start training on them.

.. code-block:: console
    trainandpredict --train-files=some_train.tsv.bz2\
        --valid-files=some_validation.tsv.bz2\
        --test-files=test_mel_features.joblib\
        --features-files=valid_mel_features.joblib,train_mel_features.joblib\
        --job-dir=output_dir

Again, the filenames in the sample above are just placeholders. The ``...tsv.bz2`` files represent genre
annotation files as made available by the task organizers
`here <https://github.com/multimediaeval/2018-AcousticBrainz-Genre-Task/tree/master/data>`_.

The ``...mel_features.joblib`` files are the files create by the script ``extractmelfeatures`` based on the data
also made available `here <https://github.com/multimediaeval/2018-AcousticBrainz-Genre-Task/tree/master/data>`_.

Note that for subtask 2 (train on multiple datasets, predict for a single dataset), you can specify multiple
dataset for training, validation and test-prediction like this:

.. code-block:: console
    trainandpredict --train-files=train__1__.tsv.bz2,train__2__.tsv.bz2,train__3__.tsv.bz2\
        --valid-files=validation__1__.tsv.bz2,validation__2__.tsv.bz2,validation__3__.tsv.bz2\
        --test-files=test__1__mel_features.joblib,test__2__mel_features.joblib,test__3__mel_features.joblib\
        --features-files=valid_mel_features.joblib,train_mel_features.joblib\
        --job-dir=output_dir

Note that when providing multiple ground-truth datasets, you should specify them in the *same order* for
all parameters, so that predictions are calibrated on the correct validation ground-truth and for the right
label namespace.


Google Cloud ML
---------------

Instead of executing the training and prediction process locally, you can also use `Google Cloud Machine Learning
Engine <https://cloud.google.com/ml-engine/>`_. To do so, you have to

- `Sign up and install the Google Cloud SDK
<https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#setup>`_.
- Upload the task ground-truths and ``mel_features.joblib`` files to Google Storage.
- Edit the provided script ``trainandpredict_ml_engine.sh`` to reflect your naming choices.
- Run ``trainandpredict_ml_engine.sh``.


License
=======

Source code and models can be licensed under the GNU AFFERO GENERAL PUBLIC LICENSE v3.
For details, please see the `LICENSE <LICENSE>`_ file.


Citation
========

If you use this project in your work, please consider citing this publication:

.. code-block:: latex

   @inproceedings{
      Title = {Media{E}val 2018 Acoustic{B}rainz Genre Task: A {CNN} Baseline Relying on Mel-Features},
      Author = {Schreiber, Hendrik},
      Booktitle = {Proceedings of the Media{E}val 2018 Multimedia Benchmark Workshop},
      Month = {10},
      Year = {2018},
      Address = {Sophia Antipolis, France}
   }


References
==========

.. [1] George Tzanetakis, Perry Cook, `Musical Genre Classification of Audio Signals
    <https://dspace.library.uvic.ca/bitstream/handle/1828/1344/tsap02gtzan.pdf>`_
    IEEE Transactions on Speech and Audio Processing, 10.5 (2002): 293-302.
