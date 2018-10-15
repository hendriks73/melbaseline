#!/bin/bash

#
# Helper script to train and predict on Google ML Engine.
#
# To make this work, upload your data (.tsv.bz2 files and mel_features.joblib files) to
# Google Cloud Storage. Note that mel_features.joblib need to be created from the provided
# JSON files. Use the provided script 'extractmelfeatures' for this.
#
# Then adjust the variables below to match your region and data layout.
# Finally, run the script.
#
# Predictions will be stored in the remote folder JOB_DIR.
#


export BUCKET_NAME=mediaeval18
export REGION=europe-west1

export FEATURE_FILES=gs://$BUCKET_NAME/valid/mel_features.joblib,gs://$BUCKET_NAME/train/mel_features.joblib

# lastfm
export JOB_NAME="mediaeval18_lastfm_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export TRAIN_FILES=gs://$BUCKET_NAME/train/acousticbrainz-mediaeval2017-lastfm-train.tsv.bz2
export VALID_FILES=gs://$BUCKET_NAME/valid/acousticbrainz-mediaeval-lastfm-validation.tsv.bz2
export TEST_FILES=gs://$BUCKET_NAME/test-lastfm/mel_features.joblib

gcloud ml-engine jobs submit training $JOB_NAME --module-name=melbaseline.training --region=$REGION --package-path=./melbaseline --job-dir=$JOB_DIR --runtime-version=1.10 --config=./cloudml-gpu.yaml -- --train-files=$TRAIN_FILES --valid-file=$VALID_FILES --test-files=$TEST_FILES --feature-files=$FEATURE_FILES

# tagtraum
export JOB_NAME="mediaeval18_tagtraum_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export TRAIN_FILES=gs://$BUCKET_NAME/train/acousticbrainz-mediaeval2017-tagtraum-train.tsv.bz2
export VALID_FILES=gs://$BUCKET_NAME/valid/acousticbrainz-mediaeval-tagtraum-validation.tsv.bz2
export TEST_FILES=gs://$BUCKET_NAME/test-tagtraum/mel_features.joblib

gcloud ml-engine jobs submit training $JOB_NAME --module-name=melbaseline.training --region=$REGION --package-path=./melbaseline --job-dir=$JOB_DIR --runtime-version=1.10 --config=./cloudml-gpu.yaml -- --train-files=$TRAIN_FILES --valid-file=$VALID_FILES --test-files=$TEST_FILES --feature-files=$FEATURE_FILES

# discogs
export JOB_NAME="mediaeval18_discogs_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export TRAIN_FILES=gs://$BUCKET_NAME/train/acousticbrainz-mediaeval2017-discogs-train.tsv.bz2
export VALID_FILES=gs://$BUCKET_NAME/valid/acousticbrainz-mediaeval-discogs-validation.tsv.bz2
export TEST_FILES=gs://$BUCKET_NAME/test-discogs/mel_features.joblib

gcloud ml-engine jobs submit training $JOB_NAME --module-name=melbaseline.training --region=$REGION --package-path=./melbaseline --job-dir=$JOB_DIR --runtime-version=1.10 --config=./cloudml-gpu.yaml -- --train-files=$TRAIN_FILES --valid-file=$VALID_FILES --test-files=$TEST_FILES --feature-files=$FEATURE_FILES

# allmusic
export JOB_NAME="mediaeval18_allmusic_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export TRAIN_FILES=gs://$BUCKET_NAME/train/acousticbrainz-mediaeval2017-allmusic-train.tsv.bz2
export VALID_FILES=gs://$BUCKET_NAME/valid/acousticbrainz-mediaeval2017-allmusic-validation.tsv.bz2
export TEST_FILES=gs://$BUCKET_NAME/test-allmusic/mel_features.joblib

gcloud ml-engine jobs submit training $JOB_NAME --module-name=melbaseline.training --region=$REGION --package-path=./melbaseline --job-dir=$JOB_DIR --runtime-version=1.10 --config=./cloudml-gpu.yaml -- --train-files=$TRAIN_FILES --valid-file=$VALID_FILES --test-files=$TEST_FILES --feature-files=$FEATURE_FILES

# all
export JOB_NAME="mediaeval18_combined_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export TRAIN_FILES=gs://$BUCKET_NAME/train/acousticbrainz-mediaeval2017-allmusic-train.tsv.bz2,gs://$BUCKET_NAME/train/acousticbrainz-mediaeval2017-discogs-train.tsv.bz2,gs://$BUCKET_NAME/train/acousticbrainz-mediaeval2017-tagtraum-train.tsv.bz2,gs://$BUCKET_NAME/train/acousticbrainz-mediaeval2017-lastfm-train.tsv.bz2
export VALID_FILES=gs://$BUCKET_NAME/valid/acousticbrainz-mediaeval2017-allmusic-validation.tsv.bz2,gs://$BUCKET_NAME/valid/acousticbrainz-mediaeval-discogs-validation.tsv.bz2,gs://$BUCKET_NAME/valid/acousticbrainz-mediaeval-tagtraum-validation.tsv.bz2,gs://$BUCKET_NAME/valid/acousticbrainz-mediaeval-lastfm-validation.tsv.bz2
export TEST_FILES=gs://$BUCKET_NAME/test-allmusic/mel_features.joblib,gs://$BUCKET_NAME/test-discogs/mel_features.joblib,gs://$BUCKET_NAME/test-tagtraum/mel_features.joblib,gs://$BUCKET_NAME/test-lastfm/mel_features.joblib

gcloud ml-engine jobs submit training $JOB_NAME --module-name=melbaseline.training --region=$REGION --package-path=./melbaseline --job-dir=$JOB_DIR --runtime-version=1.10 --config=./cloudml-gpu.yaml -- --train-files=$TRAIN_FILES --valid-file=$VALID_FILES --test-files=$TEST_FILES --feature-files=$FEATURE_FILES
