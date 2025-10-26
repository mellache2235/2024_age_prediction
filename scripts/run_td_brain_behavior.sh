#!/bin/bash
#
# Example commands for running brain-behavior analysis on TD cohorts
#

# CMI-HBN TD
python brain_behavior_td_cohorts.py \
  --dataset cmihbn_td \
  --data_dir /oak/stanford/groups/menon/projects/wdcai/2019_ADHD_NN/data/imaging/timeseries/CMIHBN/restfmri/timeseries/group_level/brainnetome/normz/ \
  --behavioral_csv /oak/stanford/groups/menon/projects/mellache/2021_foundation_model/scripts/dnn/prepare_data/adhd/C3SR.csv \
  --ig_csv /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/cmihbn_td_count_data.csv \
  --behavioral_columns HY IN \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_td \
  --n_components 10

# ADHD200 TD
python brain_behavior_td_cohorts.py \
  --dataset adhd200_td \
  --data_file /oak/stanford/groups/menon/projects/wdcai/2019_ADHD_NN/data/imaging/timeseries/ADHD200/restfmri/timeseries/group_level/brainnetome/normz/adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs_nn.pklz \
  --ig_csv /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/count_data/adhd200_td_count_data.csv \
  --behavioral_columns HY IN \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_td \
  --n_components 10

