#!/bin/bash
#
# Brain-behavior analysis for TD cohorts using .pklz files and IG CSV
#

# CMI-HBN TD
python brain_behavior_td_simple.py \
  --dataset cmihbn_td \
  --pklz_file /oak/stanford/groups/menon/projects/wdcai/2019_ADHD_NN/data/imaging/timeseries/CMIHBN/restfmri/timeseries/group_level/brainnetome/normz/cmihbn_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs_nn.pklz \
  --ig_csv /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/cmihbn_td_ig_scores.csv \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/cmihbn_td \
  --n_components 10

# ADHD200 TD
python brain_behavior_td_simple.py \
  --dataset adhd200_td \
  --pklz_file /oak/stanford/groups/menon/projects/wdcai/2019_ADHD_NN/data/imaging/timeseries/ADHD200/restfmri/timeseries/group_level/brainnetome/normz/adhd200_run-rest_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs_nn.pklz \
  --ig_csv /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/integrated_gradients/adhd200_td_ig_scores.csv \
  --output_dir /oak/stanford/groups/menon/projects/mellache/2024_age_prediction_test/results/brain_behavior/adhd200_td \
  --n_components 10

