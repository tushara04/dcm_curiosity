#!/bin/bash

SUB="sub-control002"
REG_DIR="/home/tushara/Documents/projects/dcm_curiosity/data/ds004182/derivatives/$SUB/regressors"

for f in "$REG_DIR"/*magictrickwatching_run-*_label-motdemean_regressor.1D \
         "$REG_DIR"/*magictrickwatching_run-*_label-motderiv_regressor.1D \
         "$REG_DIR"/*magictrickwatching_run-*_label-ventriclePC_regressor.1D \
         "$REG_DIR"/*magictrickwatching_label-mot_regressor.1D
do
    datalad get "$f"
done

