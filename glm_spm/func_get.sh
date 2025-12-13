#!/bin/bash

SUB="sub-control002"
FUNC_DIR="/home/tushara/Documents/projects/dcm_curiosity/data/ds004182/derivatives/$SUB/func"

datalad get "$FUNC_DIR/${SUB}_task-magictrickwatching_desc-fullpreproc_bold.nii.gz"

