#!/bin/bash

BASE_DIR="/home/tushara/Documents/projects/dcm_curiosity/data/ds004182/derivatives"

cd "$BASE_DIR"

for SUB_DIR in sub-control* sub-experimental*; do
    if [ -d "$SUB_DIR" ]; then
        SUB_NAME=$(basename "$SUB_DIR")
        FUNC_FILE="$SUB_DIR/func/${SUB_NAME}_task-magictrickwatching_desc-fullpreproc_bold.nii.gz"
        
        if [ -L "$FUNC_FILE" ]; then
            echo "Getting $FUNC_FILE"
            datalad get "$FUNC_FILE"
        else
            echo "File not found: $FUNC_FILE, skipping."
        fi
    fi
done

echo "Done downloading all functional files!"
