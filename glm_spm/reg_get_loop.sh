
#!/bin/bash

BASE_DIR="/home/tushara/Documents/projects/dcm_curiosity/data/ds004182/derivatives"

cd "$BASE_DIR"

for SUB_DIR in sub-control* sub-experimental*; do
    if [ -d "$SUB_DIR" ]; then
        REG_DIR="$SUB_DIR/regressors"
        
        if [ -d "$REG_DIR" ]; then
            echo "Processing regressors for $SUB_DIR..."
            
            for f in "$REG_DIR"/*magictrickwatching_run-*_label-motdemean_regressor.1D \
                     "$REG_DIR"/*magictrickwatching_run-*_label-motderiv_regressor.1D \
                     "$REG_DIR"/*magictrickwatching_run-*_label-ventriclePC_regressor.1D \
                     "$REG_DIR"/*magictrickwatching_label-mot_regressor.1D; do
                if [ -L "$f" ]; then
                    echo "  Getting $(basename $f)"
                    datalad get "$f"
                fi
            done
        else
            echo "Regressors directory not found for $SUB_DIR, skipping."
        fi
    fi
done

echo "Done downloading all regressor files!"
