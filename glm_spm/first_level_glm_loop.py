import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
import numpy as np
import glob
import os

BASE_DIR = "./dcm_curiosity/data/ds004182/derivatives"
EVENTS_FILE = "./dcm_curiosity/glm_spm/MMC_experimental_data_edited.csv"
OUTPUT_DIR = "./first_level_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

subjects = sorted(glob.glob(os.path.join(BASE_DIR, "sub-control*")) + 
                  glob.glob(os.path.join(BASE_DIR, "sub-experimental*")))

for SUB in subjects:
    sub_name = os.path.basename(SUB)
    out_file = os.path.join(OUTPUT_DIR, f"{sub_name}_con_curiosity.nii.gz")
    
    if os.path.exists(out_file):
        print(f"{sub_name}: Output already exists, skipping")
        continue
    
    try:
        bold_file = os.path.join(SUB, "func", f"{sub_name}_task-magictrickwatching_desc-fullpreproc_bold.nii.gz")
        confounds_dir = os.path.join(SUB, "regressors")
        
        if not os.path.exists(bold_file):
            print(f"{sub_name}: BOLD file not found, skipping")
            continue
            
        if not os.path.exists(confounds_dir):
            print(f"{sub_name}: Regressors directory not found, skipping")
            continue
        
        events = pd.read_csv(EVENTS_FILE)
        events = events[events["BIDS"] == sub_name]
        
        if len(events) == 0:
            print(f"{sub_name}: No events found in CSV, skipping")
            continue
            
        events_glm = pd.DataFrame({
            "trial_type": "magic",
            "onset": events["displayVidOnset"],
            "duration": events["displayVidDuration"],
            "modulation": events["rC_mean-centered"]
        })
        
        confound_files = sorted(glob.glob(os.path.join(confounds_dir, "*magic*mot*regressor.1D")))
        
        if len(confound_files) == 0:
            print(f"{sub_name}: No confound files found, skipping")
            continue
        
        confounds_list = []
        for i, f in enumerate(confound_files):
            conf = pd.read_csv(f, delim_whitespace=True, header=None)
            conf.columns = [f"motion_run{i+1}_param{j}" for j in range(conf.shape[1])]
            confounds_list.append(conf)
        
        confounds = pd.concat(confounds_list, axis=1)
        
        print(f"{sub_name}: Confounds shape {confounds.shape}, Events shape {events_glm.shape}")
        
        fmri_glm = FirstLevelModel(
            t_r=2.0,                 
            hrf_model="spm",
            high_pass=1/128,
            noise_model="ar1",
            standardize=False
        )
        
        fmri_glm = fmri_glm.fit(
            bold_file,
            events=events_glm,
            confounds=confounds
        )
        
        print(f"{sub_name}: Design matrix columns: {fmri_glm.design_matrices_[0].columns.tolist()}")
        
        contrast = fmri_glm.compute_contrast(
            "magic",
            output_type="stat"
        )
        
        contrast.to_filename(out_file)
        print(f"{sub_name}: contrast saved to {out_file}\n")
        
    except Exception as e:
        print(f"{sub_name}: error - {str(e)}, skipping\n")
        continue

print("all subjects processed!")
