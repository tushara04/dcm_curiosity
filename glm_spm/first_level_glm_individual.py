import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
import numpy as np
import glob

SUB = "sub-control002"

bold_file = f"/home/tushara/Documents/projects/dcm_curiosity/data/ds004182/derivatives/{SUB}/func/{SUB}_task-magictrickwatching_desc-fullpreproc_bold.nii.gz"
events_file = "/home/tushara/Documents/projects/dcm_curiosity/glm_spm/MMC_experimental_data_edited.csv"
confounds_dir = f"/home/tushara/Documents/projects/dcm_curiosity/data/ds004182/derivatives/{SUB}/regressors/"

events = pd.read_csv(events_file)
events = events[events["BIDS"] == SUB]
events_glm = pd.DataFrame({
    "trial_type": "magic",
    "onset": events["displayVidOnset"],
    "duration": events["displayVidDuration"],
    "modulation": events["rC_mean-centered"]
})

confound_files = sorted(glob.glob(confounds_dir + "*magic*mot*regressor.1D"))

confounds_list = []
for i, f in enumerate(confound_files):
    conf = pd.read_csv(f, delim_whitespace=True, header=None)
    conf.columns = [f"motion_run{i+1}_param{j}" for j in range(conf.shape[1])]
    confounds_list.append(conf)

confounds = pd.concat(confounds_list, axis=1)

print(f"Confounds shape: {confounds.shape}")
print(f"Events shape: {events_glm.shape}")

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

print("Design matrix columns:")
print(fmri_glm.design_matrices_[0].columns.tolist())

contrast = fmri_glm.compute_contrast(
    "magic",
    output_type="stat"
)

contrast.to_filename(f"./first_level_analysis/{SUB}_con_curiosity.nii.gz")
print(f"Contrast saved for {SUB}!")

