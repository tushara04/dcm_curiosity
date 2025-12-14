import os
import pandas as pd
import numpy as np
from nilearn import image
from nilearn.maskers import NiftiSpheresMasker
import glob

BASE_DIR = "/home/tushara/Documents/projects/dcm_curiosity/data/ds004182/derivatives"
OUTPUT_DIR = "/home/tushara/Documents/projects/dcm_curiosity/dcm_spm"

os.makedirs(f"{OUTPUT_DIR}/ROIs", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/time_series", exist_ok=True)

roi_coords = {
    'ROI1_mPFC': (-40.5, 28.5, 46.5),
    'ROI2_Hipp': (-43.5, -40.5, -52.5),
    'ROI3_Striatum': (-46.5, 16.5, -37.5)
}

radius = 8

roi_info = []
for roi_name, coords in roi_coords.items():
    roi_info.append({
        'roi_name': roi_name,
        'x': coords[0],
        'y': coords[1],
        'z': coords[2],
        'radius_mm': radius
    })

roi_df = pd.DataFrame(roi_info)
roi_df.to_csv(f"{OUTPUT_DIR}/ROI_definitions.csv", index=False)

subjects = sorted([s for s in glob.glob(f"{BASE_DIR}/sub-control*") if os.path.isdir(s)])[:25]

print(f"Processing {len(subjects)} subjects...")

extraction_results = []

for idx, sub_path in enumerate(subjects, 1):
    sub_name = os.path.basename(sub_path)
    print(f"[{idx}/{len(subjects)}] {sub_name}...", end=' ', flush=True)
    
    bold_file = f"{sub_path}/func/{sub_name}_task-magictrickwatching_desc-fullpreproc_bold.nii.gz"
    
    if not os.path.exists(bold_file):
        print("SKIP (no BOLD file)")
        continue
    
    os.makedirs(f"{OUTPUT_DIR}/time_series/{sub_name}", exist_ok=True)
    
    bold_img = image.load_img(bold_file)
    n_timepoints = bold_img.shape[3]
    
    subject_data = {
        'subject': sub_name,
        'n_timepoints': n_timepoints,
        'bold_file': bold_file
    }
    
    for roi_name, coords in roi_coords.items():
        masker = NiftiSpheresMasker(
            [coords],
            radius=radius,
            standardize=False,
            detrend=False
        )
        
        time_series = masker.fit_transform(bold_img)
        
        output_file = f"{OUTPUT_DIR}/time_series/{sub_name}/{roi_name}_timeseries.txt"
        np.savetxt(output_file, time_series, fmt='%.6f')
        
        subject_data[f'{roi_name}_mean'] = time_series.mean()
        subject_data[f'{roi_name}_std'] = time_series.std()
        subject_data[f'{roi_name}_min'] = time_series.min()
        subject_data[f'{roi_name}_max'] = time_series.max()
        subject_data[f'{roi_name}_file'] = f"time_series/{sub_name}/{roi_name}_timeseries.txt"
    
    extraction_results.append(subject_data)
    print("DONE")

print("\nSaving summaries...")

extraction_df = pd.DataFrame(extraction_results)
extraction_df.to_csv(f"{OUTPUT_DIR}/time_series_summary.csv", index=False)

quality_check = []
for _, row in extraction_df.iterrows():
    for roi_name in roi_coords.keys():
        std_val = row[f'{roi_name}_std']
        issues = []
        
        if std_val < 10:
            issues.append('low_variance')
        
        quality_check.append({
            'subject': row['subject'],
            'roi': roi_name,
            'mean': row[f'{roi_name}_mean'],
            'std': std_val,
            'range': row[f'{roi_name}_max'] - row[f'{roi_name}_min'],
            'quality_pass': len(issues) == 0,
            'issues': ','.join(issues) if issues else 'none'
        })

quality_df = pd.DataFrame(quality_check)
quality_df.to_csv(f"{OUTPUT_DIR}/time_series_quality.csv", index=False)

subject_list = extraction_df[['subject']].copy()
subject_list.to_csv(f"{OUTPUT_DIR}/subject_list.csv", index=False)

print(f"Complete! Processed {len(extraction_results)} subjects")

