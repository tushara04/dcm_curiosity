import os
import glob
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table
import pandas as pd
import numpy as np

os.makedirs("./second_level_analysis", exist_ok=True)

contrast_dir = "./dcm_curiosity/glm_spm/first_level_analysis/"
contrast_imgs = sorted(glob.glob(os.path.join(contrast_dir, "sub-*con_curiosity.nii.gz")))

print(f"found {len(contrast_imgs)} subjects")

n_subjects = len(contrast_imgs)
design_matrix = pd.DataFrame({'intercept': [1] * n_subjects})

second_level_model = SecondLevelModel(smoothing_fwhm=5.0)
second_level_model = second_level_model.fit(contrast_imgs, design_matrix=design_matrix)

z_map = second_level_model.compute_contrast('intercept', output_type='z_score')

z_data = z_map.get_fdata()
print(f"\nZ-score range: [{np.nanmin(z_data):.2f}, {np.nanmax(z_data):.2f}]")

z_map.to_filename("./second_level_analysis/group_curiosity_zmap.nii.gz")

print("\ntrying multiple thresholds to find peaks")

thresholds = [
    (3.1, "p<0.001"),
    (2.58, "p<0.005"),
    (2.33, "p<0.01"),
    (2.0, "p<0.023"),
    (1.96, "p<0.05")
]

best_table = None
best_threshold_name = None

for z_thresh, name in thresholds:
    try:
        table = get_clusters_table(z_map, stat_threshold=z_thresh, cluster_threshold=10)
        if len(table) > 0:
            print(f"\n=== Found {len(table)} clusters at {name} (z>{z_thresh}) ===")
            print(table[['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)']].to_string())
            
            if best_table is None:
                best_table = table
                best_threshold_name = name
                best_table.to_csv("./second_level_analysis/peaks_lenient.csv", index=False)
                print(f"Saved to peaks_lenient.csv")
            break
    except:
        continue

if best_table is None:
    print("\nno significant clusters found even at p<0.05 uncorrected")
    print("extracting top peaks regardless of significance...")
    
    threshold_for_top = np.percentile(z_data[~np.isnan(z_data)], 99.9)
    print(f"using 99.9th percentile: z > {threshold_for_top:.2f}")
    
    table_top = get_clusters_table(z_map, stat_threshold=threshold_for_top, cluster_threshold=10)
    if len(table_top) > 0:
        print(table_top[['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)']].to_string())
        table_top.to_csv("./second_level_analysis/peaks_top_percentile.csv", index=False)
        print("csv")
    else:
        print("no peak")'''
