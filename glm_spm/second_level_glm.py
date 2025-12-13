import os
import glob
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting
from nilearn.reporting import make_glm_report
import pandas as pd
import numpy as np


contrast_dir = "/home/tushara/Documents/projects/dcm_curiosity/glm_spm/first_level_analysis/"
contrast_imgs = sorted(glob.glob(os.path.join(contrast_dir, "sub-*_*con_curiosity.nii.gz")))
print(f"Found {len(contrast_imgs)} subjects")
print("First few files:")
for img in contrast_imgs[:3]:
    print(f"  {img}")

n_subjects = len(contrast_imgs)
design_matrix = pd.DataFrame({'intercept': [1] * n_subjects})

second_level_model = SecondLevelModel(smoothing_fwhm=5.0)
second_level_model = second_level_model.fit(
    contrast_imgs,
    design_matrix=design_matrix
)

z_map = second_level_model.compute_contrast(
    'intercept',
    output_type='z_score'
)

z_data = z_map.get_fdata()
print(f"\nZ-score statistics:")
print(f"  Min: {np.nanmin(z_data):.2f}")
print(f"  Max: {np.nanmax(z_data):.2f}")
print(f"  Mean: {np.nanmean(z_data):.2f}")
print(f"  Median: {np.nanmedian(z_data):.2f}")

from nilearn.glm import threshold_stats_img

print("\n=== Testing uncorrected threshold p<0.001 ===")
thresholded_map_unc, threshold_unc = threshold_stats_img(
    z_map,
    alpha=0.001,
    height_control=None,
    cluster_threshold=0
)
print(f"Uncorrected threshold (p<0.001): z > {threshold_unc:.2f}")

print("\n=== Testing uncorrected threshold p<0.01 ===")
thresholded_map_unc01, threshold_unc01 = threshold_stats_img(
    z_map,
    alpha=0.01,
    height_control=None,
    cluster_threshold=0
)
print(f"Uncorrected threshold (p<0.01): z > {threshold_unc01:.2f}")

thresholded_map_fdr, threshold_fdr = threshold_stats_img(
    z_map,
    alpha=0.05,
    height_control='fdr',
    cluster_threshold=0
)
print(f"\nFDR threshold (p<0.05): z > {threshold_fdr:.2f}")

z_map.to_filename("./second_level_analysis/group_curiosity_zmap.nii.gz")
thresholded_map_unc.to_filename("./second_level_analysis/group_curiosity_unc001.nii.gz")
thresholded_map_unc01.to_filename("./second_level_analysis/group_curiosity_unc01.nii.gz")
thresholded_map_fdr.to_filename("./second_level_analysis/group_curiosity_fdr05.nii.gz")

plotting.plot_stat_map(
    z_map,
    title='Curiosity effect (unthresholded)',
    display_mode='ortho',
    colorbar=True,
    threshold=2.0  
)
plotting.show()

plotting.plot_stat_map(
    thresholded_map_unc,
    title='Curiosity effect (p<0.001 uncorrected)',
    threshold=threshold_unc,
    display_mode='ortho',
    colorbar=True
)
plotting.show()

from nilearn.reporting import get_clusters_table

print("\n=== Attempting to extract peaks at p<0.001 uncorrected ===")
try:
    table_unc = get_clusters_table(
        z_map,
        stat_threshold=threshold_unc,
        cluster_threshold=0
    )
    print(table_unc.to_string())
    table_unc.to_csv("./second_level_analysis/group_curiosity_peaks_unc001.csv", index=False)
    print(f"Found {len(table_unc)} clusters")
except:
    print("No clusters found at p<0.001 uncorrected")

print("\n=== Attempting to extract peaks at p<0.01 uncorrected ===")
try:
    table_unc01 = get_clusters_table(
        z_map,
        stat_threshold=threshold_unc01,
        cluster_threshold=0
    )
    print(table_unc01.to_string())
    table_unc01.to_csv("./second_level_analysis/group_curiosity_peaks_unc01.csv", index=False)
    print(f"Found {len(table_unc01)} clusters")
except:
    print("No clusters found at p<0.01 uncorrected")

print("\n=== Attempting to extract peaks at FDR p<0.05 ===")
try:
    table_fdr = get_clusters_table(
        z_map,
        stat_threshold=threshold_fdr,
        cluster_threshold=0
    )
    print(table_fdr.to_string())
    table_fdr.to_csv("./second_level_analysis/group_curiosity_peaks_fdr.csv", index=False)
    print(f"Found {len(table_fdr)} clusters")
except:
    print("No clusters found at FDR p<0.05")

print("\n✓ Analysis complete! Check the z-score statistics above to see if there's signal.")
