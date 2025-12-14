from nilearn import plotting
import matplotlib.pyplot as plt

z_map = "mean_contrast.nii.gz"

plotting.plot_stat_map(z_map, threshold=2.0, display_mode='ortho')
plt.show()
