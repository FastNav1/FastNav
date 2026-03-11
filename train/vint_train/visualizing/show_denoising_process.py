# Visualizing denoised trajectories overlaid on the observation image.

import numpy as np
import matplotlib
# Force Agg backend to avoid GUI windows
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from vint_train.visualizing.action_utils import plot_trajs_and_points_on_image


data_path = "path_to_data_of_denoising_trajectories/your_name.npz"

# Load and verify data
# Expected data format in .npz file:
# - 'img': numpy array of shape (H, W, 3) representing RGB observation image
# - 'trajs': numpy array of shape (S, N, T, 2) where:
#   S = number of denoising steps (e.g., 10)
#   N = number of trajectory samples (e.g., 8)
#   T = trajectory length (e.g., 8)
#   2 = (x, y)

data = np.load(data_path)
img = data['img']
all_trajs = data['trajs']

print(f"Image shape: {img.shape}")
print(f"Trajectories shape: {all_trajs.shape}")

# Visualization parameters， change it in visualize_utils.py
VIZ_IMAGE_SIZE = (1280, 720)

# Process each denoising step
for step in range(len(all_trajs)):
    # Extract trajectories for current step: shape (N, T, 2)
    current_step_trajs = [all_trajs[step][i] for i in range(len(all_trajs[step]))]
    
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot trajectories on image
    plot_trajs_and_points_on_image(
        ax=ax,
        img=img,
        dataset_name="myrobot", 
        list_trajs=current_step_trajs,
        list_points=[]
    )
    
    # Add step indicator
    ax.set_title(f"Denoising Step: {step}")

    # Save output
    output_path = f"path_to_your_folder"
 
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # Close figure to free memory
    
    print(f"Step {step} saved to {output_path}")