import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os 

from constants import *

def plot_final_frames_grid(F_values, k, base_path="../frames"):

    n_plots = len(F_values)
    
    # Calculate grid dimensions (try to make it as square as possible)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # Flatten axes array for easier iteration
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for idx, F_val in enumerate(F_values):
        frame_path = os.path.join(base_path, f"f{F_val:.3f}", "frame_000001.png")
        
        if os.path.exists(frame_path):
            img = mpimg.imread(frame_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"F = {F_val:.3f}, k = {k:.3f}", fontsize=12)
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, f"Frame not found\nF={F_val:.3f}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    # Hide any unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"comp_k{k:.3f}.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison image saved to {os.path.join(base_path, 'final_frames_comparison.png')}")

if __name__ == "__main__":
    plot_final_frames_grid(F_values=F_values, k=k, base_path="../frames")