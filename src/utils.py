import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image

def parse_filename(filename):
    """Extract t, k, and f values from filename."""
    pattern = r'frame_t([-+]?[0-9]*\.?[0-9]+)_k([-+]?[0-9]*\.?[0-9]+)_f([-+]?[0-9]*\.?[0-9]+)\.png'
    match = re.match(pattern, filename)
    if match:
        return {
            't': float(match.group(1)),
            'k': float(match.group(2)),
            'f': float(match.group(3))
        }
    return None

def group_images_by_f(folder_path):
    """Group image filenames by their f value."""
    groups = defaultdict(list)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            params = parse_filename(filename)
            if params:
                filepath = os.path.join(folder_path, filename)
                groups[params['f']].append({
                    'path': filepath,
                    't': params['t'],
                    'k': params['k'],
                    'filename': filename
                })
    
    # Sort images within each group by t and k for consistent display
    for f_val in groups:
        groups[f_val].sort(key=lambda x: (x['t'], x['k']))
    
    return groups

def plot_images_by_f(folder_path, cols=3, output_folder='output_plots', show=False):
    """Display and save images grouped by f value."""
    groups = group_images_by_f(folder_path)
    
    if not groups:
        print(f"No valid images found in {folder_path}")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for f_val, images in sorted(groups.items()):
        n_images = len(images)
        rows = (n_images + cols - 1) // cols  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        fig.suptitle(f'Images with f = {f_val}', fontsize=16, fontweight='bold')
        
        # Flatten axes array for easier iteration
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, img_info in enumerate(images):
            img = Image.open(img_info['path'])
            axes[idx].imshow(img)
            axes[idx].set_title(f"t={img_info['t']}, k={img_info['k']}", fontsize=10)
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_folder, f'combined_f_{f_val}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)

# Usage
if __name__ == "__main__":
    folder_path = "../frames"  # Change this to your folder path
    output_folder = "../frames/tests"       # Folder where combined images will be saved
    plot_images_by_f(folder_path, cols=10, output_folder=output_folder, show=False)