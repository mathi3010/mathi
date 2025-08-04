from pathlib import Path
import cv2
import numpy as np
 
def process_image_stats_only(image_path, block_size=60, base_save_dir="output_with_stats", trial_folder_name="trialX"):
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
 
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
 
    height, width, _ = img.shape
    lines = []
    block_number = 1
 
    # Create output directory for this trial
    image_save_dir = Path(base_save_dir) / trial_folder_name
    image_save_dir.mkdir(parents=True, exist_ok=True)
 
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = img[y:y+block_size, x:x+block_size]
            gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
 
            mean = np.mean(gray_block)
            median = np.median(gray_block)
            min_val = np.min(gray_block)
            max_val = np.max(gray_block)
            stddev = np.std(gray_block)
 
            lines.append(f"Block_{block_number}")
            lines.append(f"Position: ({x}, {y})")
            lines.append(f"Mean: {mean:.2f}")
            lines.append(f"Median: {median:.2f}")
            lines.append(f"Min: {min_val}")
            lines.append(f"Max: {max_val}")
            lines.append(f"Stddev: {stddev:.2f}")
            lines.append("-" * 50)
 
            block_number += 1
 
    stats_txt_path = image_save_dir / f"{image_path.stem}_stats.txt"
    with open(stats_txt_path, "w") as f:
        f.write("\n".join(lines))
 
    return stats_txt_path
 
 
def process_all_images_stats_only(base_input_dir="output", block_size=60, base_output_dir="output_with_stats"):
    base_input = Path(base_input_dir)
    trial_folders = ["trial1", "trial2", "trial3", "trial4"]
 
    for folder_name in trial_folders:
        folder_path = base_input / folder_name
        if not folder_path.exists():
            print(f"⚠️ Skipping missing folder: {folder_path}")
            continue
 
        bmp_images = list(folder_path.glob("*.bmp"))
        if not bmp_images:
            print(f"⚠️ No BMP images in: {folder_path}")
            continue
 
        for image_path in bmp_images:
            stats_file = process_image_stats_only(
                image_path, block_size, base_output_dir, trial_folder_name=folder_name
            )
            print(f"✅ Stats saved: {stats_file}")
 
if __name__ == "__main__":
    process_all_images_stats_only()
 
 