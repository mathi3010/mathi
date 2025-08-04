#pixel calculation:
from PIL import Image
import numpy as np
from pathlib import Path
 
def calculate_block_rgb(image_path, block_size=60):
    """
    Returns a list of strings containing average RGB values for each block.
    """
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    img_array = np.array(img)
 
    block_data = []
    block_data.append(f"Image: {image_path.name}")
    block_data.append("Block Coordinates (x, y) | Average RGB")
    block_data.append("-" * 50)
 
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            block = img_array[y:y_end, x:x_end]
            avg_rgb = block.mean(axis=(0, 1))
            line = f"({x:4}, {y:4}) | ({int(avg_rgb[0]):3}, {int(avg_rgb[1]):3}, {int(avg_rgb[2]):3})"
            block_data.append(line)
 
    block_data.append("\n")  # Separate images
    return block_data
 
def process_folders_to_combined_reports(base_input_dir="output", block_size=60):
    base_input = Path(base_input_dir)
    output_dir = Path("block_rgb_reports_combined")
    output_dir.mkdir(parents=True, exist_ok=True)
 
    for folder_name in ["trial1", "trial2", "trial3", "trail4"]:
        folder_path = base_input / folder_name
        if not folder_path.exists():
            print(f"⚠️ Folder not found: {folder_path}")
            continue
 
        output_file = output_dir / f"{folder_name}_report.txt"
        with open(output_file, 'w') as f:
            f.write(f"Combined RGB Block Report for Folder: {folder_name}\n")
            f.write("=" * 60 + "\n\n")
 
            for image_path in sorted(folder_path.glob("*.bmp")):
                block_data = calculate_block_rgb(image_path, block_size)
                f.write("\n".join(block_data))
       
        print(f"✅ Saved report for '{folder_name}' → {output_file}")
 
if __name__ == "__main__":
    process_folders_to_combined_reports()
 
 