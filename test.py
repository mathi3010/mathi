import cv2
from PIL import Image
import numpy as np
from pathlib import Path
 
def calculate_block_rgb(image_path, block_size=60):
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
 
def label_and_save_blocks(image_path, block_size=60, output_base_dir="labeled_blocks"):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
 
    height, width, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    block_number = 1
    output_img = img.copy()
 
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = img[y:y+block_size, x:x+block_size]
            h, w = block.shape[:2]
 
            label = f"{block_number}"
            scale = max(0.4, block_size / 120)
            thickness = max(1, block_size // 60)
            text_size = cv2.getTextSize(label, font, scale, thickness)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
 
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 0), 1)
            cv2.putText(output_img, label, (text_x, text_y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)
 
            block_number += 1
 
    parent_folder = image_path.parent.name
    output_folder = Path(output_base_dir) / parent_folder
    output_folder.mkdir(parents=True, exist_ok=True)
 
    output_file = output_folder / f"{image_path.stem}_labeled.bmp"
    cv2.imwrite(str(output_file), output_img)
    print(f"✅ Saved labeled image → {output_file}")
 
def process_folders(base_input_dir="output", block_size=60):
    base_input = Path(base_input_dir)
    output_reports_dir = Path("block_rgb_reports_combined")
    output_reports_dir.mkdir(parents=True, exist_ok=True)
 
    folder_names = ["trial1", "trial2", "trial3", "trial4"]
 
    for folder_name in folder_names:
        folder_path = base_input / folder_name
        if not folder_path.exists():
            print(f"⚠️ Folder not found: {folder_path}. Skipping.")
            continue
 
        bmp_images = sorted(folder_path.glob("*.bmp"))
        if not bmp_images:
            print(f"⚠️ No .bmp images found in: {folder_path}. Skipping.")
            continue
 
        output_file = output_reports_dir / f"{folder_name}_report.txt"
        with open(output_file, 'w') as f:
            f.write(f"Combined RGB Block Report for Folder: {folder_name}\n")
            f.write("=" * 60 + "\n\n")
 
            for image_path in bmp_images:
                # Calculate and write RGB block report
                block_data = calculate_block_rgb(image_path, block_size)
                f.write("\n".join(block_data))
 
                # Label and save the image blocks visually
                label_and_save_blocks(image_path, block_size)
 
        print(f"✅ Saved report for '{folder_name}' → {output_file}")
 
if __name__ == "__main__":
    process_folders()
  