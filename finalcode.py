#final code
 
import cv2
import numpy as np
from pathlib import Path
 
# ---------------------------
def step1_extract_thread(input_path, output_path):
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"❌ Step 1: Failed to load image {input_path}")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    white_bg = np.ones_like(image) * 255
    result = np.where(mask[:, :, np.newaxis] == 255, image, white_bg)
    cv2.imwrite(str(output_path), result)
    print(f"✅ Step 1 complete: saved {output_path}")
    return output_path
 
# ---------------------------
def step2_remove_center_ring(input_path, output_path):
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"❌ Step 2: Failed to load image {input_path}")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ring_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_clean = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_clean, [largest], -1, 255, -1)
    image[mask_clean == 255] = [255, 255, 255]
    cv2.imwrite(str(output_path), image)
    print(f"✅ Step 2 complete: saved {output_path}")
    return output_path
 
# ---------------------------
def step3_remove_gray_circle(input_path, output_path, removed_part_path):
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"❌ Step 3: Failed to load image {input_path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    candidate_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if 0.7 < circularity < 1.2:
                candidate_contours.append(cnt)
 
    if not candidate_contours:
        print("⚠️ Step 3: Inner gray circle not detected")
        return None
 
    inner_circle = max(candidate_contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [inner_circle], -1, 255, thickness=cv2.FILLED)
 
    # Save removed center part
    white_bg = np.full_like(img, 255)
    thread_part = cv2.bitwise_and(img, img, mask=mask)
    inv_mask = cv2.bitwise_not(mask)
    white_bg_with_hole = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
    removed_part = cv2.add(thread_part, white_bg_with_hole)
 
    removed_part_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(removed_part_path), removed_part)
    print(f"✅ Step 3: Removed part saved to: {removed_part_path}")
 
    img_result = img.copy()
    img_result[mask == 255] = [255, 255, 255]
    cv2.imwrite(str(output_path), img_result)
    print(f"✅ Step 3 complete: saved {output_path}")
    return output_path
 
# ---------------------------
def process_all_images(input_folder="input_images"):
    input_dir = Path(input_folder)
    output_base = Path("output")
    removed_parts_dir = output_base / "trial4"
    trial_dirs = {
        "trial1": output_base / "trial1",
        "trial2": output_base / "trial2",
        "trial3": output_base / "trial3"
    }
 
    for folder in trial_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    removed_parts_dir.mkdir(parents=True, exist_ok=True)
 
    for file in input_dir.glob("*.bmp"):
        base_name = file.stem
        print(f"\n📌 Processing {file.name}...")
 
        out1 = step1_extract_thread(file, trial_dirs["trial1"] / f"{base_name}_trial1.bmp")
        if not out1: continue
 
        out2 = step2_remove_center_ring(out1, trial_dirs["trial2"] / f"{base_name}_trial2.bmp")
        if not out2: continue
 
        out3 = step3_remove_gray_circle(
            input_path=out2,
            output_path=trial_dirs["trial3"] / f"{base_name}_trial3.bmp",
            removed_part_path=removed_parts_dir / f"{base_name}_removed_center.bmp"
        )
        if not out3: continue
 
# ---------------------------
if __name__ == "__main__":
    process_all_images("input_images")  # Make sure your input images are inside 'input_images'
 
 
