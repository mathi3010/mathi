import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------- Step Functions ---------------------------
def step1_extract_thread(image):
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
    return result

def step2_remove_center_ring(image):
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
    return image

def step3_remove_gray_circle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        print("⚠️ Inner gray circle not detected")
        return image, None

    inner_circle = max(candidate_contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [inner_circle], -1, 255, thickness=cv2.FILLED)

    white_bg = np.full_like(image, 255)
    thread_part = cv2.bitwise_and(image, image, mask=mask)
    inv_mask = cv2.bitwise_not(mask)
    white_bg_with_hole = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
    removed_part = cv2.add(thread_part, white_bg_with_hole)

    # Crop bounding box
    x, y, w, h = cv2.boundingRect(inner_circle)
    cropped_ring = removed_part[y:y+h, x:x+w]

    img_result = image.copy()
    img_result[mask == 255] = [255, 255, 255]

    return img_result, cropped_ring

# --------------------------- Thread Analysis (Block-wise) ---------------------------
def analyze_blocks(image, block_size=50, region_name="Thread"):
    h, w, _ = image.shape
    results = []
    block_id = 1

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            if block.size == 0:
                continue

            mask = np.any(block < 250, axis=-1)  # non-white pixels
            roi = block[mask]

            if roi.size == 0:
                mean_bgr = [255, 255, 255]
                std_bgr = [0, 0, 0]
                min_bgr = [255, 255, 255]
                max_bgr = [255, 255, 255]
                lap_var = 0
                pixel_count = 0
            else:
                mean_bgr = roi.mean(axis=0)
                std_bgr = roi.std(axis=0)
                min_bgr = roi.min(axis=0)
                max_bgr = roi.max(axis=0)
                gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray_block, cv2.CV_64F).var()
                pixel_count = len(roi)

            results.append({
                "Region": region_name,
                "Block_ID": block_id,
                "Block_X": x,
                "Block_Y": y,
                "Pixel_Count": pixel_count,
                "Mean_B": mean_bgr[0], "Mean_G": mean_bgr[1], "Mean_R": mean_bgr[2],
                "Std_B": std_bgr[0], "Std_G": std_bgr[1], "Std_R": std_bgr[2],
                "Min_B": min_bgr[0], "Min_G": min_bgr[1], "Min_R": min_bgr[2],
                "Max_B": max_bgr[0], "Max_G": max_bgr[1], "Max_R": max_bgr[2],
                "Laplacian_Var": lap_var
            })
            block_id += 1

    return results

# --------------------------- Ring Analysis (Single Region) ---------------------------
def analyze_whole_image(image, region_name="Ring"):
    mask = np.any(image < 250, axis=-1)  # non-white pixels
    roi = image[mask]

    if roi.size == 0:
        mean_bgr = [255, 255, 255]
        std_bgr = [0, 0, 0]
        min_bgr = [255, 255, 255]
        max_bgr = [255, 255, 255]
        lap_var = 0
        pixel_count = 0
    else:
        mean_bgr = roi.mean(axis=0)
        std_bgr = roi.std(axis=0)
        min_bgr = roi.min(axis=0)
        max_bgr = roi.max(axis=0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        pixel_count = len(roi)

    return [{
        "Region": region_name,
        "Block_ID": 0,
        "Block_X": 0,
        "Block_Y": 0,
        "Pixel_Count": pixel_count,
        "Mean_B": mean_bgr[0], "Mean_G": mean_bgr[1], "Mean_R": mean_bgr[2],
        "Std_B": std_bgr[0], "Std_G": std_bgr[1], "Std_R": std_bgr[2],
        "Min_B": min_bgr[0], "Min_G": min_bgr[1], "Min_R": min_bgr[2],
        "Max_B": max_bgr[0], "Max_G": max_bgr[1], "Max_R": max_bgr[2],
        "Laplacian_Var": lap_var
    }]

# --------------------------- Main Function ---------------------------
def process_single_image(image_path, output_excel):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load {image_path}")
        return

    # Run image processing pipeline
    img_step1 = step1_extract_thread(img)
    img_step2 = step2_remove_center_ring(img_step1)
    thread_img, ring_img = step3_remove_gray_circle(img_step2)

    # Save outputs
    out_dir = Path("output11_analysis")
    out_dir.mkdir(exist_ok=True)
    thread_path = out_dir / "thread.bmp"
    ring_path = out_dir / "ring.bmp"

    cv2.imwrite(str(thread_path), thread_img)
    print(f"✅ Thread image saved: {thread_path}")
    if ring_img is not None:
        cv2.imwrite(str(ring_path), ring_img)
        print(f"✅ Ring image saved: {ring_path}")

    # Analyze thread in blocks
    thread_results = analyze_blocks(thread_img, block_size=50, region_name="Thread")

    # Analyze ring as single region
    ring_results = []
    if ring_img is not None:
        ring_results = analyze_whole_image(ring_img, region_name="Ring")

    df_thread = pd.DataFrame(thread_results)
    df_ring = pd.DataFrame(ring_results)

    # Save Excel
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df_thread.to_excel(writer, index=False, sheet_name="Thread")
        df_ring.to_excel(writer, index=False, sheet_name="Ring")

    print(f"✅ Analysis complete. Results saved to {output_excel}")

# --------------------------- Run Example ---------------------------
if __name__ == "__main__":
    process_single_image(
        r"D:\project_cone\new4\input_img_11.bmp",
        r"D:\project_cone\new4\output11_analysis.xlsx"
    )
