'''import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================
# --------------------------- Step Functions -------------------
# ==============================================================

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

    x, y, w, h = cv2.boundingRect(inner_circle)
    cropped_ring = removed_part[y:y+h, x:x+w]

    img_result = image.copy()
    img_result[mask == 255] = [255, 255, 255]

    return img_result, cropped_ring

# ==============================================================
# ------------------- Excel Normalization ----------------------
# ==============================================================

def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _to_grayscale_mean(df: pd.DataFrame) -> pd.Series:
    if {'Mean_R','Mean_G','Mean_B'}.issubset(df.columns):
        return 0.299*df['Mean_R'] + 0.587*df['Mean_G'] + 0.114*df['Mean_B']
    raise KeyError("No Mean_Intensity or Mean_R/G/B columns found.")

def prepare_thread_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns: Block_X, Block_Y, Mean_Intensity, Laplacian_Var, Pixel_Count
    """
    df = _strip_cols(df_raw)

    # coords must exist
    for k in ['Block_X','Block_Y']:
        if k not in df.columns:
            raise KeyError(f"Missing key column '{k}' in Thread sheet.")

    if 'Mean_Intensity' not in df.columns:
        df['Mean_Intensity'] = _to_grayscale_mean(df)

    # Laplacian column normalization
    if 'Laplacian_Var' not in df.columns:
        for c in df.columns:
            if c.lower().replace(" ","") in ("laplacian_var","laplacianvariance","lapvar"):
                df.rename(columns={c: 'Laplacian_Var'}, inplace=True)
                break

    if 'Laplacian_Var' not in df.columns:
        raise KeyError("Missing 'Laplacian_Var' column (or equivalent).")

    # Pixel count normalization
    if 'Pixel_Count' not in df.columns:
        for c in df.columns:
            if c.lower().replace(" ","") in ("pixelcount","pixels","count"):
                df.rename(columns={c: 'Pixel_Count'}, inplace=True)
                break

    if 'Pixel_Count' not in df.columns:
        raise KeyError("Missing 'Pixel_Count' column (or equivalent).")

    keep = ['Block_X','Block_Y','Mean_Intensity','Laplacian_Var','Pixel_Count']
    return df[keep].copy()

# ==============================================================
# --------------------- Analysis Functions ---------------------
# ==============================================================

def analyze_blocks(image, block_size=50, region_name="Thread"):
    h, w, _ = image.shape
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = []
    block_id = 1

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = gray_img[y:y+block_size, x:x+block_size]
            if block.size == 0:
                continue

            mask = block < 250  # non-white pixels only
            roi = block[mask]

            if roi.size == 0:
                mean_val, std_val, min_val, max_val, lap_var, pixel_count = 255.0, 0.0, 255, 255, 0.0, 0
            else:
                mean_val = float(roi.mean())
                std_val  = float(roi.std())
                min_val  = int(roi.min())
                max_val  = int(roi.max())
                lap_var  = float(cv2.Laplacian(block, cv2.CV_64F).var())
                pixel_count = int(roi.size)

            results.append({
                "Region": region_name,
                "Block_ID": block_id,
                "Block_X": x,
                "Block_Y": y,
                "Pixel_Count": pixel_count,
                "Mean_Intensity": mean_val,
                "Std_Intensity": std_val,
                "Min_Intensity": min_val,
                "Max_Intensity": max_val,
                "Laplacian_Var": lap_var
            })
            block_id += 1

    return results

def analyze_whole_image(image, region_name="Ring"):
    if image is None:
        return [{
            "Region": region_name, "Block_ID": 0, "Block_X": 0, "Block_Y": 0,
            "Pixel_Count": 0, "Mean_Intensity": 255.0, "Std_Intensity": 0.0,
            "Min_Intensity": 255, "Max_Intensity": 255, "Laplacian_Var": 0.0
        }]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < 250
    roi = gray[mask]

    if roi.size == 0:
        mean_val, std_val, min_val, max_val, lap_var, pixel_count = 255.0, 0.0, 255, 255, 0.0, 0
    else:
        mean_val = float(roi.mean())
        std_val  = float(roi.std())
        min_val  = int(roi.min())
        max_val  = int(roi.max())
        lap_var  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        pixel_count = int(roi.size)

    return [{
        "Region": region_name,
        "Block_ID": 0, "Block_X": 0, "Block_Y": 0,
        "Pixel_Count": pixel_count,
        "Mean_Intensity": mean_val, "Std_Intensity": std_val,
        "Min_Intensity": min_val, "Max_Intensity": max_val,
        "Laplacian_Var": lap_var
    }]

# ==============================================================
# ----------------------- Geometry (Ring) ----------------------
# ==============================================================

def ring_geometry_metrics(ring_img):
    """
    Returns dict with area, perimeter, circularity, eq_diameter (pixels).
    Uses largest non-white contour.
    """
    if ring_img is None:
        return {"area": 0, "perimeter": 0, "circularity": 0.0, "eq_diameter": 0.0}

    gray = cv2.cvtColor(ring_img, cv2.COLOR_BGR2GRAY)
    _, binv = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area": 0, "perimeter": 0, "circularity": 0.0, "eq_diameter": 0.0}
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    circ = float(4 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
    eq_d = float(np.sqrt(4 * area / np.pi)) if area > 0 else 0.0
    return {"area": float(area), "perimeter": float(peri), "circularity": circ, "eq_diameter": eq_d}

# ==============================================================
# --------------------------- SSIM -----------------------------
# ==============================================================

def ssim_block(a, b):
    """Compute SSIM between two same-sized grayscale blocks (uint8)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(a, (3, 3), 1.5)
    mu2 = cv2.GaussianBlur(b, (3, 3), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(a * a, (3, 3), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b * b, (3, 3), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(a * b, (3, 3), 1.5) - mu1_mu2
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-12)
    return float(ssim_map.mean())

def blockwise_ssim(ref_img, test_img, block_size=50):
    """
    Returns DataFrame with Block_X, Block_Y, SSIM per block on grayscale.
    If sizes differ, resizes test to ref size.
    """
    hRef, wRef = ref_img.shape[:2]
    hT, wT = test_img.shape[:2]
    if (hRef, wRef) != (hT, wT):
        test_img = cv2.resize(test_img, (wRef, hRef), interpolation=cv2.INTER_AREA)

    gRef = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gT   = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    rows = []
    for y in range(0, hRef, block_size):
        for x in range(0, wRef, block_size):
            br = gRef[y:y+block_size, x:x+block_size]
            bt = gT[y:y+block_size, x:x+block_size]
            if br.size == 0 or bt.size == 0:
                continue
            white_r = (br >= 250).mean()
            white_t = (bt >= 250).mean()
            ssim_val = 1.0 if (white_r > 0.95 and white_t > 0.95) else ssim_block(br, bt)
            rows.append({"Block_X": x, "Block_Y": y, "SSIM": ssim_val})
    return pd.DataFrame(rows)

# ==============================================================
# ----------------------- Visualization ------------------------
# ==============================================================

def make_thread_quality_overlay(thread_img, df_thread_result, block_size=50, alpha=0.35):
    overlay = thread_img.copy()
    h, w, _ = overlay.shape
    for row in df_thread_result.itertuples(index=False):
        x, y = int(row.Block_X), int(row.Block_Y)
        x2, y2 = min(x + block_size, w), min(y + block_size, h)
        if x >= w or y >= h:
            continue
        color = (0, 200, 0) if row.Quality == "Good" else (0, 0, 255)
        cv2.rectangle(overlay, (x, y), (x2, y2), color, thickness=-1)
    blended = cv2.addWeighted(overlay, alpha, thread_img, 1 - alpha, 0)
    return blended

def make_thread_intensity_heatmap(thread_img, df_thread, block_size=50):
    h, w, _ = thread_img.shape
    grid_h = (h + block_size - 1) // block_size
    grid_w = (w + block_size - 1) // block_size
    heat_grid = np.full((grid_h, grid_w), 255, dtype=np.uint8)

    for row in df_thread.itertuples(index=False):
        gx = row.Block_X // block_size
        gy = row.Block_Y // block_size
        if 0 <= gy < grid_h and 0 <= gx < grid_w:
            val = np.clip(int(round(row.Mean_Intensity)), 0, 255)
            heat_grid[gy, gx] = val

    heat_img = cv2.resize(heat_grid, (w, h), interpolation=cv2.INTER_NEAREST)
    heat_color = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
    return heat_color

# ==============================================================
# -------------------- Processing + Comparison -----------------
# ==============================================================

def process_and_compare(
    input_image,
    ref_excel,
    test_excel,
    comp_output_excel,
    block_size=50,
    ref_image=None  # path to reference cone image (recommended)
):
    # ---- Process test image ----
    img = cv2.imread(input_image)
    if img is None:
        print(f"❌ Error: Could not load {input_image}")
        return

    img_step1 = step1_extract_thread(img)
    img_step2 = step2_remove_center_ring(img_step1)
    thread_img, ring_img = step3_remove_gray_circle(img_step2)

    out_dir = Path("output_images6")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "thread.bmp"), thread_img)
    if ring_img is not None:
        cv2.imwrite(str(out_dir / "ring.bmp"), ring_img)

    # ---- Save grayscale analysis Excel (test) ----
    df_thread = pd.DataFrame(analyze_blocks(thread_img, block_size=block_size, region_name="Thread"))
    df_ring = pd.DataFrame(analyze_whole_image(ring_img, region_name="Ring")) if ring_img is not None else pd.DataFrame()

    with pd.ExcelWriter(test_excel, engine="openpyxl") as writer:
        df_thread.to_excel(writer, index=False, sheet_name="Thread")
        if not df_ring.empty:
            df_ring.to_excel(writer, index=False, sheet_name="Ring")

    print(f"✅ Analysis saved: {test_excel}")

    # ---- Optional: process reference image for SSIM + ring geometry ----
    ref_thread_img = None
    ref_ring_img = None
    if ref_image is not None:
        rimg = cv2.imread(ref_image)
        if rimg is None:
            print(f"⚠️ Warning: Could not load ref_image: {ref_image} — SSIM/geometry skipped.")
        else:
            r1 = step1_extract_thread(rimg)
            r2 = step2_remove_center_ring(r1)
            ref_thread_img, ref_ring_img = step3_remove_gray_circle(r2)

    # ---- Load + normalize thread sheets (handles old RGB refs) ----
    df_ref_thread_raw = pd.read_excel(ref_excel, sheet_name="Thread")
    df_test_thread_raw = pd.read_excel(test_excel, sheet_name="Thread")
    df_ref_thread = prepare_thread_df(df_ref_thread_raw)
    df_test_thread = prepare_thread_df(df_test_thread_raw)

    # ---- Ring sheets optional ----
    try:
        df_ref_ring = pd.read_excel(ref_excel, sheet_name="Ring")
    except Exception:
        df_ref_ring = pd.DataFrame()
    try:
        df_test_ring = pd.read_excel(test_excel, sheet_name="Ring")
    except Exception:
        df_test_ring = pd.DataFrame()

    # ---- Thresholds (tune as needed) ----
    thresholds = {
        "mean_int_diff": 20,      # intensity units (0-255)
        "lap_var_diff":  30,      # Laplacian variance difference
        "pixel_diff":    50,      # pixel-count difference
        "ssim_min":      0.75,    # block SSIM minimum
        "norm_int":      20.0,    # normalization for score fusion
        "norm_lap":      30.0,
        "norm_pix":      50.0,
        "score_T":       1.0
    }
    weights = {"ssim": 0.6, "lap": 0.2, "int": 0.15, "pix": 0.05}
    fail_threshold = 0.1  # <=10% "Bad" blocks allowed

    # ---- Optional SSIM per-block ----
    df_ssim = None
    if ref_thread_img is not None:
        df_ssim = blockwise_ssim(ref_thread_img, thread_img, block_size=block_size)

    # ---- Merge test vs ref by Block coords ----
    key_cols = ["Block_X", "Block_Y"]
    metrics = ["Mean_Intensity","Laplacian_Var","Pixel_Count"]
    df_ref_trim  = df_ref_thread[key_cols + metrics].copy()
    df_test_trim = df_test_thread[key_cols + metrics].copy()
    df_join = pd.merge(df_ref_trim, df_test_trim, on=key_cols, suffixes=("_ref","_test"))
    if df_ssim is not None:
        df_join = pd.merge(df_join, df_ssim, on=key_cols, how="left")

    # ---- Scoring ----
    def score_row(r):
        dint = abs(r["Mean_Intensity_test"] - r["Mean_Intensity_ref"]) / thresholds["norm_int"]
        dlap = abs(r["Laplacian_Var_test"] - r["Laplacian_Var_ref"]) / thresholds["norm_lap"]
        dpix = abs(r["Pixel_Count_test"]   - r["Pixel_Count_ref"])   / thresholds["norm_pix"]
        if "SSIM" in r and not pd.isna(r["SSIM"]):
            ssim_term = (1.0 - float(r["SSIM"]))
            ssim_bad = (r["SSIM"] < thresholds["ssim_min"])
        else:
            ssim_term = min(1.0, dint * 0.5)  # fallback penalty
            ssim_bad = False
        score = weights["ssim"]*ssim_term + weights["lap"]*dlap + weights["int"]*dint + weights["pix"]*dpix
        hard_fail = ssim_bad or \
                    abs(r["Mean_Intensity_test"] - r["Mean_Intensity_ref"]) > thresholds["mean_int_diff"] or \
                    abs(r["Laplacian_Var_test"] - r["Laplacian_Var_ref"]) > thresholds["lap_var_diff"] or \
                    abs(r["Pixel_Count_test"]   - r["Pixel_Count_ref"])   > thresholds["pixel_diff"]
        quality = "Bad" if (score > thresholds["score_T"] or hard_fail) else "Good"
        return score, quality

    scores, quals = [], []
    for row in df_join.itertuples(index=False):
        r = row._asdict() if hasattr(row, "_asdict") else dict(row._mapping)
        s, q = score_row(r)
        scores.append(s); quals.append(q)

    df_thread_result = df_test_thread.copy()
    df_thread_result = pd.merge(
        df_thread_result,
        pd.DataFrame({
            "Block_X": df_join["Block_X"],
            "Block_Y": df_join["Block_Y"],
            "Score":   scores,
            "Quality": quals
        }),
        on=["Block_X","Block_Y"], how="left"
    )

    # ---- Ring geometry comparison ----
    ring_ok = True
    ring_comment = "Ring geometry OK"
    test_geom = ring_geometry_metrics(ring_img)
    if ref_thread_img is not None and ref_ring_img is not None:
        ref_geom = ring_geometry_metrics(ref_ring_img)
        circ_ok = (abs(test_geom["circularity"] - ref_geom["circularity"]) <= 0.10)
        diam_tol = max(10.0, 0.05 * ref_geom["eq_diameter"])
        diam_ok = (abs(test_geom["eq_diameter"] - ref_geom["eq_diameter"]) <= diam_tol)
        ring_ok = circ_ok and diam_ok
        if not circ_ok or not diam_ok:
            ring_comment = f"Ring mismatch (Δcirc={abs(test_geom['circularity']-ref_geom['circularity']):.3f}, Δdiam={abs(test_geom['eq_diameter']-ref_geom['eq_diameter']):.1f}px)"
    else:
        ring_comment = "Ring geometry skipped (no ref_image)"

    # ---- Overall decision ----
    def part_quality(df):
        total = len(df)
        if total == 0:
            return True
        bad = (df["Quality"] == "Bad").sum()
        return bad / total <= fail_threshold

    thread_ok = part_quality(df_thread_result)

    if thread_ok and ring_ok:
        cone_quality = "Good Cone"
    elif not thread_ok and ring_ok:
        cone_quality = "Bad Cone (Thread mismatched)"
    elif thread_ok and not ring_ok:
        cone_quality = "Bad Cone (Ring mismatched)"
    else:
        cone_quality = "Bad Cone (Thread + Ring mismatched)"

    print(f"✅ Overall Result: {cone_quality} | {ring_comment}")

    # ---- Visualizations ----
    overlay_img = make_thread_quality_overlay(thread_img, df_thread_result.fillna({"Quality":"Good"}), block_size=block_size, alpha=0.35)
    cv2.imwrite(str(out_dir / "thread_quality_overlay.png"), overlay_img)
    heatmap_img = make_thread_intensity_heatmap(thread_img, df_thread, block_size=block_size)
    cv2.imwrite(str(out_dir / "thread_intensity_heatmap.png"), heatmap_img)
    print(f"🖼️ Saved visuals to: {out_dir / 'thread_quality_overlay.bmp'} and {out_dir / 'thread_intensity_heatmap.bmp'}")

    # ---- Save comparison sheets ----
    with pd.ExcelWriter(comp_output_excel, engine="openpyxl") as writer:
        df_thread_result.to_excel(writer, index=False, sheet_name="Thread_Comparison")
        pd.DataFrame([{
            "circularity": test_geom["circularity"],
            "eq_diameter_px": test_geom["eq_diameter"],
            "Comment": ring_comment
        }]).to_excel(writer, index=False, sheet_name="Ring_Comparison")

    print(f"✅ Comparison results saved: {comp_output_excel}")

# ==============================================================
# ----------------------------- MAIN ----------------------------
# ==============================================================

if __name__ == "__main__":
    process_and_compare(
        input_image=r"D:\project_cone\new4\input_img_2.bmp",
        ref_excel=r"D:\project_cone\new4\output1_analysis.xlsx",
        test_excel=r"D:\project_cone\new4\output1_analysis.xlsx",
        comp_output_excel=r"D:\project_cone\new4\comparison_result_Overall01.xlsx",
        block_size=50,
        ref_image=r"D:\project_cone\new4\ref_img_6.bmp"  # optional but recommended
    )'''




'''import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ======== SETTINGS (use ring.bmp in current folder) ========
INPUT_IMAGE = "ring.bmp"
OUT_DIR = "qc_outputs"
WHITE_THR = 250        # >= treated as background
TARGET_CELLS = 14      # grid density (auto block size)
SCORE_T = 0.90         # fused-score threshold (lower = stricter)
W_SSIM, W_INT, W_LAP = 0.60, 0.25, 0.15
CIRC_TOL = 0.10        # |circularity - 1.0| tolerance
MIN_PIX_FRAC = 0.02    # min content per block to judge
# ===========================================================

def ssim_block(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1 = (0.01 * 255) ** 2; C2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(a, (3,3), 1.5); mu2 = cv2.GaussianBlur(b, (3,3), 1.5)
    mu1_sq = mu1*mu1; mu2_sq = mu2*mu2; mu1_mu2 = mu1*mu2
    sigma1_sq = cv2.GaussianBlur(a*a, (3,3), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b*b, (3,3), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(a*b, (3,3), 1.5) - mu1_mu2
    num = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return float((num / (den + 1e-12)).mean())

def ring_geometry_metrics(img_bgr, white_thr=WHITE_THR):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binv = cv2.threshold(gray, white_thr, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c)); peri = float(cv2.arcLength(c, True))
    circularity = float(4*np.pi*area/(peri*peri)) if peri > 0 else 0.0
    eq_diam = float(np.sqrt(4*area/np.pi)) if area > 0 else 0.0
    h, w = gray.shape
    mask = np.zeros((h,w), np.uint8); cv2.drawContours(mask, [c], -1, 255, -1)
    (xc, yc), rad = cv2.minEnclosingCircle(c)
    tmpl = np.zeros((h,w), np.uint8)
    cv2.circle(tmpl, (int(round(xc)), int(round(yc))), max(1,int(round(rad))), 255, -1)
    return area, peri, circularity, eq_diam, mask, tmpl

def analyze_blocks(img_bgr, ring_mask, tmpl_mask, target_cells=TARGET_CELLS, white_thr=WHITE_THR):
    h, w = ring_mask.shape
    block = max(16, int(round(min(h, w)/target_cells))); block = int(np.clip(block, 16, 96))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rows = []
    for y in range(0, h, block):
        for x in range(0, w, block):
            y2, x2 = min(y+block, h), min(x+block, w)
            gb = gray[y:y2, x:x2]; mb = ring_mask[y:y2, x:x2]; tb = tmpl_mask[y:y2, x:x2]
            content = gb < white_thr; roi = gb[content]
            if roi.size == 0:
                mean_int, lap_var, pix = 255.0, 0.0, 0
            else:
                mean_int = float(roi.mean())
                lap_var = float(cv2.Laplacian(gb, cv2.CV_64F).var())
                pix = int(roi.size)
            empty_o = (mb == 0).mean() > 0.95; empty_t = (tb == 0).mean() > 0.95
            ssim_b = 1.0 if (empty_o and empty_t) else ssim_block(mb, tb)
            rows.append({"Block_X": x, "Block_Y": y, "Pixel_Count": pix,
                         "Mean_Intensity": mean_int, "Laplacian_Var": lap_var,
                         "SSIM_to_Ideal": ssim_b})
    return pd.DataFrame(rows), block

def compute_scores(df, block_size):
    eps = 1e-6; df = df.copy(); valid = df["Pixel_Count"] > 0
    if valid.any():
        med_int = float(df.loc[valid, "Mean_Intensity"].median())
        mad_int = float((np.abs(df.loc[valid, "Mean_Intensity"] - med_int)).median())
        med_lap = float(df.loc[valid, "Laplacian_Var"].median())
        mad_lap = float((np.abs(df.loc[valid, "Laplacian_Var"] - med_lap)).median())
    else:
        med_int, mad_int, med_lap, mad_lap = 255.0, 1.0, 0.0, 1.0
    scale_int = max(mad_int*1.4826, 5.0); scale_lap = max(mad_lap*1.4826, 5.0)
    z_int = np.abs((df["Mean_Intensity"] - med_int) / (scale_int + eps))
    z_lap = np.abs((df["Laplacian_Var"] - med_lap) / (scale_lap + eps))
    ssim_pen = 1.0 - df["SSIM_to_Ideal"]
    score = W_SSIM*ssim_pen + W_INT*z_int + W_LAP*z_lap
    df["Z_Intensity"] = z_int; df["Z_Laplacian"] = z_lap; df["Score"] = score
    min_pix = max(20, int(MIN_PIX_FRAC * (block_size*block_size)))
    df["Quality"] = np.where((df["Pixel_Count"] >= min_pix) & (df["Score"] <= SCORE_T), "Good", "Bad")
    return df

def build_visuals(img_bgr, df, block_size):
    h, w = img_bgr.shape[:2]
    # overlay
    overlay = img_bgr.copy()
    for r in df.itertuples(index=False):
        x, y = int(r.Block_X), int(r.Block_Y)
        x2, y2 = min(x+block_size, w), min(y+block_size, h)
        color = (0,200,0) if r.Quality == "Good" else (0,0,255)
        cv2.rectangle(overlay, (x,y), (x2,y2), color, -1)
    blend = cv2.addWeighted(overlay, 0.35, img_bgr, 0.65, 0)
    # heatmap
    grid_h = (h + block_size - 1)//block_size
    grid_w = (w + block_size - 1)//block_size
    heat = np.full((grid_h, grid_w), 255, np.uint8)
    for r in df.itertuples(index=False):
        gx = int(r.Block_X // block_size); gy = int(r.Block_Y // block_size)
        val = int(np.clip(round(r.Mean_Intensity), 0, 255))
        heat[gy, gx] = val
    heat_img = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
    heat_color = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
    return blend, heat_color

def main():
    in_path = Path(INPUT_IMAGE)
    if not in_path.exists():
        print(f"❌ Not found: {in_path}"); return
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(in_path))
    h, w = img.shape[:2]

    geom = ring_geometry_metrics(img, WHITE_THR)
    if geom is None:
        print("❌ No ring contour found."); return
    area, peri, circularity, eq_diam, ring_mask, tmpl_mask = geom

    ssim_global = ssim_block(ring_mask, tmpl_mask)
    df, block = analyze_blocks(img, ring_mask, tmpl_mask, TARGET_CELLS, WHITE_THR)
    df = compute_scores(df, block)

    geom_ok = (abs(circularity - 1.0) <= CIRC_TOL)
    overlay, heatmap = build_visuals(img, df, block)

    # save
    out_overlay = Path(OUT_DIR) / "thread_quality_overlay.png"
    out_heat    = Path(OUT_DIR) / "thread_intensity_heatmap.png"
    out_csv     = Path(OUT_DIR) / "block_metrics.csv"
    cv2.imwrite(str(out_overlay), overlay)
    cv2.imwrite(str(out_heat), heatmap)
    df.to_csv(out_csv, index=False)

    bad = int((df["Quality"] == "Bad").sum()); total = int(len(df))
    print("=== QC SUMMARY ===")
    print(f"Image: {in_path}  Size: {h}x{w}")
    print(f"Geometry: area={area:.0f}, peri={peri:.2f}, circ={circularity:.3f} ({'OK' if geom_ok else 'Mismatch'}), eq_diam={eq_diam:.1f}px")
    print(f"Global SSIM (mask vs ideal): {ssim_global:.4f}")
    print(f"Blocks: {bad}/{total} Bad, block_size={block}px")
    print(f"Saved:\n - {out_overlay}\n - {out_heat}\n - {out_csv}")

if __name__ == "__main__":
    main()'''

import sys, json, shutil
import cv2, numpy as np, pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QMessageBox, QListWidget, QListWidgetItem, QLineEdit, QFormLayout,
    QMenu, QInputDialog, QFrame, QGridLayout
)
from PyQt5.QtCore import QTimer, Qt, QSize, QPoint
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

# ============================ Processing ============================

def step1_extract_thread(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
    white = np.full_like(image, 255)
    return np.where(mask[:,:,None] == 255, image, white)

def step2_remove_center_ring(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, ring = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    ring = cv2.morphologyEx(ring, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    cnts, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    img[mask==255] = [255,255,255]
    return img

def step3_remove_gray_circle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 3000:
            per = cv2.arcLength(c, True); 
            if per == 0: continue
            circ = 4*np.pi*area/(per*per)
            if 0.7 < circ < 1.2: cand.append(c)
    if not cand:
        return image.copy(), None
    inner = max(cand, key=cv2.contourArea)
    m = np.zeros_like(gray)
    cv2.drawContours(m, [inner], -1, 255, cv2.FILLED)
    white = np.full_like(image, 255)
    thread_part = cv2.bitwise_and(image, image, mask=m)
    inv = cv2.bitwise_not(m)
    removed = cv2.add(thread_part, cv2.bitwise_and(white, white, mask=inv))
    x,y,w,h = cv2.boundingRect(inner)
    ring_crop = removed[y:y+h, x:x+w]
    res = image.copy(); res[m==255] = [255,255,255]
    return res, ring_crop

def analyze_blocks(image, block=40, region="Thread"):
    h,w,_ = image.shape; rows=[]; bid=1
    for y in range(0,h,block):
        for x in range(0,w,block):
            blk = image[y:y+block, x:x+block]
            if blk.size==0: continue
            mask = np.any(blk<250, axis=-1); roi = blk[mask]
            if roi.size==0:
                mean=[255,255,255]; std=[0,0,0]; mn=[255]*3; mx=[255]*3; lap=0; px=0
            else:
                mean=roi.mean(axis=0); std=roi.std(axis=0); mn=roi.min(axis=0); mx=roi.max(axis=0)
                lap = cv2.Laplacian(cv2.cvtColor(blk, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                px = len(roi)
            rows.append({"Region":region,"Block_ID":bid,"Block_X":x,"Block_Y":y,"Pixel_Count":px,
                         "Mean_B":mean[0],"Mean_G":mean[1],"Mean_R":mean[2],
                         "Std_B":std[0],"Std_G":std[1],"Std_R":std[2],
                         "Min_B":mn[0],"Min_G":mn[1],"Min_R":mn[2],
                         "Max_B":mx[0],"Max_G":mx[1],"Max_R":mx[2],
                         "Laplacian_Var":lap})
            bid+=1
    return pd.DataFrame(rows)

def analyze_whole(image, region="Ring"):
    if image is None or image.size==0:
        return pd.DataFrame([{"Region":region,"Block_ID":0,"Block_X":0,"Block_Y":0,"Pixel_Count":0,
                              "Mean_B":255,"Mean_G":255,"Mean_R":255,"Std_B":0,"Std_G":0,"Std_R":0,
                              "Min_B":255,"Min_G":255,"Min_R":255,"Max_B":255,"Max_G":255,"Max_R":255,
                              "Laplacian_Var":0}])
    mask=np.any(image<250,axis=-1); roi=image[mask]
    if roi.size==0:
        mean=[255]*3; std=[0]*3; mn=[255]*3; mx=[255]*3; lap=0; px=0
    else:
        mean=roi.mean(axis=0); std=roi.std(axis=0); mn=roi.min(axis=0); mx=roi.max(axis=0)
        lap=cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(); px=len(roi)
    return pd.DataFrame([{"Region":region,"Block_ID":0,"Block_X":0,"Block_Y":0,"Pixel_Count":px,
                          "Mean_B":mean[0],"Mean_G":mean[1],"Mean_R":mean[2],
                          "Std_B":std[0],"Std_G":std[1],"Std_R":std[2],
                          "Min_B":mn[0],"Min_G":mn[1],"Min_R":mn[2],
                          "Max_B":mx[0],"Max_G":mx[1],"Max_R":mx[2],
                          "Laplacian_Var":lap}])

def write_analysis_excel(path, df_thread, df_ring):
    with pd.ExcelWriter(path, engine="openpyxl") as wr:
        df_thread.to_excel(wr, index=False, sheet_name="Thread")
        df_ring.to_excel(wr, index=False, sheet_name="Ring")

THRESH = {"mean_diff":20, "lap_var_diff":30, "pixel_diff":50}
FAIL_ALLOWED = 0.10

def compare_df(df_ref, df_test):
    m = df_ref.merge(df_test, on="Block_ID", suffixes=("_ref","_test"))
    def judge(r):
        for ch in ["R","G","B"]:
            if abs(r[f"Mean_{ch}_ref"]-r[f"Mean_{ch}_test"])>THRESH["mean_diff"]: return "Bad"
        if abs(r["Laplacian_Var_ref"]-r["Laplacian_Var_test"])>THRESH["lap_var_diff"]: return "Bad"
        if abs(r["Pixel_Count_ref"]-r["Pixel_Count_test"])>THRESH["pixel_diff"]: return "Bad"
        return "Good"
    m["Quality"]=m.apply(judge, axis=1)
    return m

def verdict(thread_cmp, ring_cmp):
    def ok(df):
        if df is None or len(df)==0: return True
        bad=(df["Quality"]=="Bad").sum()
        return (bad/len(df))<=FAIL_ALLOWED
    t_ok=ok(thread_cmp); r_ok=ok(ring_cmp)
    if t_ok and r_ok: return "Good"
    if not t_ok and r_ok: return "Bad (Thread mismatched)"
    if t_ok and not r_ok: return "Bad (Ring mismatched)"
    return "Bad (Thread + Ring mismatched)"

def process_image_to_excels(img_bgr, out_dir, base):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    img1 = step1_extract_thread(img_bgr)
    img2 = step2_remove_center_ring(img1)
    thread_img, ring_img = step3_remove_gray_circle(img2)
    thread_path = out / f"{base}_thread.bmp"; cv2.imwrite(str(thread_path), thread_img)
    ring_path = None
    if ring_img is not None:
        ring_path = out / f"{base}_ring.bmp"; cv2.imwrite(str(ring_path), ring_img)
    df_thread = analyze_blocks(thread_img, block=40, region="Thread")
    df_ring   = analyze_whole(ring_img, region="Ring")
    xlsx_path = out / f"{base}_analysis.xlsx"; write_analysis_excel(str(xlsx_path), df_thread, df_ring)
    return (str(thread_path), (str(ring_path) if ring_path else None), str(xlsx_path)), (df_thread, df_ring)

# ============================ Metadata ============================

def ref_dirs():
    root = Path.cwd()/ "reports"
    refs = root / "references"
    root.mkdir(exist_ok=True); refs.mkdir(parents=True, exist_ok=True)
    return root, refs

def meta_path(name): 
    _, refs = ref_dirs(); return refs / f"{name}.json"

def save_meta(name, d):
    mp = meta_path(name); mp.write_text(json.dumps(d, indent=2))

def load_meta(name):
    mp = meta_path(name)
    if mp.exists():
        try: return json.loads(mp.read_text())
        except: return {}
    return {}

def count_goods_and_recent(n_recent=6):
    _, refs = ref_dirs()
    files = sorted(refs.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    names = [p.stem for p in files]
    return len(names), names[:n_recent]

# ============================ UI Styles (Red-wine, no solid blocks) ============================

REDWINE = "#7B1E3A"
REDWINE_DARK = "#5E172C"
TEXT = "#1F2937"
MUTED = "#6B7280"
BORDER = "#E5E7EB"
BG = "#F8FAFC"
CARD_BG = "#FFFFFF"

APP_STYLE = f"""
QWidget {{
  background: {BG};
  color: {TEXT};
  font-family: 'Segoe UI';
  font-size: 11pt;
}}
/* Headings */
QLabel.title {{
  color: {TEXT};
  font-weight: 700;
}}
QLabel.sub {{
  color: {MUTED};
}}
/* Cards: white only, thin border (no colored blocks) */
QLabel.card {{
  background: {CARD_BG};
  border: 1px solid {BORDER};
  border-radius: 12px;
  padding: 10px;
}}
QFrame.kpi {{
  background: transparent;
  border: 1px solid {BORDER};
  border-radius: 10px;
}}
/* Outline chips/badges */
QLabel.badge {{
  background: transparent;
  border: 1px solid {BORDER};
  border-radius: 999px;
  padding: 6px 10px;
  color: {TEXT};
}}
/* Buttons: outline/ghost with red-wine accent; no solid fills */
QPushButton {{
  background: transparent;
  color: {TEXT};
  border: 1px solid {BORDER};
  border-radius: 10px;
  padding: 8px 14px;
  font-weight: 600;
}}
QPushButton:hover {{
  background: rgba(0,0,0,0.03);
}}
QPushButton#primary {{
  border: 1px solid {REDWINE};
  color: {REDWINE};
}}
QPushButton#primary:hover {{
  background: rgba(123,30,58,0.06);
}}
QPushButton#success {{
  border: 1px solid {REDWINE};
  color: {REDWINE};
}}
QPushButton#success:hover {{
  background: rgba(123,30,58,0.06);
}}
QPushButton#danger {{
  border: 1px solid #DC2626;
  color: #DC2626;
}}
QPushButton#danger:hover {{
  background: rgba(220,38,38,0.06);
}}
QPushButton#muted {{
  border: 1px solid {BORDER};
  color: {MUTED};
}}
/* Lists/inputs kept white, subtle borders */
QListWidget, QLineEdit {{
  background: {CARD_BG};
  border: 1px solid {BORDER};
  border-radius: 10px;
  padding: 6px 8px;
}}
"""
# ============================ Config ============================
# Print only final verdict in terminal
FINAL_TO_TERMINAL_ONLY = True

# ============================ Pages ============================

class TrainPage(QWidget):
    def __init__(self, refresh_refs, go_back):
        super().__init__()
        self.refresh_refs = refresh_refs; self.go_back = go_back
        self.root, self.refs = ref_dirs()
        self.setStyleSheet(APP_STYLE)

        lay = QVBoxLayout()
        title = QLabel("Train – Capture GOOD and Name it")
        title.setObjectName("title")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold)); title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        form = QFormLayout()
        self.name_edit = QLineEdit(self.next_name()); self.name_edit.setPlaceholderText("e.g., 1, 2, 3 ...")
        form.addRow("GOOD name:", self.name_edit); lay.addLayout(form)

        self.cam = QLabel(); self.cam.setFixedSize(480,360)
        self.cam.setStyleSheet(f"border:1px solid {BORDER}; border-radius:12px;")
        lay.addWidget(self.cam, alignment=Qt.AlignCenter)

        self.status = QLabel("ready"); self.status.setProperty("class","card"); self.status.setObjectName("card")
        lay.addWidget(self.status, alignment=Qt.AlignCenter)

        row = QHBoxLayout()
        self.btn_cap = QPushButton("Capture & Save GOOD"); self.btn_cap.setObjectName("success")
        self.btn_cap.clicked.connect(self.capture_and_save); row.addWidget(self.btn_cap)
        back = QPushButton("Back"); back.setObjectName("danger"); back.clicked.connect(self.close_back); row.addWidget(back)
        lay.addLayout(row)
        self.setLayout(lay)

        self.cap = cv2.VideoCapture(0); self.timer = QTimer(); self.timer.timeout.connect(self._upd); self.timer.start(30)

    def next_name(self):
        names=[]
        for p in (self.refs).glob("*.xlsx"):
            try: names.append(int(p.stem))
            except: pass
        return str(max(names)+1) if names else "1"

    def _upd(self):
        ok, f = self.cap.read()
        if ok:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB); h,w,ch = rgb.shape
            self.cam.setPixmap(QPixmap.fromImage(QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)))

    def capture_and_save(self):
        name = self.name_edit.text().strip()
        if not name: 
            QMessageBox.warning(self,"Name?","Enter name like 1, 2, 3..."); return
        ok, frame = self.cap.read()
        if not ok: QMessageBox.critical(self,"Camera","Capture failed."); return
        (thread_path, ring_path, _), (df_t, df_r) = process_image_to_excels(frame, self.root/"train", f"good_{name}")
        xlsx = self.refs / f"{name}.xlsx"; write_analysis_excel(str(xlsx), df_t, df_r)
        # thumbnail
        thumb = self.refs / f"{name}.bmp"
        img = cv2.imread(thread_path) if Path(thread_path).exists() else frame
        cv2.imwrite(str(thumb), cv2.resize(img,(160,120), interpolation=cv2.INTER_AREA))
        # meta init
        save_meta(name, {"last_fail_pct": None})
        self.status.setText(f"Saved GOOD ✅  name={name}\n{xlsx}")
        self.name_edit.setText(self.next_name())
        self.refresh_refs()

    def close_back(self):
        if self.cap.isOpened(): self.cap.release()
        self.timer.stop(); self.go_back()

    def closeEvent(self,e):
        if self.cap.isOpened(): self.cap.release()
        e.accept()

class PredictionPage(QWidget):
    def __init__(self, go_back):
        super().__init__()
        self.root, self.refs = ref_dirs()
        self.setStyleSheet(APP_STYLE)

        main = QHBoxLayout()

        # LEFT: references
        left = QVBoxLayout()
        t = QLabel("GOOD References"); t.setFont(QFont("Segoe UI", 14, QFont.Bold)); t.setAlignment(Qt.AlignCenter)
        left.addWidget(t)
        self.list = QListWidget(); self.list.setIconSize(QSize(80,60)); self.list.setMinimumWidth(220)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu); self.list.customContextMenuRequested.connect(self.on_ctx)
        left.addWidget(self.list, 1)
        row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh"); btn_refresh.setObjectName("muted"); btn_refresh.clicked.connect(self.populate)
        row.addWidget(btn_refresh)
        left.addLayout(row)
        main.addLayout(left,0)

        # RIGHT: camera + result
        right = QVBoxLayout()
        title = QLabel("Prediction – Capture & Compare"); title.setFont(QFont("Segoe UI", 16, QFont.Bold)); title.setAlignment(Qt.AlignCenter)
        right.addWidget(title)

        self.cam = QLabel(); self.cam.setFixedSize(480,360)
        self.cam.setStyleSheet(f"border:1px solid {BORDER}; border-radius:12px;")
        right.addWidget(self.cam, alignment=Qt.AlignCenter)

       
        self.card = QLabel("Result: —"); self.card.setProperty("class","card"); self.card.setObjectName("card"); self.card.setAlignment(Qt.AlignCenter)
        if FINAL_TO_TERMINAL_ONLY:
             self.card.hide()   # don’t show result on UI

        right.addWidget(self.card)

        row2 = QHBoxLayout()
        btn_cap = QPushButton("Capture & Predict"); btn_cap.setObjectName("primary"); btn_cap.clicked.connect(self.capture_predict)
        row2.addWidget(btn_cap)
        back = QPushButton("Back"); back.setObjectName("danger"); back.clicked.connect(go_back)
        row2.addWidget(back)
        right.addLayout(row2)

        main.addLayout(right,1)
        self.setLayout(main)

        self.cap = cv2.VideoCapture(0); self.timer = QTimer(); self.timer.timeout.connect(self._upd); self.timer.start(30)
        self.populate()

    def _upd(self):
        ok, f = self.cap.read()
        if ok:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB); h,w,ch = rgb.shape
            self.cam.setPixmap(QPixmap.fromImage(QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)))

    # ------- list mgmt -------
    def populate(self):
        self.list.clear()
        for x in sorted(self.refs.glob("*.xlsx"), key=lambda p: (len(p.stem), p.stem)):
            name = x.stem
            meta = load_meta(name)
            fail_txt = "" if meta.get("last_fail_pct") is None else f"  |  last: {meta['last_fail_pct']:.1f}% bad"
            txt = f"{name}{fail_txt}"
            icon = QIcon()
            thumb = self.refs / f"{name}.bmp"
            if thumb.exists(): 
                pix = QPixmap(str(thumb))
                if not pix.isNull(): icon = QIcon(pix)
            it = QListWidgetItem(icon, txt); it.setData(Qt.UserRole, str(x))
            self.list.addItem(it)

    def on_ctx(self, pos: QPoint):
        it = self.list.itemAt(pos)
        if not it: return
        ref_path = Path(it.data(Qt.UserRole))
        name = ref_path.stem
        menu = QMenu(self)
        act_rename = menu.addAction("Rename...")
        act_delete = menu.addAction("Delete")
        chosen = menu.exec_(self.list.mapToGlobal(pos))
        if chosen == act_rename:
            new_name, ok = QInputDialog.getText(self, "Rename GOOD", "New name:", text=name)
            if ok and new_name.strip():
                self.rename_good(name, new_name.strip())
        elif chosen == act_delete:
            self.delete_good(name)

    def rename_good(self, old, new):
        old_x = self.refs/f"{old}.xlsx"; old_b = self.refs/f"{old}.bmp"; old_j = self.refs/f"{old}.json"
        new_x = self.refs/f"{new}.xlsx"; new_b = self.refs/f"{new}.bmp"; new_j = self.refs/f"{new}.json"
        if new_x.exists():
            QMessageBox.warning(self,"Exists",f"{new}.xlsx already exists."); return
        try:
            if old_x.exists(): shutil.move(str(old_x), str(new_x))
            if old_b.exists(): shutil.move(str(old_b), str(new_b))
            if old_j.exists(): shutil.move(str(old_j), str(new_j))
            self.populate()
        except Exception as e:
            QMessageBox.critical(self,"Rename failed", str(e))

    def delete_good(self, name):
        try:
            for p in [self.refs/f"{name}.xlsx", self.refs/f"{name}.bmp", self.refs/f"{name}.json"]:
                if p.exists(): p.unlink()
            self.populate()
        except Exception as e:
            QMessageBox.critical(self,"Delete failed", str(e))

    # ------- predict -------
    def selected_ref(self):
        it = self.list.currentItem()
        return Path(it.data(Qt.UserRole)) if it else None
    def capture_predict(self):
        # ---- ensure a reference is selected ----
        sel = self.selected_ref()
        if not sel or not sel.exists():
            QMessageBox.warning(self, "Select GOOD", "Choose a GOOD on the left.")
            return


        # ---- capture a frame from camera ----
        ok, frame = self.cap.read()
        if not ok:
            QMessageBox.critical(self, "Camera", "Capture failed.")
            return

        # ---- run processing on captured frame ----
        (tpath, rpath, test_xlsx), (df_tst_thread, df_tst_ring) = process_image_to_excels(
            frame, Path(self.root) / "prediction", "test"
        )


        # ---- load reference features from the selected GOOD ----
        try:
            df_ref_thread = pd.read_excel(sel, sheet_name="Thread")
        except Exception:
            df_ref_thread = pd.DataFrame()

        try:
            df_ref_ring = pd.read_excel(sel, sheet_name="Ring")
        except Exception:
            df_ref_ring = pd.DataFrame()

        # ---- compare (define both vars no matter what) ----
        if not df_ref_thread.empty and not df_tst_thread.empty:
            cmp_thread = compare_df(df_ref_thread, df_tst_thread)
        else:
            cmp_thread = pd.DataFrame()

        if not df_ref_ring.empty and not df_tst_ring.empty:
            cmp_ring = compare_df(df_ref_ring, df_tst_ring)
        else:
            cmp_ring = pd.DataFrame()

        # ---- compute simple metrics safely ----
        def fail_pct(df):
            if df is None or len(df) == 0:
                return 0.0
            bad = (df["Quality"] == "Bad").sum() if "Quality" in df.columns else 0
            return float(bad) * 100.0 / float(len(df)) if len(df) else 0.0

        f_thread = fail_pct(cmp_thread)
        f_ring = fail_pct(cmp_ring)
        combined = (f_thread + f_ring) / 2.0

        # ---- verdict (works even if one/both are empty) ----
        v = verdict(cmp_thread, cmp_ring)

        # -------- FINAL RESULT (terminal-first) --------
        final_simple = "GOOD" if v == "Good" else "BAD"
        print(f"FINAL_RESULT:{final_simple}")
        print(f"FINAL_DETAIL:{v}")
        print(f"THREAD_BAD_PCT:{f_thread:.1f}")
        print(f"RING_BAD_PCT:{f_ring:.1f}")

        # Update UI only if you want on-screen text
        if not FINAL_TO_TERMINAL_ONLY:
            self.card.setText(
                f"Result: {v}\nPer-block bad rate → Thread: {f_thread:.1f}% | Ring: {f_ring:.1f}%"
            )

        # ---- save comparison workbook (always defined paths) ----
        out_x = Path(self.root) / "prediction" / f"comparison_with_{sel.stem}.xlsx"
        with pd.ExcelWriter(out_x, engine="openpyxl") as wr:
            # test sheets
            df_tst_thread.to_excel(wr, index=False, sheet_name="Thread_Test")
            df_tst_ring.to_excel(wr, index=False, sheet_name="Ring_Test")
            # compare sheets (can be empty)
            (cmp_thread if not cmp_thread.empty else pd.DataFrame()).to_excel(
                wr, index=False, sheet_name="Thread_Compare"
            )
            (cmp_ring if not cmp_ring.empty else pd.DataFrame()).to_excel(
                wr, index=False, sheet_name="Ring_Compare"
            )

        # ---- store meta & refresh list ----
        m = load_meta(sel.stem)
        m["last_fail_pct"] = combined
        save_meta(sel.stem, m)
        self.populate()


# ============================ Home (red-wine, no block color) ============================

from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class HomePage(QWidget):
    """
    Page-1 with:
      - White background
      - UV violet letters and buttons only
    """
    def __init__(self, open_train_cb, open_pred_cb, refresh_pred_list_cb=None):
        super().__init__()
        self.open_train_cb = open_train_cb
        self.open_pred_cb = open_pred_cb
        self.refresh_pred_list_cb = refresh_pred_list_cb

        # 🔹 White background only
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 80, 24, 40)
        root.setSpacing(24)

        # Title line 1
        title1 = QLabel("TEXA INNOVATES")
        title1.setAlignment(Qt.AlignCenter)
        title1.setFont(QFont("Segoe UI", 32, QFont.Bold))
        title1.setStyleSheet("color: #6a0dad;")   # UV violet text
        root.addWidget(title1)

        # Title line 2
        title2 = QLabel("Cone Inspection")
        title2.setAlignment(Qt.AlignCenter)
        t2font = QFont("Segoe UI", 20, QFont.Medium)
        title2.setFont(t2font)
        title2.setStyleSheet("color: #7b1fa2;")   # softer UV violet
        root.addWidget(title2)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(20)

        btn_train = QPushButton("Train Mode")
        btn_train.clicked.connect(self.open_train_cb)

        btn_pred = QPushButton("Prediction Mode")
        btn_pred.clicked.connect(self._go_prediction)

        # Buttons — UV outline with UV text
        btn_style = """
            QPushButton {
                background: transparent;
                border: 2px solid #6a0dad;   /* UV outline */
                border-radius: 12px;
                padding: 12px 28px;
                font-weight: 600;
                color: #6a0dad;              /* UV text */
            }
            QPushButton:hover {
                background: rgba(106,13,173,0.1); /* faint UV glow */
            }
        """
        btn_train.setStyleSheet(btn_style)
        btn_pred.setStyleSheet(btn_style)

        btn_row.addStretch(1)
        btn_row.addWidget(btn_train)
        btn_row.addWidget(btn_pred)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        root.addStretch(1)

    def _go_prediction(self):
        if self.refresh_pred_list_cb:
            try:
                self.refresh_pred_list_cb()
            except Exception:
                pass
        self.open_pred_cb()


# ============================ Main Window ============================

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Texa Innovates – Cone Inspection (Red-wine Industrial)")
        self.setGeometry(200, 120, 960, 640)
        self.stack = QStackedWidget(); self.setCentralWidget(self.stack)

        self.home = HomePage(self.open_train, self.open_pred, self.refresh_pred_list)
        self.stack.addWidget(self.home)

    def open_train(self):
        self.tp = TrainPage(self.refresh_pred_list, self.go_home)
        self.stack.addWidget(self.tp); self.stack.setCurrentWidget(self.tp)

    def open_pred(self):
        self.pp = PredictionPage(self.go_home)
        self.stack.addWidget(self.pp); self.stack.setCurrentWidget(self.pp)

    def refresh_pred_list(self):
        if hasattr(self, "pp"):
            self.pp.populate()

    def go_home(self):
        self.stack.setCurrentIndex(0)

# ============================ Run ============================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Main(); w.show()
    sys.exit(app.exec_())

