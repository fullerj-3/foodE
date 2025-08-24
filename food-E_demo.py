import streamlit as st
import cv2
import zxingcpp
import numpy as np
from PIL import Image
import requests
import platform

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Food Scanner + 1‑Photo Mass Estimate", layout="wide")
st.title("🍎 Food Scanner & Scale (1 Photo)")
st.caption("Scan a product barcode OR estimate portion mass from a single photo. Both show confidence.")

# --- Optional CSS to style Calories font (affects only our custom 'kcal-*' blocks)
st.markdown(
    """
    <style>
    :root { --kcal-font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; }
    .kcal-label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 2px; }
    .kcal-val { font-family: var(--kcal-font); font-size: 28px; font-weight: 800; line-height: 1.1; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Simple debug helper: prints to terminal and sidebar (best‑effort)
def _dbg(msg: str):
    try:
        print(msg)
    except Exception:
        pass
    try:
        st.sidebar.write(str(msg))
    except Exception:
        pass

_dbg("[BOOT] App starting…")
_dbg(f"[ENV] Python {platform.python_version()} | OpenCV {cv2.__version__} | NumPy {np.__version__}")
_dbg(f"[ENV] zxingcpp present: {zxingcpp is not None}")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _dbg(msg: str):
    """Lightweight debug to console + sidebar."""
    try:
        st.sidebar.text(msg)
    except Exception:
        pass
    try:
        print(msg)
    except Exception:
        pass

def decode_barcodes(img: np.ndarray):
    """Return a list of ZXing results from an RGB image array."""
    _dbg(f"[decode_barcodes] input shape={getattr(img, 'shape', None)} dtype={getattr(img, 'dtype', None)}")
    try:
        results = zxingcpp.read_barcodes(img)
        _dbg(f"[decode_barcodes] multi read → {len(results) if results else 0} results")
    except Exception as e:
        _dbg(f"[decode_barcodes] multi read failed: {e}. Falling back to single read.")
        res = zxingcpp.read_barcode(img)
        results = [res] if res else []
    if results:
        try:
            preview = [getattr(r, 'text', '') for r in results]
            _dbg(f"[decode_barcodes] texts={preview}")
        except Exception:
            pass
    return [r for r in results if r]


# ---- Image selection helpers (prefer curated 'front of pack') -------------
# OFF exposes curated images under selected_images.{front,ingredients,nutrition}.display[lang]
# We'll prefer those over raw image_url/image_front_url.
def _pick_from_selected_images(p: dict, panel: str = "front", kind: str = "display", pref_langs=None):
    pref_langs = pref_langs or []
    si = p.get("selected_images") or {}
    panel_dict = si.get(panel) or {}
    d = panel_dict.get(kind)
    if isinstance(d, dict):
        for lg in pref_langs:
            url = d.get(lg)
            if url:
                return url
        for url in d.values():
            if url:
                return url
    return None

def _best_product_image_urls(p: dict):
    lang = (p.get("lang") or "en").split(",")[0]
    pref = [lang, "en", "fr"]
    front  = _pick_from_selected_images(p, panel="front",       kind="display", pref_langs=pref)  or p.get("image_front_url")     or p.get("image_url")
    ingred = _pick_from_selected_images(p, panel="ingredients", kind="display", pref_langs=pref)  or p.get("image_ingredients_url")
    nutr   = _pick_from_selected_images(p, panel="nutrition",   kind="display", pref_langs=pref)  or p.get("image_nutrition_url")
    return front, ingred, nutr


def fetch_openfoodfacts(barcode: str):
    """Fetch product + nutriments from Open Food Facts by UPC/EAN."""
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    _dbg(f"[OFF] GET {url}")
    try:
        resp = requests.get(url, timeout=10)
        _dbg(f"[OFF] status_code={getattr(resp, 'status_code', None)}")
        data = resp.json()
        _dbg(f"[OFF] json status={data.get('status', None)} product_name={data.get('product', {}).get('product_name', None)}")
    except Exception as e:
        _dbg(f"[OFF] request failed: {e}")
        return None
    if data.get("status") == 1:
        p = data["product"]
        # Pass through Nutri-Score fields so the badge can render
        ns_grade = p.get("nutriscore_grade") or p.get("nutrition_grade_fr")
        ns_score = p.get("nutriscore_score")
        if ns_score is None:
            ns_score = (p.get("nutriscore_data") or {}).get("score")
        # choose best curated images
        front, ingred_img, nutr_img = _best_product_image_urls(p)
        _dbg(f"[OFF] nutriscore_grade={ns_grade} nutriscore_score={ns_score}")
        _dbg(f"[OFF] images front={bool(front)} ingred={bool(ingred_img)} nutr={bool(nutr_img)}")
        return {
            "name": p.get("product_name") or p.get("generic_name") or "Unknown product",
            "brand": p.get("brands"),
            "image_url": front,
            "image_front_best": front,
            "image_ingredients_best": ingred_img,
            "image_nutrition_best": nutr_img,
            "nutriments": p.get("nutriments", {}),
            "serving_size": p.get("serving_size"),
            "nutriscore_grade": ns_grade,
            "nutrition_grade_fr": p.get("nutrition_grade_fr"),
            "nutriscore_score": ns_score,
        }
    _dbg("[OFF] product not found")
    return None


def grade_food(n: dict):
    """Simple example grading. Customize per your rubric."""
    protein = float(n.get("proteins_100g") or 0)
    sugar = float(n.get("sugars_100g") or 0)
    satfat = float(n.get("saturated-fat_100g") or 0)
    fiber = float(n.get("fiber_100g") or 0)

    score = 0
    if protein >= 10: score += 2
    if fiber >= 5: score += 1
    if sugar >= 10: score -= 2
    if satfat >= 5: score -= 1

    if score >= 2:
        label = "✅ Healthy"
    elif score >= 0:
        label = "⚖️ Moderate"
    else:
        label = "❌ Unhealthy"

    return label

# ---- Nutrition & additives rendering -------------------------------------

def _kcal_from(n: dict, suffix: str):
    # Prefer energy-kcal_*, else convert energy_* (kJ)→kcal
    kcal = n.get(f"energy-kcal_{suffix}")
    if kcal is None:
        kj = n.get(f"energy_{suffix}")
        if kj is not None:
            kcal = float(kj) / 4.184
    return kcal

def _num(v):
    try:
        return None if v is None else float(v)
    except Exception:
        return None

def _pretty_additives(tags):
    out = []
    for t in (tags or []):
        # tags like "en:e330" or "en:monosodium-glutamate"
        name = t.split(":", 1)[-1]
        if name.startswith("e") and name[1:].isdigit():
            out.append(name.upper())
        else:
            out.append(name.replace("-", " ").title())
    # de-dup while preserving order
    seen = set(); uniq = []
    for a in out:
        if a not in seen:
            uniq.append(a); seen.add(a)
    return uniq

# ---- Nutri-Score rendering ------------------------------------------------

def render_nutriscore(product: dict):
    grade = (product.get("nutriscore_grade") or product.get("nutrition_grade_fr") or "").strip().upper()
    score = product.get("nutriscore_score")
    if not grade:
        return
    color_map = {"A":"#2ecc71","B":"#8bd05b","C":"#f1c40f","D":"#e67e22","E":"#e74c3c"}
    color = color_map.get(grade, "#9ca3af")
    score_html = f"<span style='font-size:12px;color:#6b7280'>(score {int(score)})</span>" if isinstance(score, (int,float)) else ""
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:.5rem;margin:8px 0 10px">
            <div style="font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em">Nutri‑Score</div>
            <span style="display:inline-block;padding:6px 10px;border-radius:8px;color:#fff;font-weight:800;font-size:18px;min-width:36px;text-align:center;background:%s">%s</span>
            %s
        </div>
        """ % (color, grade, score_html),
        unsafe_allow_html=True,
    )


def render_nutrition_and_additives(product: dict):
    # Nutri-Score (if available)
    render_nutriscore(product)

    n = product.get("nutriments", {})
    serving = product.get("serving_size")

    kcal_100 = _kcal_from(n, "100g")
    kcal_srv = _kcal_from(n, "serving")

    fat_100 = _num(n.get("fat_100g"))
    carb_100 = _num(n.get("carbohydrates_100g"))
    sugar_100 = _num(n.get("sugars_100g"))
    prot_100 = _num(n.get("proteins_100g"))

    fat_srv = _num(n.get("fat_serving"))
    carb_srv = _num(n.get("carbohydrates_serving"))
    sugar_srv = _num(n.get("sugars_serving"))
    prot_srv = _num(n.get("proteins_serving"))

    st.markdown("#### Nutrition")
    if any(v is not None for v in [kcal_srv, fat_srv, carb_srv, sugar_srv, prot_srv]):
        st.caption(f"Per serving{f' ({serving})' if serving else ''}")
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1:
            if kcal_srv is not None:
                st.markdown(f"<div class='kcal-label'>Calories</div><div class='kcal-val'>{kcal_srv:.0f} kcal</div>", unsafe_allow_html=True)
            else:
                st.metric("Calories", "—")
        with c2: st.metric("Fat", f"{fat_srv:.1f} g" if fat_srv is not None else "—")
        with c3: st.metric("Carbs", f"{carb_srv:.1f} g" if carb_srv is not None else "—")
        with c4: st.metric("Sugars", f"{sugar_srv:.1f} g" if sugar_srv is not None else "—")
        with c5: st.metric("Protein", f"{prot_srv:.1f} g" if prot_srv is not None else "—")

    # st.caption("Per 100 g/ml")
    # c1,c2,c3,c4,c5 = st.columns(5)
    # with c1:
    #     if kcal_100 is not None:
    #         st.markdown(f"<div class='kcal-label'>Calories</div><div class='kcal-val'>{kcal_100:.0f} kcal</div>", unsafe_allow_html=True)
    #     else:
    #         st.metric("Calories", "—")
    # with c2: st.metric("Fat", f"{fat_100:.1f} g" if fat_100 is not None else "—")
    # with c3: st.metric("Carbs", f"{carb_100:.1f} g" if carb_100 is not None else "—")
    # with c4: st.metric("Sugars", f"{sugar_100:.1f} g" if sugar_100 is not None else "—")
    # with c5: st.metric("Protein", f"{prot_100:.1f} g" if prot_100 is not None else "—")

    # Additives
    additives = product.get("additives_tags") or product.get("additives_old_tags") or []
    additives_n = product.get("additives_n")
    pretty = _pretty_additives(additives)
    st.markdown("#### Additives")
    if additives_n is not None:
        st.caption(f"Additives count: {additives_n}")
    if pretty:
        st.write(", ".join(pretty))
    else:
        st.write("No additives listed or not provided.")

# ---------------------------------------------------------------------------
# 1‑Photo Mass Estimation (no marker). Uses a plate prior + Monte Carlo.
# ---------------------------------------------------------------------------
# Assumptions/prior distributions used in the MC sampler:
# - Plate diameter: dinner plate ~ 270 mm with SD 20 mm
# - Thickness (height) of food: mean 25 mm, SD 10 mm, truncated > 5 mm
# - Shape factor (pile vs slab): mean 0.65, SD 0.15, truncated [0.4, 1.0]
# - Density: mean 1.00 g/mL, SD 0.20, truncated > 0.5
# These are intentionally wide; you can tighten per category later.

PLATE_DIAMETER_MEAN_MM = 270.0
PLATE_DIAMETER_SD_MM   = 20.0
THICKNESS_MEAN_MM      = 25.0
THICKNESS_SD_MM        = 10.0
SHAPE_MEAN             = 0.65
SHAPE_SD               = 0.15
DENSITY_MEAN           = 1.00
DENSITY_SD             = 0.20

# HSV thresholds for removing the (typically white) plate/background
HSV_SAT_PLATE_MAX = 60      # plate tends to have low saturation
HSV_VAL_PLATE_MIN = 140     # plate tends to be bright


def detect_plate_circle(gray: np.ndarray):
    """Detect the largest circular plate via HoughCircles. Return (x,y,r) in px or None."""
    h, w = gray.shape[:2]
    _dbg(f"[plate] gray shape={gray.shape}")
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.1, minDist=h//6,
        param1=120, param2=30, minRadius=max(40, min(h, w)//10), maxRadius=min(h, w)//2
    )
    if circles is None:
        _dbg("[plate] no circles detected")
        return None
    circles = np.uint16(np.around(circles))
    _dbg(f"[plate] circles detected: {circles.shape[1] if len(circles.shape)>1 else 'unknown'}")
    # choose the biggest
    x, y, r = max(circles[0, :], key=lambda c: c[2])
    _dbg(f"[plate] chosen circle x={x} y={y} r={r}")
    return int(x), int(y), int(r)


def segment_food_on_plate(bgr: np.ndarray, plate_xy_r):
    """Segment non-plate pixels inside the detected plate.
    Returns binary mask (uint8 0/255) where 255 ~ food.
    """
    x, y, r = plate_xy_r
    h, w = bgr.shape[:2]
    _dbg(f"[segment] img shape={bgr.shape} plate=(x={x},y={y},r={r})")

    # mask for the circular ROI (plate area)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, (x, y), r-5, 255, thickness=-1)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # heuristic: plate is bright & low saturation
    plate_like = (S < HSV_SAT_PLATE_MAX) & (V > HSV_VAL_PLATE_MIN)
    plate_like = (plate_like.astype(np.uint8) * 255)
    _dbg(f"[segment] plate_like count={int(np.count_nonzero(plate_like))}")

    # Food candidate = inside plate AND NOT plate-like
    food_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(plate_like))

    # clean up
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=1)

    # keep largest blob to avoid rim/noise
    cnts, _ = cv2.findContours(food_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _dbg(f"[segment] contours found={len(cnts) if cnts is not None else 0}")
    if not cnts:
        _dbg("[segment] no contours after cleanup")
        return None
    mask = np.zeros_like(food_mask)
    big = max(cnts, key=cv2.contourArea)
    area_big = int(cv2.contourArea(big))
    cv2.drawContours(mask, [big], -1, 255, thickness=cv2.FILLED)
    _dbg(f"[segment] largest area px={area_big}")
    return mask


def monte_carlo_mass_from_one_photo(area_px: float, plate_radius_px: float, n=3000):
    """Monte Carlo mass estimate using plate prior (one photo). Returns (mean_g, lo_g, hi_g, conf25)."""
    _dbg(f"[mc] area_px={area_px} plate_radius_px={plate_radius_px} n={n}")
    # Sample priors
    plate_diam_mm = np.random.normal(PLATE_DIAMETER_MEAN_MM, PLATE_DIAMETER_SD_MM, n)
    plate_diam_mm = np.clip(plate_diam_mm, 200, 320)  # truncate to a sane range
    mm_per_px = plate_diam_mm / (2.0 * plate_radius_px)

    thickness_mm = np.random.normal(THICKNESS_MEAN_MM, THICKNESS_SD_MM, n)
    thickness_mm = np.clip(thickness_mm, 5.0, None)

    shape = np.random.normal(SHAPE_MEAN, SHAPE_SD, n)
    shape = np.clip(shape, 0.4, 1.0)

    density = np.random.normal(DENSITY_MEAN, DENSITY_SD, n)
    density = np.clip(density, 0.5, None)

    # area in mm^2, volume in mm^3 then mL
    area_mm2 = area_px * (mm_per_px ** 2)
    vol_ml = (area_mm2 * thickness_mm * shape) / 1000.0
    mass_g = vol_ml * density

    mass_g = mass_g[mass_g > 0]
    mean = float(np.mean(mass_g))
    lo, hi = np.percentile(mass_g, [2.5, 97.5])
    conf = float(np.mean((mass_g >= 0.75*mean) & (mass_g <= 1.25*mean))) * 100.0
    _dbg(f"[mc] mean={mean:.2f}g lo={lo:.2f}g hi={hi:.2f}g conf±25%={conf:.1f}%")
    return mean, lo, hi, conf

# ---------------------------------------------------------------------------
# UI: Tabs for (A) Barcode scan and (B) 1‑Photo mass estimate
# ---------------------------------------------------------------------------

scan_tab, mass_tab = st.tabs(["📦 Scan Barcode", "⚖️ 1‑Photo Mass Estimate"]) 

with scan_tab:
    st.subheader("Scan a product barcode")
    compare_two = st.toggle("Compare two items", value=False, help="Scan two barcodes side-by-side and view both results.")
    _dbg(f"[scan] compare_two={compare_two}")

    if not compare_two:
        col1, col2 = st.columns(2, vertical_alignment="center")
        with col1:
            camera_image = st.camera_input("Use your camera", help="Center the barcode and snap a photo.")
            _dbg(f"[scan] camera_image present={camera_image is not None}")
        with col2:
            uploaded_file = st.file_uploader("...or upload a barcode image", type=["png", "jpg", "jpeg"])
            _dbg(f"[scan] uploaded_file present={uploaded_file is not None}")
        img_source = camera_image or uploaded_file

        if img_source:
            image = Image.open(img_source).convert("RGB")
            _dbg(f"[scan] image mode={image.mode} size={image.size}")
            img_array = np.array(image)
            results = decode_barcodes(img_array)

            if results:
                st.success("Barcode detected ✅")
                _dbg(f"[scan] {len(results)} barcode(s) detected")
                for i, r in enumerate(results, start=1):
                    symb = getattr(r, "format", None) or getattr(r, "symbology", "")
                    code_text = getattr(r, "text", "")
                    # st.write(f"**{i}. {symb}** → `{code_text}`")
                    _dbg(f"[scan] result#{i} symb={symb} text={code_text}")

                first_code = getattr(results[0], "text", None)
                _dbg(f"[scan] first_code={first_code}")
                if first_code:
                    with st.spinner("Looking up product in Open Food Facts..."):
                        product = fetch_openfoodfacts(first_code)
                    if product:
                        st.subheader(product["name"])
                        meta = []
                        if product.get("brand"):
                            meta.append(product["brand"])
                        if product.get("serving_size"):
                            meta.append(f"Serving: {product['serving_size']}")
                        if meta:
                            st.caption(" • ".join(meta))
                        best_img = product.get("image_front_best") or product.get("image_url")
                        if best_img:
                            st.image(best_img, width=220)

                        label = grade_food(product.get("nutriments", {}))
                        st.markdown(f"### Grade: {label}")
                        render_nutrition_and_additives(product)
                        _dbg("[scan] rendered nutrition & additives")
                        # _dbg(f"[scan] grade={label} details_keys={list(details.keys())}")
                    else:
                        _dbg("[scan] OFF lookup returned None")
                        st.info("Product not found in Open Food Facts. Consider a fallback API (USDA/Nutritionix).")
            else:
                _dbg("[scan] no barcode detected")
                st.error("Couldn't detect a barcode. Try a sharper, closer photo with good lighting.")
    else:
        st.caption("Scan **Item A** and **Item B** (one photo each). You can use camera or upload for each.")
        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Item A**")
            camA = st.camera_input("Use your camera (A)", key="camA")
            upA  = st.file_uploader("...or upload (A)", type=["png","jpg","jpeg"], key="upA")
            srcA = camA or upA
        with colB:
            st.markdown("**Item B**")
            camB = st.camera_input("Use your camera (B)", key="camB")
            upB  = st.file_uploader("...or upload (B)", type=["png","jpg","jpeg"], key="upB")
            srcB = camB or upB

        def _process_compare(src, tag):
            if not src:
                _dbg(f"[compare] {tag}: no source")
                return None
            image = Image.open(src).convert("RGB")
            _dbg(f"[compare] {tag}: image size={image.size}")
            arr = np.array(image)
            res = decode_barcodes(arr)
            if not res:
                st.error(f"{tag}: No barcode detected.")
                _dbg(f"[compare] {tag}: decode failed")
                return None
            code = getattr(res[0], "text", None)
            symb = getattr(res[0], "format", None) or getattr(res[0], "symbology", "")
            # st.write(f"{tag}: **{symb}** → `{code}`")
            _dbg(f"[compare] {tag}: code={code}")
            if not code:
                return None
            with st.spinner(f"{tag}: Looking up product..."):
                prod = fetch_openfoodfacts(code)
            if prod:
                st.write(f"{tag}: **{prod['name']}**")
                best_img = prod.get("image_front_best") or prod.get("image_url")
                if best_img:
                    st.image(best_img, width=180)
                label = grade_food(prod.get("nutriments", {}))
                st.caption(f"{tag}: Grade {label}")
                # st.json(details)
                render_nutrition_and_additives(prod)
                return {"name": prod["name"], "label": label}
            else:
                st.info(f"{tag}: Not found in Open Food Facts.")
                return None

        with colA:
            infoA = _process_compare(srcA, "Item A")
        with colB:
            infoB = _process_compare(srcB, "Item B")

        _dbg(f"[compare] A present={infoA is not None} B present={infoB is not None}")

with mass_tab:
    st.subheader("Estimate portion mass from a single photo")
    st.caption("Put the food on a plate and include the **entire plate** in the shot. No extra tools.")

    photo = st.camera_input("Take one photo of the food on a plate")
    _dbg(f"[mass] photo present={photo is not None}")

    if photo:
        bgr = cv2.cvtColor(np.array(Image.open(photo).convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _dbg(f"[mass] bgr shape={bgr.shape} gray shape={gray.shape}")

        plate = detect_plate_circle(gray)
        if plate is None:
            _dbg("[mass] plate detection failed")
            st.error("Couldn't detect a plate in the image. Please include the full plate in view.")
        else:
            x, y, r = plate
            _dbg(f"[mass] plate=(x={x}, y={y}, r={r})")
            # visualize detection
            vis = bgr.copy()
            cv2.circle(vis, (x, y), r, (0, 255, 0), 3)

            mask = segment_food_on_plate(bgr, plate)
            if mask is None or np.count_nonzero(mask) < 200:
                _dbg(f"[mass] segmentation failed or too small. mask None? {mask is None} area={0 if mask is None else int(np.count_nonzero(mask))}")
                st.error("Food segmentation failed. Try better lighting/contrast and avoid busy patterns.")
            else:
                area_px = int(np.count_nonzero(mask))
                _dbg(f"[mass] area_px={area_px}")
                mean_g, lo_g, hi_g, conf = monte_carlo_mass_from_one_photo(area_px, r)

                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Detected plate", width=380)
                st.image(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), caption="Food mask", width=380)

                st.markdown(f"### Estimated weight: **{mean_g:.0f} g**")
                st.caption(f"95% CI: {lo_g:.0f}–{hi_g:.0f} g  •  Confidence (±25% band): {conf:.0f}%")
                _dbg(f"[mass] output grams={mean_g:.2f} CI=({lo_g:.2f},{hi_g:.2f}) conf={conf:.1f}%")

    with st.expander("How this works & caveats"):
        st.markdown(
            """
            **Single photo cannot recover true metric scale** without a reference. To stay one‑photo/zero‑extra‑tools, this estimator:
            - Detects a circular **plate** via HoughCircles.
            - Assumes a typical dinner plate diameter (\~27±2 cm) to convert pixels→mm.
            - Segments non‑plate pixels as food and estimates area.
            - Uses broad priors for thickness, shape, and density to compute mass.
            - Runs a Monte Carlo simulation to show a **95% CI** and a **confidence** that the true mass is within **±25%** of the estimate.

            Accuracy improves if your product category is known (to tighten density/shape priors). You can later auto‑select priors after scanning the barcode.
            """
        )

# End of app
