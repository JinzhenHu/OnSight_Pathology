import mss
import numpy as np
import cv2
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

from utils import extract_tiles


def fix_region(region, tile_size):
    reg = region.copy()
    reg['width'] = max(reg['width'], tile_size)
    reg['height'] = max(reg['height'], tile_size)
    return reg


def visualize_ki67_overlay(img, masks, labels, ki67_positive, alpha=0.4):
    """
    Color each nucleus red (positive) or green (negative).
    """
    # Start from a copy of the original
    overlay = img.copy()

    # Make sure we work in BGR for drawing
    # (Assuming img is BGR from OpenCV)
    for lab, is_pos in zip(labels, ki67_positive):
        region = (masks == lab)

        if is_pos:
            color = (0, 0, 255)  # red in BGR
        else:
            color = (255, 0, 0)  # blue in BGR

        overlay[region] = color

    blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return blended


def compute_hed_and_dab(img):
    """
    img: HxWx3, BGR (OpenCV) or RGB (if you already converted).
    Returns:
        img_rgb: HxWx3 RGB image
        hed:     HxWx3 HED image
        dab_od:  HxW, positive OD-like measure where higher = more DAB
    """
    # Assume BGR input from OpenCV; convert to RGB for skimage
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # RGB -> HED
    hed = rgb2hed(img)

    # DAB is channel 2 (D). In HED, more stain ≈ more negative.
    # Invert sign so "more brown" => larger positive value
    # dab_od = -hed[..., 2]
    dab_od = hed[..., 2]

    return img, hed, dab_od


# ---------- GMM helper: intersection of two 1D Gaussians ----------

def gmm_two_component_threshold(gmm):
    """
    Given a fitted 2-component 1D GaussianMixture, find the intersection point
    of the two weighted Gaussians as the threshold.

    If no sign change is found, use the point where |p1 - p2| is minimal,
    clamped to [m1, m2].
    """
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    # sort components by mean (c0 = low, c1 = high)
    order = np.argsort(means)
    m1, m2 = means[order]
    v1, v2 = variances[order]
    w1, w2 = weights[order]

    s1, s2 = np.sqrt(v1), np.sqrt(v2)

    # numeric search for intersection in a reasonable range
    x_min = m1 - 3 * s1
    x_max = m2 + 3 * s2
    xs = np.linspace(x_min, x_max, 512)

    def pdf(x, m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    p1 = w1 * pdf(xs, m1, s1)
    p2 = w2 * pdf(xs, m2, s2)
    diff = p1 - p2

    # look for sign change between p1 and p2
    sign_change = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0]

    if len(sign_change) > 0:
        i = sign_change[0]
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = diff[i], diff[i + 1]
        thr = x0 - y0 * (x1 - x0) / (y1 - y0)
        return float(thr)

    # fallback: point of minimal |p1 - p2|
    i_min = np.argmin(np.abs(diff))
    thr = xs[i_min]

    # clamp to [m1, m2]
    thr = float(np.clip(thr, m1, m2))
    return thr


# ---------- Main classifier ----------

def classify_ki67_gmm_hybrid(
        img,
        masks,
        min_area=20,
        gmm_random_state=0,
        extreme_pct=1.0,  # use midpoint if intersection is outside [p1, p99]
):
    """
    Ki-67 classifier that:

      1. ALWAYS tries 2-component GMM on per-nucleus DAB means.
         - Primary threshold = intersection of weighted Gaussians
         - If intersection is "extreme" (e.g., < p1 or > p99), fall back to midpoint.
      2. If GMM fails altogether, uses Otsu.
      3. If Otsu also fails, uses homogeneous fallback.

    Returns:
        nucleus_labels, dab_means, ki67_positive, dab_thresh, method_used
        method_used in {"gmm-intersection", "gmm-midpoint", "otsu", "homogeneous"}
    """
    _, _, dab_od = compute_hed_and_dab(img)

    labels = np.unique(masks)
    labels = labels[labels != 0]

    nucleus_labels = []
    dab_means = []

    for lab in labels:
        region = (masks == lab)
        if region.sum() < min_area:
            continue
        dab_means.append(float(dab_od[region].mean()))
        nucleus_labels.append(lab)

    if not nucleus_labels:
        return np.array([]), np.array([]), np.array([]), None, "none"

    nucleus_labels = np.array(nucleus_labels)
    dab_means = np.array(dab_means, dtype=float)

    tiny = 1e-8

    # ------------------------------
    # 1) TRY GMM ALWAYS
    # ------------------------------
    try:
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=gmm_random_state,
        )
        gmm.fit(dab_means.reshape(-1, 1))

        means = gmm.means_.flatten()
        order = np.argsort(means)
        m1, m2 = means[order]
        mid = 0.5 * (m1 + m2)

        # primary candidate: intersection threshold
        thr_int = gmm_two_component_threshold(gmm)

        # sanity: if intersection is too close to extremes, use midpoint instead
        p_low, p_high = np.percentile(dab_means, [extreme_pct, 100 - extreme_pct])

        if thr_int < p_low or thr_int > p_high:
            # suspicious intersection -> midpoint fallback
            dab_thresh = float(mid)
            method_used = "gmm-midpoint"
        else:
            dab_thresh = float(thr_int)
            method_used = "gmm-intersection"

        ki67_positive = dab_means > dab_thresh
        return nucleus_labels, dab_means, ki67_positive, dab_thresh, method_used

    except Exception:
        # GMM failed (rare): fall through to Otsu + homogeneous
        pass

    # ------------------------------
    # 2) Otsu fallback
    # ------------------------------
    try:
        dab_thresh = float(threshold_otsu(dab_means))
        ki67_positive = dab_means > dab_thresh
        return nucleus_labels, dab_means, ki67_positive, dab_thresh, "otsu"
    except Exception:
        pass

    # ------------------------------
    # 3) Homogeneous fallback
    # ------------------------------
    med = float(np.median(dab_means))
    dab_thresh = med + tiny
    ki67_positive = dab_means > dab_thresh
    return nucleus_labels, dab_means, ki67_positive, dab_thresh, "homogeneous"


def process_region(region, **kwargs):
    metadata = kwargs['metadata']
    model = kwargs['model']

    ###

    tile_size = metadata['tile_size']

    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)

    frame_orig = np.array(screenshot, dtype=np.uint8)
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGRA2RGB)  # cellpose-sam takes in rgb!
    frame = frame[:max((frame.shape[0] // tile_size) * tile_size, tile_size),
            :max((frame.shape[1] // tile_size) * tile_size, tile_size), :]

    slices = extract_tiles(frame, tile_size)

    # Detect and render
    masks, _, _ = model.eval(slices)

    try:
        _mask_alpha = 1 - float(kwargs['additional_configs'].get('mask_transparency', 0.5))
    except:
        _mask_alpha = 0.5

    tile_size_y = tile_size_x = tile_size
    if frame.shape[0] < tile_size:
        tile_size_y = frame.shape[0]
    if frame.shape[1] < tile_size:
        tile_size_x = frame.shape[1]

    num_pos = 0
    num_pos_neg = 0
    seg_mask = np.zeros(frame.shape)
    k = 0
    for i in range(frame.shape[0] // tile_size_y):
        for j in range(frame.shape[1] // tile_size_x):

            if masks[k].max() > 0:

                labels, dab_means, ki67_pos, dab_thresh, method = classify_ki67_gmm_hybrid(
                    slices[k],
                    masks[k],
                    min_area=20,
                )
                # print("Threshold:", dab_thresh, "Method:", method)

                num_pos += ki67_pos.sum()
                num_pos_neg += len(ki67_pos)

                # Opencv stuff in BGR
                overlayed_img = visualize_ki67_overlay(cv2.cvtColor(slices[k], cv2.COLOR_RGB2BGR), masks[k], labels,
                                                       ki67_pos, alpha=_mask_alpha)

                seg_mask[
                (i * tile_size):((i * tile_size) + tile_size),
                (j * tile_size):((j * tile_size) + tile_size),
                :
                ] = overlayed_img

            else:
                # no masks found
                seg_mask[
                (i * tile_size):((i * tile_size) + tile_size),
                (j * tile_size):((j * tile_size) + tile_size),
                :
                ] = frame[
                    (i * tile_size):((i * tile_size) + tile_size),
                    (j * tile_size):((j * tile_size) + tile_size),
                    :
                    ]

            k += 1

    text = '(+) {:.2f} %\n'.format(num_pos / num_pos_neg * 100 if num_pos_neg > 0 else 0)
    text += '(+) cells: {}\n'.format(num_pos)
    text += '(-) cells: {}\n'.format(num_pos_neg - num_pos)
    metrics = {
        "mib_pos": int(num_pos),
        "mib_total": int(num_pos_neg),
        "area_px": frame.shape[0] * frame.shape[1],
        "mpp": metadata.get("mpp", 0.25),
        "orig_img": cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    }
    return seg_mask.astype(np.uint8), text, metrics
