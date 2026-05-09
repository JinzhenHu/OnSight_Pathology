import mss
import numpy as np
import cv2
from skimage.color import rgb2lab
import histomicstk as htk
from numpy.linalg import LinAlgError
from histomicstk.features import compute_nuclei_features
from skimage.segmentation import clear_border
from utils import extract_tiles
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def _safe_float(value, default=0.1):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def fix_region(region, tile_size):
    reg = region.copy()
    reg['width'] = max(reg['width'], tile_size)
    reg['height'] = max(reg['height'], tile_size)
    return reg


def visualize_ki67_refined_overlay(img, masks, labels, is_kept_arr, is_pos_arr, alpha=0.4):

    overlay = img.copy()
    max_label = masks.max()
    
    color_lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
    paint_mask = np.zeros(max_label + 1, dtype=bool)
    
    for lab, is_kept, is_pos in zip(labels, is_kept_arr, is_pos_arr):
        if is_kept:
            if is_pos:
                color_lut[lab] = [0, 0, 255]  
            else:
                color_lut[lab] = [255, 0, 0]  
        else:
            color_lut[lab] = [255, 255, 0]
            
        paint_mask[lab] = True
            
    pixel_colors = color_lut[masks]
    pixel_paint = paint_mask[masks]
    
    overlay[pixel_paint] = pixel_colors[pixel_paint]
    blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    bg_mask = (masks == 0)
    blended[bg_mask] = img[bg_mask]
    
    return blended

def deconvolution_ihc_ki67_dynamic(img_rgb):
    I_0 = 255
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    try:
        W_estimated = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(img_rgb, I_0)
        h_index = htk.preprocessing.color_deconvolution.find_stain_index(
            stain_color_map['hematoxylin'], W_estimated
        )
        e_index = htk.preprocessing.color_deconvolution.find_stain_index(
            stain_color_map['dab'], W_estimated
        )
        W_custom = np.zeros((3, 3))
        W_custom[:, 0] = W_estimated[:, h_index]
        W_custom[:, 1] = W_estimated[:, e_index]

        W_custom = htk.preprocessing.color_deconvolution.complement_stain_matrix(W_custom)
        deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(
            img_rgb, W_custom, I_0
        )
    except (LinAlgError, IndexError, ValueError):
        #print("SVD did not converge. Using default stain matrix.")
        stains = ['hematoxylin', 'dab', 'null']
        W_custom = np.array([stain_color_map[st] for st in stains]).T
        deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(
            img_rgb, W_custom, I_0
        )

    im_nuclei = deconv_result.Stains[:, :, 0]
    im_cytoplasm = deconv_result.Stains[:, :, 1]
    return im_nuclei, im_cytoplasm


def classify_ki67_labb_approach(img, masks, min_area=10, threshold=0):
    def compute_lab_b_signal(img_rgb, masks, labels, min_area=10):
        lab = rgb2lab(img_rgb)
        b_channel = lab[..., 2]  # b*: brown = positive, blue = negative

        scores = []
        valid_labels = []
        for lab_id in labels:
            region = masks == lab_id
            if region.sum() < min_area:
                continue
            if lab[..., 0][region].mean() < 40:
                scores.append(999)
            else:
                scores.append(float(b_channel[region].mean()))
            valid_labels.append(lab_id)

        return np.array(valid_labels), np.array(scores)

    labels = np.unique(masks)
    labels = labels[labels != 0]

    valid_labels, scores = compute_lab_b_signal(img, masks, labels, min_area)

    return valid_labels, scores > threshold


def um2_to_px(area_um2, mpp):
        return area_um2 / (mpp ** 2)

def process_region(region, **kwargs):
    metadata = kwargs['metadata']
    model = kwargs['model']
    configs = kwargs.get('additional_configs', {})

    tile_size = metadata.get('tile_size', 256)
    mpp = _safe_float(configs.get('mpp', metadata.get('mpp', 0.25)))

    metrics = {
        "mib_pos": 0, "mib_total": 0,
        "area_px": 0, "mpp": mpp,
        "orig_img": None
    }

    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)

    frame_orig = np.array(screenshot, dtype=np.uint8)
    frame_rgb = cv2.cvtColor(frame_orig, cv2.COLOR_BGRA2RGB)
    metrics["orig_img"] = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    metrics["area_px"] = frame_rgb.shape[0] * frame_rgb.shape[1]
    
    masks, _, _ = model.eval(frame_rgb)
    im_nuclei, im_cytoplasm = deconvolution_ihc_ki67_dynamic(frame_rgb)

    masks = clear_border(masks)
    if masks is None or np.max(masks) == 0:
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), "<b>No cells detected.</b>", metrics
        
    features_df = compute_nuclei_features(
        im_label=masks, im_nuclei=im_nuclei,          
        morphometry_features_flag=True, fsd_features_flag=False,
        intensity_features_flag=True, gradient_features_flag=False, haralick_features_flag=False
    )
    valid_labels = features_df['Label'].values.astype(int)

    raw_transparency = _safe_float(configs.get('mask_transparency', 0.5))
    _mask_alpha = max(0.0, min(1.0, 1.0 - (raw_transparency / 100.0 if raw_transparency > 1.0 else raw_transparency)))
    _positivity_thresh = _safe_float(configs.get('positivity_thresh', 0))
    _min_area_px = um2_to_px(_safe_float(configs.get('min_area_um2', 1)), mpp)


   
    base_labels, base_ki67_pos = classify_ki67_labb_approach(
        frame_rgb, masks, min_area=_min_area_px, threshold=_positivity_thresh
    )

    base_pos_dict = dict(zip(base_labels, base_ki67_pos))
    features_df['is_positive'] = features_df['Label'].map(base_pos_dict).fillna(False).astype(bool)
    features_df['is_kept'] = True 
    # ====================================================================
    # Refine ALL Cells
    # ===================================================================
    refine_method = configs.get("Refine Cell Subpopulations", "None") 
    show_preview = configs.get("Show Refinement Preview", True)
    warning_html = "" 
    cluster_table_html = "" 
    final_vis_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    preview_drawn = False 

    if refine_method == "K-Means" and len(features_df) > 0:
        all_idx = features_df.index 
        selected_feats = configs.get("Refine Features (Multi-Select)", [])
        keep_grps = configs.get("Select target groups (Multi-Select)", [])
        
        if not selected_feats:
            warning_html = "<div style='color:#e74c3c; margin-top:10px;'>⚠️ <b>Warning:</b> Please select at least 1 feature to enable clustering.</div>"
        else:
            k = int(_safe_float(configs.get("K Clusters", 2), 2))
            valid_feats = [f for f in selected_feats if f in features_df.columns]
            keep_indices = [int(g.replace("Group", "")) for g in keep_grps] if keep_grps else []
            
            if len(valid_feats) > 0 and len(all_idx) >= k:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                X = features_df.loc[all_idx, valid_feats].fillna(0).values
                X_scaled = StandardScaler().fit_transform(X)
                kmeans = KMeans(n_clusters=k, random_state=42)
                sub_labels = kmeans.fit_predict(X_scaled)
                
                if 'Size.Area' in features_df.columns:
                    centers = [features_df.loc[all_idx[sub_labels == i], 'Size.Area'].mean() if (sub_labels == i).any() else 0 for i in range(k)]
                    mapping = {old: new for new, old in enumerate(np.argsort(centers))}
                    sub_labels = np.array([mapping[l] for l in sub_labels])

                c_colors_bgr = [[0,255,0],[0,0,255]] if k==2 else [] 
                c_colors_hex = ["#2ecc71", "#e74c3c"] if k==2 else []
                if k > 2:
                    import matplotlib.pyplot as plt
                    cmap = plt.get_cmap('gist_rainbow')
                    for i in range(k):
                        rgba = cmap(i/max(1,k-1))
                        c_colors_hex.append("#{:02x}{:02x}{:02x}".format(int(rgba[0]*255),int(rgba[1]*255),int(rgba[2]*255)))
                        c_colors_bgr.append([int(rgba[2]*255),int(rgba[1]*255),int(rgba[0]*255)])

                cluster_table_html = "<div style='margin-top:10px; border:1px solid #ddd; border-radius:4px; overflow:hidden;'>"
                cluster_table_html += "<table width='100%' cellspacing='0' cellpadding='4' style='font-size:8.5pt;'>"
                cluster_table_html += "<tr style='background:#f8f9fa;'><th style='text-align:left;'>Group</th><th style='text-align:center;'>Area (μm²)</th><th style='text-align:center;'>Int. (M±S)</th><th style='text-align:center;'>Solid.</th><th style='text-align:center;'>Ecce.</th></tr>"
                
                filtered_details = []
                for i in range(k):
                    g_mask = (sub_labels == i)
                    g_df = features_df.loc[all_idx[g_mask]] # 🚀
                    n = len(g_df)
                    m_area = g_df['Size.Area'].mean() * (mpp**2) if n>0 else 0
                    m_int = g_df['Nucleus.Intensity.Mean'].mean() if n>0 else 0
                    s_int = g_df['Nucleus.Intensity.Std'].mean() if n>0 else 0
                    m_sol = g_df['Shape.Solidity'].mean() if n>0 else 0
                    m_ecc = g_df['Shape.Eccentricity'].mean() if n>0 else 0
                    
                    status = "✔" if i in keep_indices else "✖"
                    if i not in keep_indices and n > 0: filtered_details.append(f"{n} from Group{i}")
                    
                    cluster_table_html += f"<tr><td style='text-align:left; color:{c_colors_hex[i]};'><b>■ Group{i}</b> (n={n}) {status}</td>"
                    cluster_table_html += f"<td style='text-align:center;'>{m_area:.1f}</td><td style='text-align:center;'>{m_int:.1f}±{s_int:.1f}</td>"
                    cluster_table_html += f"<td style='text-align:center;'>{m_sol:.2f}</td><td style='text-align:center;'>{m_ecc:.2f}</td></tr>"
                cluster_table_html += "</table></div>"
                
                if not keep_grps:
                    warning_html = "<div style='color:#f39c12; font-size:9pt; margin-top:8px;'>⚠️ <b>Warning:</b> No groups selected. Please select subpopulations to analyze.</div>"
                elif filtered_details:
                    warning_html = f"<div style='color:#f39c12; font-size:9pt; margin-top:8px;'>* Excluded: {', '.join(filtered_details)}</div>"
                else:
                    warning_html = f"<div style='color:#2ecc71; font-size:9pt; margin-top:8px;'>* All cells retained for analysis</div>"

                if show_preview:
                    overlay = final_vis_bgr.copy()
                    lut = np.zeros((masks.max()+1, 3), dtype=np.uint8)
                    p_mask = np.zeros(masks.max()+1, dtype=bool)
                    for lab, cid in zip(features_df.loc[all_idx, 'Label'].values, sub_labels):
                        lut[lab] = c_colors_bgr[cid]; p_mask[lab] = True
                    overlay[p_mask[masks]] = lut[masks][p_mask[masks]]
                    final_vis_bgr = cv2.addWeighted(final_vis_bgr, 1-_mask_alpha, overlay, _mask_alpha, 0)
                    
                    preview_drawn = True 
                
                if keep_grps:
                    keep_mask = np.isin(sub_labels, keep_indices)
                    features_df.loc[all_idx, 'is_kept'] = keep_mask

    elif refine_method == "Threshold" and len(features_df) > 0:
        all_idx = features_df.index
        feat = configs.get("Refine Feature", "Size.Area")
        cutoff = _safe_float(configs.get("Threshold Cutoff", 0.0))
        if feat in features_df.columns:
            vals = features_df.loc[all_idx, feat].values.copy()
            if 'Area' in feat or 'Size' in feat: vals *= (mpp**2)
            keep = (vals >= cutoff)
            features_df.loc[all_idx, 'is_kept'] = keep # 🚀
            
            n_filtered = (~keep).sum()
            if n_filtered > 0:
                short_feat = feat.split('.')[-1]
                cluster_table_html = f"<div style='color:#f39c12; font-size:9pt; margin-top:8px;'>* Excluded: {n_filtered} cells ({short_feat} < {cutoff})</div>"

    # ====================================================================
    # 6. final visualization and Ki-67 index calculation (Only on the refined subset!)
    # ====================================================================
    if not preview_drawn:
        final_vis_bgr = visualize_ki67_refined_overlay(
            final_vis_bgr, masks, valid_labels,
            features_df['is_kept'].values, features_df['is_positive'].values, alpha=_mask_alpha
        )

    kept_df = features_df[features_df['is_kept']]
    num_pos = int(kept_df['is_positive'].sum())
    num_total = len(kept_df)
    pos_percent = (num_pos / num_total * 100) if num_total > 0 else 0
    
    is_actually_refined = (refine_method != "None" and "⚠️" not in warning_html)
    index_label = "Gated Ki-67 Index" if is_actually_refined else "Ki-67 Index"

    text = (
        f"<b style='color:#9b59b6;'>{index_label}:</b> {pos_percent:.2f}%<br>"  
        f"<span style='color:#e74c3c;'>Target Positives: {num_pos}</span><br>"
        f"<span style='color:#3498db;'>Target Negatives: {num_total - num_pos}</span>" 
    )
    
    text += warning_html
    text += cluster_table_html

    metrics["mib_pos"] = num_pos
    metrics["mib_total"] = num_total
    
    return final_vis_bgr, text, metrics