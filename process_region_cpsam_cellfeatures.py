import mss
import numpy as np
import cv2
from skimage.color import rgb2hsv
from histomicstk.features import compute_nuclei_features
import histomicstk as htk
from numpy.linalg import LinAlgError
import ctypes
from ctypes import wintypes
from skimage.segmentation import clear_border
import warnings
import pandas as pd
import torch
from skimage.measure import regionprops, regionprops_table
warnings.filterwarnings("ignore", category=FutureWarning)
from utils_clustering import (
    relabel_and_filter_masks, build_cell_crops,get_cached_lemon,
      extract_embeddings_lemon, fast_cluster_overlay,hand_crafted_clustering,perform_keamns, 
      cluster_embeddings_lemon, get_cached_kaiko, extract_embeddings_kaiko, custom_clustering
)


def _safe_float(value, default=0.2):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def fix_region(region, tile_size):
    reg = region.copy()
    reg['width'] = max(reg['width'], tile_size)
    reg['height'] = max(reg['height'], tile_size)
    return reg

def visualize_overlay(img, mask, color=(0, 255, 0), alpha=0.5):
    if alpha <= 0:
        return img
    overlay = img.copy()
    cell_region = (mask > 0)
    overlay[cell_region] = color
    blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return blended

def get_tissue_mask_global(img_rgb):
    hsv = rgb2hsv(img_rgb)

    saturation = hsv[:, :, 1]

    tissue_mask = saturation > 0.05
    return tissue_mask

def get_foreground_window_scale():
    # user32 = ctypes.WinDLL("user32", use_last_error=True)
    # shcore = ctypes.WinDLL("Shcore", use_last_error=True)

    # PROCESS_PER_MONITOR_DPI_AWARE = 2
    # shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)

    # MDT_EFFECTIVE_DPI = 0
    # MONITOR_DEFAULTTONEAREST = 2

    # def get_scale_for_foreground_window():
    #     hwnd = user32.GetForegroundWindow()
    #     if not hwnd:
    #         raise ctypes.WinError(ctypes.get_last_error())

    #     hMonitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    #     if not hMonitor:
    #         raise ctypes.WinError(ctypes.get_last_error())

    #     dpiX = wintypes.UINT()
    #     dpiY = wintypes.UINT()

    #     result = shcore.GetDpiForMonitor(
    #         hMonitor,
    #         MDT_EFFECTIVE_DPI,
    #         ctypes.byref(dpiX),
    #         ctypes.byref(dpiY)
    #     )
    #     if result != 0:
    #         raise OSError(f"GetDpiForMonitor failed, HRESULT={result}")

    #     scale_percent = dpiX.value / 96 * 100
    #     return {
    #         "hwnd": hwnd,
    #         "dpi": dpiX.value,
    #         "scale_percent": scale_percent
    #     }
    # info = get_scale_for_foreground_window()
    #return info['scale_percent']/100
    return 1.0

def deconvolution_hema_eso(img_rgb):
    I_0 = 255
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    try:
        W_estimated = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(img_rgb, I_0)
        h_index = htk.preprocessing.color_deconvolution.find_stain_index(
            stain_color_map['hematoxylin'], W_estimated
        )
        e_index = htk.preprocessing.color_deconvolution.find_stain_index(
            stain_color_map['eosin'], W_estimated
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
        stains = ['hematoxylin', 'eosin', 'null']
        W_custom = np.array([stain_color_map[st] for st in stains]).T
        deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(
            img_rgb, W_custom, I_0
        )

    im_nuclei = deconv_result.Stains[:, :, 0]
    im_cytoplasm = deconv_result.Stains[:, :, 1]
    return im_nuclei, im_cytoplasm



def process_region(region, **kwargs):
    metadata = kwargs['metadata']
    model = kwargs['model']
    configs = kwargs.get('additional_configs', {})
    cluster_method = configs.get('Clustering Method', 'None')
    k_clusters = int(_safe_float(configs.get('Clusters (k)', 3)))
    k_clusters = max(1, k_clusters)
    context_scale = _safe_float(configs.get('Context Scale', 1.5))
    tile_size = metadata.get('tile_size', 256) 
    show_overlay = configs.get('show_overlay', True)
    
    raw_transparency = _safe_float(configs.get('mask_transparency', 0.5))
    if raw_transparency > 1.0:
        raw_transparency = raw_transparency / 100.0
        
    _overlay_alpha = max(0.0, min(1.0, 1.0 - raw_transparency))
    mpp = _safe_float(configs.get('mpp', metadata.get('mpp', 0.25)))
    is_calibrated = configs.get('is_calibrated', False)
    if not is_calibrated:
        os_scale = get_foreground_window_scale()
        mpp = mpp / os_scale
        
    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)

    frame_orig = np.array(screenshot, dtype=np.uint8)
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGRA2RGB)
    h, w = frame.shape[:2]

    masks, _,_ = model.eval(frame)
    
    im_nuclei, im_cytoplasm = deconvolution_hema_eso(frame)

    area_px = h * w
    
    if masks is None or np.max(masks) == 0:
        metrics = {
            "cell_count":0,
            "cellularity_percent":0.0,
            "area_px":frame.shape[0]*frame.shape[1],
            "mpp":mpp
        }
        text = "<b>No nuclei detected</b>"

        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), text, metrics

    
    final_vis_img = None
    if cluster_method =="None":
            masks = clear_border(masks)
            
            features_df = compute_nuclei_features(
            im_label=masks,
            im_nuclei=im_nuclei,          
            #im_cytoplasm=im_cytoplasm,     
            morphometry_features_flag=True,
            fsd_features_flag=False,
            intensity_features_flag=True,
            gradient_features_flag=False,
            haralick_features_flag=False
        )
    elif cluster_method != "None":
            # filter out large objects
            # bincounts = np.bincount(masks.ravel())
            # if len(bincounts) > 1:  
            #     bincounts[0] = 0  
            #     largest_label = np.argmax(bincounts) 
            #     if largest_label > 0:
            #         masks[masks == largest_label] = 0 
            if cluster_method == "Customized Features":
                         
                                image_type = configs.get('Image Type', 'H&E Cell Analysis')
                                
                                if image_type == "H&E Cell Analysis":
                                    masks = clear_border(masks)
                                    selected_feats = configs.get('H&E Features (Multi-Select)', [])
                                    
                                    if len(selected_feats) == 0:
                                        err_txt = "<b style='color:#e74c3c;'>Error: No features selected!</b><br>Please select at least 1 feature in Additional Configs."
                                        blank_metrics = {
                                            "cell_count": 0, "cellularity_percent": 0.0,
                                            "area_px": frame.shape[0] * frame.shape[1], "mpp": mpp
                                        }
                                        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), err_txt, blank_metrics
                                        
                                    features_df = compute_nuclei_features(
                                        im_label=masks,
                                        im_nuclei=im_nuclei,          
                                        #im_cytoplasm=im_cytoplasm,     
                                        morphometry_features_flag=True,
                                        fsd_features_flag=False,
                                        intensity_features_flag=True,
                                        gradient_features_flag=False,
                                        haralick_features_flag=False
                                    )

                                    valid_labels = features_df['Label'].values.astype(int)
                                    df_cluster = custom_clustering(features_df, selected_features=selected_feats, use_pca=False)
                                    clusters = perform_keamns(df_cluster, n_clusters=k_clusters)
                                    features_df['Cluster'] = clusters 
                                    #features_df.to_csv("debug_he_features.csv", index=False)
                                    final_vis_img = fast_cluster_overlay(frame, masks, valid_labels, clusters, k_clusters, alpha=_overlay_alpha)

                                elif image_type == "Muscle Fiber Typing":
                                    selected_feats = configs.get('Muscle Features (Multi-Select)', [])
                                   # print("Selected features from config:", selected_feats)
                                    
                                    if len(selected_feats) == 0:
                                        err_txt = "<b style='color:#e74c3c;'>Error: No features selected!</b><br>Please select at least 1 feature from feature list."
                                        blank_metrics = {
                                            "cell_count": 0, "cellularity_percent": 0.0,
                                            "area_px": frame.shape[0] * frame.shape[1], "mpp": mpp
                                        }
                                        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), err_txt, blank_metrics
                                        
                                    #print("Selected features for clustering:", selected_feats)
                                    

                                    full_props = [
                                        'label', 'area', 'perimeter', 'equivalent_diameter_area', 
                                        'axis_major_length', 'axis_minor_length', 'eccentricity', 
                                        'solidity', 'intensity_mean', 'intensity_max', 'intensity_min',
                                        "intensity_median", "intensity_std","area"
                                    ]
                                    props_dict = regionprops_table(
                                        label_image=masks, 
                                        intensity_image=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 
                                        properties=full_props
                                    )
                                    features_df = pd.DataFrame(props_dict)
                                    rename_map = {'label': 'Label', 'area': 'Area', 'perimeter': 'Perimeter'}
                                    features_df.rename(columns=rename_map, inplace=True)
                                    #print("Extracted features columns:", features_df.columns.tolist())
                                    
                                    valid_labels = features_df['Label'].values.astype(int)
                                    
                                    df_cluster = custom_clustering(features_df, selected_features=selected_feats, use_pca=False)
                                    clusters = perform_keamns(df_cluster, n_clusters=k_clusters)
                                    
                                    features_df['Cluster'] = clusters 
                                    #features_df.to_csv("debug_muscle_features.csv", index=False)
                                    final_vis_img = fast_cluster_overlay(frame, masks, valid_labels, clusters, k_clusters, alpha=_overlay_alpha)



            elif cluster_method == "Handcrafted Features":
                masks = clear_border(masks)
                features_df = compute_nuclei_features(
                im_label=masks,
                im_nuclei=im_nuclei,          
                #im_cytoplasm=im_cytoplasm,     
                morphometry_features_flag=True,
                fsd_features_flag=False,
                intensity_features_flag=True,
                gradient_features_flag=True,
                haralick_features_flag=True
                )
                cols_to_drop = [c for c in features_df.columns if c.startswith('Identifier.')]
                features_df = features_df.drop(columns=[c for c in cols_to_drop if c in features_df.columns])
                valid_labels = features_df['Label'].values.astype(int)
                
                # Extract numbers, scale, and cluster
                # feat_cols = [c for c in features_df.columns if c not in ['Label']]
                # X = features_df[feat_cols].fillna(0).values
                # X_scaled = StandardScaler().fit_transform(X)
                # clusters = KMeans(n_clusters=k_clusters, random_state=42).fit_predict(X_scaled)
                df_cluster = hand_crafted_clustering(features_df, use_pca=False)
                clusters = perform_keamns(df_cluster, n_clusters=k_clusters)
                features_df['Cluster'] = clusters #
                #features_df.to_csv("debug_he_features_handcrafted.csv", index=False)
                final_vis_img = fast_cluster_overlay(frame, masks, valid_labels, clusters, k_clusters, alpha=_overlay_alpha)
            elif cluster_method == "Handcrafted Features(PCA)":
                masks = clear_border(masks)
                features_df = compute_nuclei_features(
                im_label=masks,
                im_nuclei=im_nuclei,          
                #im_cytoplasm=im_cytoplasm,     
                morphometry_features_flag=True,
                fsd_features_flag=False,
                intensity_features_flag=True,
                gradient_features_flag=True,
                haralick_features_flag=True
                )
                cols_to_drop = [c for c in features_df.columns if c.startswith('Identifier.')]
                features_df = features_df.drop(columns=[c for c in cols_to_drop if c in features_df.columns])
                valid_labels = features_df['Label'].values.astype(int)
                # Extract numbers, scale, and cluster
                # feat_cols = [c for c in features_df.columns if c not in ['Label']]
                # X = features_df[feat_cols].fillna(0).values
                # X_scaled = StandardScaler().fit_transform(X)
                # clusters = KMeans(n_clusters=k_clusters, random_state=42).fit_predict(X_scaled)
                df_cluster = hand_crafted_clustering(features_df, use_pca=True)
                clusters = perform_keamns(df_cluster, n_clusters=k_clusters)
                features_df['Cluster'] = clusters 
                final_vis_img = fast_cluster_overlay(frame, masks, valid_labels, clusters, k_clusters, alpha=_overlay_alpha)
            elif cluster_method == "Single Cell Clustering":
                masks = clear_border(masks)
                features_df = compute_nuclei_features(
                    im_label=masks,
                    im_nuclei=im_nuclei,          
                    im_cytoplasm=im_cytoplasm,     
                    morphometry_features_flag=True,
                    fsd_features_flag=False,
                    intensity_features_flag=True,
                    gradient_features_flag=False,
                    haralick_features_flag=False
                )
                #print ("Using Lemon for region clustering with context scale:", context_scale)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                crops, df_meta = build_cell_crops(frame, masks, output_size=40, context_scale=context_scale, min_side=40,max_side=96)
                lemon_model, lemon_transform, device = get_cached_lemon(device)
                
                embs = extract_embeddings_lemon(crops, lemon_model, lemon_transform, device)
                clusters = cluster_embeddings_lemon(embs, k=k_clusters)
                cluster_dict = dict(zip(df_meta['label'].values, clusters))
                features_df['Cluster'] = features_df['Label'].map(cluster_dict)
                final_vis_img = fast_cluster_overlay(frame, masks, df_meta['label'].values, clusters, k_clusters, alpha=_overlay_alpha)
            
            elif cluster_method == "Hierarchical Gating":
                masks = clear_border(masks)
                features_df = compute_nuclei_features(
                    im_label=masks, im_nuclei=im_nuclei, im_cytoplasm=im_cytoplasm,     
                    morphometry_features_flag=True, fsd_features_flag=False,
                    intensity_features_flag=True, gradient_features_flag=False, haralick_features_flag=False
                )
                valid_labels = features_df['Label'].values.astype(int)
                
                pipeline = configs.get('cascade_pipeline', [])
                features_df['Cluster_Path'] = "All Cells"
                
                if not pipeline:
                    features_df['Cluster'] = 0
                    actual_k = 1
                    clusters = features_df['Cluster'].values
                    cluster_name_map = {0: "All Cells"}
                else:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler
                    
                    area_feats = ['Size.Area', 'Area']
                    linear_feats = ['Size.Perimeter', 'Size.MajorAxisLength', 'Size.MinorAxisLength', 'Shape.EquivalentDiameter', 
                                    'Perimeter', 'axis_major_length', 'axis_minor_length', 'equivalent_diameter_area']

                    for step in pipeline:
                        target = step['target']
                        method = step.get('method', 'K-Means') 
                        k = step['k']
                        
                        mask = (features_df['Cluster_Path'] == target)
                        if mask.sum() == 0: continue

                        prefix = "C" if target == "All Cells" else f"{target}."
                        new_names = [f"{prefix}{i+1}" for i in range(k)]

                        if method == "K-Means":
                            feats = step['features']
                            if mask.sum() >= k and len(feats) > 0:
                                valid_feats = [f for f in feats if f in features_df.columns]
                                if not valid_feats: continue
                                
                                X = features_df.loc[mask, valid_feats].fillna(0).values
                                X_scaled = StandardScaler().fit_transform(X)
                                
                                kmeans = KMeans(n_clusters=k, random_state=42)
                                labels = kmeans.fit_predict(X_scaled)
                                
                                centers = kmeans.cluster_centers_.mean(axis=1)
                                mapping = {old: new for new, old in enumerate(np.argsort(centers))}
                                labels = np.array([mapping[l] for l in labels])
                                
                                features_df.loc[mask, 'Cluster_Path'] = [new_names[l] for l in labels]
                            else:
                                features_df.loc[mask, 'Cluster_Path'] = new_names[0]

                        elif method == "Threshold":
                            feat = step['feature']
                            thresh = step['threshold']
                            
                            if feat in features_df.columns:
                                vals = features_df.loc[mask, feat].fillna(0).values.copy()
                                
                                if feat in area_feats: vals = vals * (mpp ** 2)
                                elif feat in linear_feats: vals = vals * mpp
                                
                                labels = (vals >= thresh).astype(int)
                                features_df.loc[mask, 'Cluster_Path'] = [new_names[l] for l in labels]
                            else:
                                features_df.loc[mask, 'Cluster_Path'] = new_names[0]
                    

                    def sort_key(x):
                        if not isinstance(x, str) or not x.startswith('C'): return (999,)
                        return tuple(int(part) for part in x.replace('C', '').split('.'))
                    
                    unique_paths = sorted(features_df['Cluster_Path'].unique(), key=sort_key)
                    path_to_id = {p: i for i, p in enumerate(unique_paths)}
                    features_df['Cluster'] = features_df['Cluster_Path'].map(path_to_id)
                    actual_k = len(unique_paths)
                    clusters = features_df['Cluster'].values
                    cluster_name_map = {i: p for p, i in path_to_id.items()}


                final_vis_img = fast_cluster_overlay(frame, masks, valid_labels, clusters, actual_k, alpha=_overlay_alpha)

            # elif cluster_method == "Region Clustering":
            #     features_df = compute_nuclei_features(
            #         im_label=masks,
            #         im_nuclei=im_nuclei,          
            #         im_cytoplasm=im_cytoplasm,     
            #         morphometry_features_flag=True,
            #         fsd_features_flag=False,
            #         intensity_features_flag=True,
            #         gradient_features_flag=False,
            #         haralick_features_flag=False
            #     )
            #     print ("Using Kaiko for region clustering with context scale:", context_scale)
            #     device = "cuda" if torch.cuda.is_available() else "cpu"
            #     crops, df_meta = build_cell_crops(frame, masks, output_size=224, context_scale=context_scale, min_side=96,max_side=224)
            #     kaiko_model, kaiko_transform, device = get_cached_kaiko(device)
                
            #     embs = extract_embeddings_kaiko(crops, kaiko_model, kaiko_transform, device)
            #     clusters = cluster_embeddings_lemon(embs, k=k_clusters)
            #     cluster_dict = dict(zip(df_meta['label'].values, clusters))
            #     features_df['Cluster'] = features_df['Label'].map(cluster_dict)
            #     final_vis_img = fast_cluster_overlay(frame, masks, df_meta['label'].values, clusters, k_clusters, alpha=_overlay_alpha)
    if final_vis_img is None:
            if configs.get('show_overlay', True):
                final_vis_img = visualize_overlay(frame, masks, alpha=_overlay_alpha)
            else:
                final_vis_img = frame

    # =========================================================================
    # Feature summarization
    # =========================================================================
    image_type = configs.get('Image Type', 'H&E Cell Analysis')
    display_category = configs.get('Feature Category', 'Nuclear Size')

    if image_type == "Muscle Fiber Typing":
        categories = {
            "Nuclear Size": ['Area', 'Perimeter', 'axis_major_length', 'axis_minor_length', 'equivalent_diameter_area'],
            "Nuclear Shape": ['eccentricity', 'solidity'],
            "Intensity": ['intensity_mean', 'intensity_median', 'intensity_max', 'intensity_min', 'intensity_std']
        }
        display_title_map = {
            "Nuclear Size": "Fiber Size", 
            "Nuclear Shape": "Fiber Shape", 
            "Intensity": "Fiber Intensity"
        }
        linear_feats = ['axis_major_length', 'axis_minor_length', 'equivalent_diameter_area', 'Perimeter']
        area_feats = ['Area']
    else:
        categories = {
            "Nuclear Size": ['Size.Area', 'Size.Perimeter', 'Size.MajorAxisLength', 'Size.MinorAxisLength', 'Shape.EquivalentDiameter'],
            "Nuclear Shape": ['Shape.Circularity', 'Shape.Eccentricity', 'Shape.Solidity', 'Shape.Extent'],
            "Intensity": ['Nucleus.Intensity.Mean', 'Nucleus.Intensity.Std', 'Nucleus.Intensity.IQR', 'Nucleus.Intensity.HistEntropy', 'Nucleus.Intensity.HistEnergy']
        }
        display_title_map = {
            "Nuclear Size": "Nuclear Size", 
            "Nuclear Shape": "Nuclear Shape", 
            "Intensity": "Intensity"
        }
        linear_feats = ['Size.MajorAxisLength', 'Size.MinorAxisLength', 'Shape.EquivalentDiameter', 'Size.Perimeter']
        area_feats = ['Size.Area']

    target_columns = [col for sublist in categories.values() for col in sublist]
    existing_cols = [c for c in target_columns if c in features_df.columns]

    report_mean_all = features_df[existing_cols].mean()
    report_std_all = features_df[existing_cols].std()
    report_median_all = features_df[existing_cols].median()

    for feat in linear_feats:
        if feat in report_mean_all:
            report_mean_all[feat] *= mpp
            report_std_all[feat] *= mpp
            report_median_all[feat] *= mpp
            
    for feat in area_feats:
        if feat in report_mean_all:
            report_mean_all[feat] *= (mpp ** 2)
            report_std_all[feat] *= (mpp ** 2)
            report_median_all[feat] *= (mpp ** 2)


    actual_k = 0
    cluster_hex_colors = []
    cluster_stats = {}

    if cluster_method != "None" and 'Cluster' in features_df.columns:
        actual_k = int(features_df['Cluster'].max() + 1)
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('gist_rainbow')
        
        for i in range(actual_k):

            if actual_k == 2:
                if i == 0:
                    hex_color = "#ff0000"  # C0: red
                else:
                    hex_color = "#00ff00"  # C1: green
            else:
                rgba = cmap(i / max(1, actual_k - 1)) 
                hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
            
            cluster_hex_colors.append(hex_color)
            
            cdf = features_df[features_df['Cluster'] == i]
            c_mean = cdf[existing_cols].mean()
            c_med = cdf[existing_cols].median()
            
            for feat in linear_feats:
                if feat in c_mean: 
                    c_mean[feat] *= mpp
                    c_med[feat] *= mpp
            for feat in area_feats:
                if feat in c_mean: 
                    c_mean[feat] *= (mpp ** 2)
                    c_med[feat] *= (mpp ** 2)
                
            cluster_stats[i] = {'mean': c_mean, 'median': c_med, 'count': len(cdf)}


    tissue_mask = get_tissue_mask_global(frame)
    tissue_area_px = np.sum(tissue_mask)
    #print(f"Tissue area (px): {tissue_area_px}, Total area (px): {area_px}")
    nuclei_area_px = np.sum((masks > 0) & tissue_mask)
    cellularity_score = (nuclei_area_px / tissue_area_px) if tissue_area_px > 0 else 0

    tissue_area_px = np.sum(tissue_mask)
    cell_count = int(len(features_df))

    if tissue_area_px > 0:
        tissue_area_mm2 = tissue_area_px * (mpp ** 2) / 1_000_000
        cellularity_score = cell_count / tissue_area_mm2
    else:
        cellularity_score = 0.0

    metrics = {
        "cell_count": int(len(features_df)),
        #"cellularity_percent": float(cellularity_score * 100),
        "cellularity_percent": cellularity_score,
        "area_px": frame.shape[0] * frame.shape[1],
        "orig_img": cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB),
        "mpp": mpp
    }

    for col in existing_cols:
        if col in report_mean_all:
            metrics[f"{col}_mean"] = float(report_mean_all[col]) if not np.isnan(report_mean_all[col]) else 0.0
            metrics[f"{col}_std"] = float(report_std_all[col]) if not np.isnan(report_std_all[col]) else 0.0
            metrics[f"{col}_median"] = float(report_median_all[col]) if not np.isnan(report_median_all[col]) else 0.0
            

    if actual_k > 0:
        for i in range(actual_k):
            c_mean = cluster_stats[i]['mean']
            c_med = cluster_stats[i]['median']
            metrics[f"C{i}_cell_count"] = cluster_stats[i]['count'] 
            
            for col in existing_cols:
                if col in c_mean:
                    metrics[f"C{i}_{col}_mean"] = float(c_mean[col]) if not np.isnan(c_mean[col]) else 0.0
                    metrics[f"C{i}_{col}_median"] = float(c_med[col]) if not np.isnan(c_med[col]) else 0.0

    display_title = display_title_map.get(display_category, display_category)
    
    val_fmt = ".2f" if display_category in ["Nuclear Shape", "Intensity"] else ".1f"
    
    # =========================================================================
    # HTML
    # =========================================================================
    
    if cluster_method == "None":
        def make_row_classic(name, key, unit=""):
            m_val = report_mean_all.get(key, 0)
            s_val = report_std_all.get(key, 0)
            med_val = report_median_all.get(key, 0)
            unit_str = f"&nbsp;<span style='color:#777777; font-size:9pt;'>{unit}</span>" if unit else ""
            
            return f"""
            <tr bgcolor="#ffffff">
                <td style="padding: 6px;"><b style="color:#000000;">{name}</b></td>
                <td align="right" style="padding: 6px;">
                    <span style="color:#000000; font-size:11pt; font-weight:bold;">{m_val:{val_fmt}}</span> 
                    <span style="color:#555555; font-size:9pt;">±{s_val:{val_fmt}}</span>{unit_str}
                </td>
                <td align="right" style="padding: 6px;">
                    <span style="color:#f39c12; font-size:11pt; font-weight:bold;">{med_val:{val_fmt}}</span>{unit_str}
                </td>
            </tr>
            """
            
        html_text = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif;">
            <div style="background-color: #ffffff; padding: 10px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid #1abc9c;">
                <span style="color: #555555; font-size: 10pt;">Count:</span> 
                <span style="color: #000000; font-size: 12pt; font-weight: bold; margin-right: 20px;">{metrics['cell_count']}</span>
                <span style="color: #555555; font-size: 10pt;">Density/Ratio:</span> 
                <span style="color: #1abc9c; font-size: 12pt; font-weight: bold;">{metrics['cellularity_percent']:.0f} cells/mm²</span>
            </div>
            <table width="100%" cellpadding="0" cellspacing="1" bgcolor="#f2f2f2" style="margin-top: 5px;">
                <tr bgcolor="#ffffff">
                    <th align="left" style="padding: 8px; color: #3498db; font-size: 10pt;">{display_title}</th>
                    <th align="right" style="padding: 8px; color: #3498db; font-size: 10pt;">Mean ± Std</th>
                    <th align="right" style="padding: 8px; color: #3498db; font-size: 10pt;">Median</th>
                </tr>
        """
        
        if image_type == "Muscle Fiber Typing":
            if display_category == "Nuclear Size":
                html_text += make_row_classic("Area", "Area", "μm²")
                html_text += make_row_classic("Perimeter", "Perimeter", "μm")
                html_text += make_row_classic("Major Axis", "axis_major_length", "μm")
                html_text += make_row_classic("Minor Axis", "axis_minor_length", "μm")
                html_text += make_row_classic("Eq. Diameter", "equivalent_diameter_area", "μm")
            elif display_category == "Nuclear Shape":
                html_text += make_row_classic("Eccentricity", "eccentricity")
                html_text += make_row_classic("Solidity", "solidity")
            elif display_category == "Intensity":
                html_text += make_row_classic("Intensity (Mean)", "intensity_mean")
                html_text += make_row_classic("Intensity (Median)", "intensity_median")
                html_text += make_row_classic("Intensity (Max)", "intensity_max")
                html_text += make_row_classic("Intensity (Min)", "intensity_min")
                html_text += make_row_classic("Intensity (Std)", "intensity_std")
        else:
            if display_category == "Nuclear Size":
                html_text += make_row_classic("Area", "Size.Area", "μm²")
                html_text += make_row_classic("Perimeter", "Size.Perimeter", "μm")
                html_text += make_row_classic("Major Axis", "Size.MajorAxisLength", "μm")
                html_text += make_row_classic("Minor Axis", "Size.MinorAxisLength", "μm")
                html_text += make_row_classic("Eq. Diameter", "Shape.EquivalentDiameter", "μm")
            elif display_category == "Nuclear Shape":
                html_text += make_row_classic("Circularity", "Shape.Circularity")
                html_text += make_row_classic("Eccentricity", "Shape.Eccentricity")
                html_text += make_row_classic("Solidity", "Shape.Solidity")
                html_text += make_row_classic("Extent", "Shape.Extent")
            elif display_category == "Intensity":
                html_text += make_row_classic("Intensity (Mean)", "Nucleus.Intensity.Mean")
                html_text += make_row_classic("Intensity (Std)", "Nucleus.Intensity.Std")
                html_text += make_row_classic("Intensity (IQR)", "Nucleus.Intensity.IQR")
                html_text += make_row_classic("Intensity (Entropy)", "Nucleus.Intensity.HistEntropy")
                html_text += make_row_classic("Intensity (Energy)", "Nucleus.Intensity.HistEnergy")

        html_text += "</table></div>" 
        
    else:
        html_text = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif;">
            <div style="background-color: #ffffff; padding: 10px; border-radius: 6px; margin-bottom: 10px; border-left: 4px solid #1abc9c;">
                <span style="color: #555555; font-size: 10pt;">Total Count:</span> 
                <span style="color: #000000; font-size: 12pt; font-weight: bold; margin-right: 20px;">{metrics['cell_count']}</span>
                <span style="color: #555555; font-size: 10pt;">Density/Ratio:</span> 
                <span style="color: #1abc9c; font-size: 12pt; font-weight: bold;">{metrics['cellularity_percent']:.0f} cells/mm²</span>
            </div>
            
            <table width="100%" cellpadding="0" cellspacing="0" style="text-align: center; background-color: #ffffff; border-radius: 6px; overflow: hidden; border: 1px solid #e0e0e0;">
                <tr style="background-color: #f8f9fa;">
                    <th align="left" style="padding: 8px; color: #333; font-size: 10pt; border-bottom: 2px solid #ccc;">{display_title}</th>
                    <th align="right" style="padding: 8px; color: #333; font-size: 10pt; border-bottom: 2px solid #ccc;">All Data</th>
        """
        for i in range(actual_k):
            count = cluster_stats[i]["count"]
            c_name = cluster_name_map.get(i, f"C{i}") if 'cluster_name_map' in locals() else f"C{i}"
            html_text += f'<th align="right" style="padding: 8px; border-bottom: 2px solid #ccc; font-size: 10pt;"><span style="color:{cluster_hex_colors[i]}; font-size:13pt; vertical-align:middle;">■</span> {c_name}<br><span style="font-size:8pt; color:#888; font-weight:normal;">n={count}</span></th>'
        html_text += "</tr>"

        def make_row_pivot(name, key, unit=""):
            m_all = report_mean_all.get(key, 0)
            s_all = report_std_all.get(key, 0)
            med_all = report_median_all.get(key, 0)
            unit_str = f" <span style='color:#999; font-size:8pt;'>{unit}</span>" if unit else ""
            
            row_html = f"""
            <tr>
                <td align="left" style="padding: 8px; border-bottom: 1px solid #eee;"><b style="color:#444;">{name}</b>{unit_str}</td>
                <td align="right" style="padding: 8px; border-bottom: 1px solid #eee;">
                    <div style="color:#000; font-size:10pt; font-weight:bold;">{m_all:{val_fmt}} <span style="color:#888; font-size:8pt; font-weight:normal;">±{s_all:{val_fmt}}</span></div>
                    <div style="color:#f39c12; font-size:9pt; font-weight:bold; margin-top:2px;">M: {med_all:{val_fmt}}</div>
                </td>
            """
            for i in range(actual_k):
                c_mean = cluster_stats[i]['mean'].get(key, 0)
                c_med = cluster_stats[i]['median'].get(key, 0)
                row_html += f"""
                <td align="right" style="padding: 8px; border-bottom: 1px solid #eee;">
                    <div style="color:#111; font-weight:600; font-size:10pt;">{c_mean:{val_fmt}}</div>
                    <div style="color:#f39c12; font-size:9pt; font-weight:bold; margin-top:2px;">M: {c_med:{val_fmt}}</div>
                </td>
                """
            row_html += "</tr>"
            return row_html

        if image_type == "Muscle Fiber Typing":
            if display_category == "Nuclear Size":
                html_text += make_row_pivot("Area", "Area", "μm²")
                html_text += make_row_pivot("Perimeter", "Perimeter", "μm")
                html_text += make_row_pivot("Major Axis", "axis_major_length", "μm")
                html_text += make_row_pivot("Minor Axis", "axis_minor_length", "μm")
                html_text += make_row_pivot("Eq. Diameter", "equivalent_diameter_area", "μm")
            elif display_category == "Nuclear Shape":
                html_text += make_row_pivot("Eccentricity", "eccentricity")
                html_text += make_row_pivot("Solidity", "solidity")
            elif display_category == "Intensity":
                html_text += make_row_pivot("Intensity (Mean)", "intensity_mean")
                html_text += make_row_pivot("Intensity (Median)", "intensity_median")
                html_text += make_row_pivot("Intensity (Max)", "intensity_max")
                html_text += make_row_pivot("Intensity (Min)", "intensity_min")
                html_text += make_row_pivot("Intensity (Std)", "intensity_std")
        else:
            if display_category == "Nuclear Size":
                html_text += make_row_pivot("Area", "Size.Area", "μm²")
                html_text += make_row_pivot("Perimeter", "Size.Perimeter", "μm")
                html_text += make_row_pivot("Major Axis", "Size.MajorAxisLength", "μm")
                html_text += make_row_pivot("Minor Axis", "Size.MinorAxisLength", "μm")
                html_text += make_row_pivot("Eq. Diameter", "Shape.EquivalentDiameter", "μm")
            elif display_category == "Nuclear Shape":
                html_text += make_row_pivot("Circularity", "Shape.Circularity")
                html_text += make_row_pivot("Eccentricity", "Shape.Eccentricity")
                html_text += make_row_pivot("Solidity", "Shape.Solidity")
                html_text += make_row_pivot("Extent", "Shape.Extent")
            elif display_category == "Intensity":
                html_text += make_row_pivot("Intensity (Mean)", "Nucleus.Intensity.Mean")
                html_text += make_row_pivot("Intensity (Std)", "Nucleus.Intensity.Std")
                html_text += make_row_pivot("Intensity (IQR)", "Nucleus.Intensity.IQR")
                html_text += make_row_pivot("Intensity (Entropy)", "Nucleus.Intensity.HistEntropy")
                html_text += make_row_pivot("Intensity (Energy)", "Nucleus.Intensity.HistEnergy")

        html_text += "</table></div>" 

    text = html_text 

    return cv2.cvtColor(final_vis_img, cv2.COLOR_RGB2BGR).astype(np.uint8), text, metrics
