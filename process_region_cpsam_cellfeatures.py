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
# 之前的辅助函数保持不变 (_safe_float, fix_region, visualize_overlay, get_tissue_mask_global)
# get_tile_coordinates 已经不再需要了，可以保留也可以删除

def _safe_float(value, default=0.2):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def fix_region(region, tile_size):
    reg = region.copy()
    # 即使不切片，保持一定的尺寸倍数通常对卷积神经网络也是好的（虽然这里主要为了防止报错）
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
    # gray_image = color.rgb2gray(img_rgb)
    # thresh = filters.threshold_otsu(gray_image)
    # tissue_mask = gray_image < thresh*1.3
    # tissue_mask = morphology.remove_small_objects(tissue_mask, max_size=500)  # 移除小于 500 像素的区域
    # tissue_mask = morphology.remove_small_holes(tissue_mask,max_size=500 )  # 填充小于 500 像素的孔洞
    return tissue_mask

def get_foreground_window_scale():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shcore = ctypes.WinDLL("Shcore", use_last_error=True)

    PROCESS_PER_MONITOR_DPI_AWARE = 2
    shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)

    MDT_EFFECTIVE_DPI = 0
    MONITOR_DEFAULTTONEAREST = 2

    def get_scale_for_foreground_window():
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())

        hMonitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        if not hMonitor:
            raise ctypes.WinError(ctypes.get_last_error())

        dpiX = wintypes.UINT()
        dpiY = wintypes.UINT()

        result = shcore.GetDpiForMonitor(
            hMonitor,
            MDT_EFFECTIVE_DPI,
            ctypes.byref(dpiX),
            ctypes.byref(dpiY)
        )
        if result != 0:
            raise OSError(f"GetDpiForMonitor failed, HRESULT={result}")

        scale_percent = dpiX.value / 96 * 100
        return {
            "hwnd": hwnd,
            "dpi": dpiX.value,
            "scale_percent": scale_percent
        }

    info = get_scale_for_foreground_window()
    print(f"当前前台窗口所在显示器: DPI={info['dpi']}, 缩放={info['scale_percent']:.0f}%")
    return info['scale_percent']/100

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
        print("SVD did not converge. Using default stain matrix.")
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
        
    # 1. 截图
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
    


    
    #masks = clear_border(masks)

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
            #masks = relabel_and_filter_masks(masks, min_area=20, max_area=50000, min_solidity=0.80)
            if cluster_method == "Customized Features":
                                
                                image_type = configs.get('Image Type', 'H&E Cell Analysis')
                                
                                if image_type == "H&E Cell Analysis":
                                    masks = clear_border(masks)
                                    selected_feats = configs.get('H&E Features (Multi-Select)', [])
                                    
                                    # 🚀 Fix: Define a basic metrics dict before returning
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
                                        im_cytoplasm=im_cytoplasm,     
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
                                    final_vis_img = fast_cluster_overlay(frame, masks, valid_labels, clusters, k_clusters, alpha=_overlay_alpha)

                                elif image_type == "Muscle Fiber Typing":
                                    selected_feats = configs.get('Muscle Features (Multi-Select)', [])
                                    print("Selected features from config:", selected_feats)
                                    
                                    # 🚀 Fix: Define a basic metrics dict before returning
                                    if len(selected_feats) == 0:
                                        err_txt = "<b style='color:#e74c3c;'>Error: No features selected!</b><br>Please select at least 1 feature from feature list."
                                        blank_metrics = {
                                            "cell_count": 0, "cellularity_percent": 0.0,
                                            "area_px": frame.shape[0] * frame.shape[1], "mpp": mpp
                                        }
                                        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), err_txt, blank_metrics
                                        
                                    print("Selected features for clustering:", selected_feats)
                                    

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
                                    print("Extracted features columns:", features_df.columns.tolist())
                                    
                                    valid_labels = features_df['Label'].values.astype(int)
                                    
                                    df_cluster = custom_clustering(features_df, selected_features=selected_feats, use_pca=False)
                                    clusters = perform_keamns(df_cluster, n_clusters=k_clusters)
                                    
                                    features_df['Cluster'] = clusters 
                                    final_vis_img = fast_cluster_overlay(frame, masks, valid_labels, clusters, k_clusters, alpha=_overlay_alpha)



            elif cluster_method == "Handcrafted Features":
                masks = clear_border(masks)
                features_df = compute_nuclei_features(
                im_label=masks,
                im_nuclei=im_nuclei,          
                im_cytoplasm=im_cytoplasm,     
                morphometry_features_flag=True,
                fsd_features_flag=False,
                intensity_features_flag=True,
                gradient_features_flag=True,
                haralick_features_flag=True
                )
                valid_labels = features_df['Label'].values.astype(int)
                # Extract numbers, scale, and cluster
                # feat_cols = [c for c in features_df.columns if c not in ['Label']]
                # X = features_df[feat_cols].fillna(0).values
                # X_scaled = StandardScaler().fit_transform(X)
                # clusters = KMeans(n_clusters=k_clusters, random_state=42).fit_predict(X_scaled)
                df_cluster = hand_crafted_clustering(features_df, use_pca=False)
                clusters = perform_keamns(df_cluster, n_clusters=k_clusters)
                features_df['Cluster'] = clusters #
                final_vis_img = fast_cluster_overlay(frame, masks, valid_labels, clusters, k_clusters, alpha=_overlay_alpha)
            elif cluster_method == "Handcrafted Features(PCA)":
                masks = clear_border(masks)
                features_df = compute_nuclei_features(
                im_label=masks,
                im_nuclei=im_nuclei,          
                im_cytoplasm=im_cytoplasm,     
                morphometry_features_flag=True,
                fsd_features_flag=False,
                intensity_features_flag=True,
                gradient_features_flag=True,
                haralick_features_flag=True
                )
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
                print ("Using Lemon for region clustering with context scale:", context_scale)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                crops, df_meta = build_cell_crops(frame, masks, output_size=40, context_scale=context_scale, min_side=40,max_side=96)
                lemon_model, lemon_transform, device = get_cached_lemon(device)
                
                embs = extract_embeddings_lemon(crops, lemon_model, lemon_transform, device)
                clusters = cluster_embeddings_lemon(embs, k=k_clusters)
                cluster_dict = dict(zip(df_meta['label'].values, clusters))
                features_df['Cluster'] = features_df['Label'].map(cluster_dict)
                final_vis_img = fast_cluster_overlay(frame, masks, df_meta['label'].values, clusters, k_clusters, alpha=_overlay_alpha)
# 在 process_region_cpsam_cellfeatures.py 中
            elif cluster_method == "Cascade Clustering":
                masks = clear_border(masks)
                # 🚀 核心对齐：调用和 H&E 模式一模一样的特征提取函数
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
                else:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler
                    
                    for step in pipeline:
                        target = step['target']
                        feats = step['features']
                        k = step['k']
                        step_id = step['step_id']
                        
                        mask = (features_df['Cluster_Path'] == target)
                        # 只有选了特征且细胞够多才聚类
                        if mask.sum() >= k and len(feats) > 0:
                            valid_feats = [f for f in feats if f in features_df.columns]
                            X = features_df.loc[mask, valid_feats].fillna(0).values
                            X_scaled = StandardScaler().fit_transform(X)
                            
                            kmeans = KMeans(n_clusters=k, random_state=42)
                            labels = kmeans.fit_predict(X_scaled)
                            
                            # 名字生成，例如 S1-C0, S1-C1
                            features_df.loc[mask, 'Cluster_Path'] = [f"S{step_id}-C{l}" for l in labels]
                    
                    # 映射回整数 ID 供渲染
                    unique_paths = sorted(features_df['Cluster_Path'].unique())
                    path_to_id = {p: i for i, p in enumerate(unique_paths)}
                    features_df['Cluster'] = features_df['Cluster_Path'].map(path_to_id)
                    actual_k = len(unique_paths)
                    clusters = features_df['Cluster'].values
                    # 🚀 [关键] 记录名字字典供 HTML 表头使用
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
    # 4. 特征统计与多列对比看板 (动态分发: Classic vs Pivot)
    # =========================================================================
    image_type = configs.get('Image Type', 'H&E Cell Analysis')
    display_category = configs.get('Feature Category', 'Nuclear Size')

    # >>> [核心修改 1]：根据 Image Type 动态定义需要统计的特征池与表头显示名称 >>>
    if image_type == "Muscle Fiber Typing":
        categories = {
            "Nuclear Size": ['Area', 'Perimeter', 'axis_major_length', 'axis_minor_length', 'equivalent_diameter_area'],
            "Nuclear Shape": ['eccentricity', 'solidity'],
            "Intensity": ['intensity_mean', 'intensity_median', 'intensity_max', 'intensity_min', 'intensity_std']
        }
        # 为了不改 settings.json 的选项名，我们在界面渲染时动态把 Nuclear 改为 Fiber
        display_title_map = {
            "Nuclear Size": "Fiber Size", 
            "Nuclear Shape": "Fiber Shape", 
            "Intensity": "Fiber Intensity"
        }
        # 兼容不同写法的长度和面积特征，用于后续 MPP 换算
        linear_feats = ['axis_major_length', 'axis_minor_length', 'equivalent_diameter_area', 'Perimeter']
        area_feats = ['Area']
    else:
        # 保持原有的 H&E 特征定义不变
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
    # <<< [修改结束 1] <<<

    target_columns = [col for sublist in categories.values() for col in sublist]
    existing_cols = [c for c in target_columns if c in features_df.columns]

    # --- A. 计算整体特征 (All Cells) 均值、标准差和中位数 ---
    report_mean_all = features_df[existing_cols].mean()
    report_std_all = features_df[existing_cols].std()
    report_median_all = features_df[existing_cols].median()

    # >>> [核心修改 2]：物理单位 (MPP) 动态换算 >>>
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
    # <<< [修改结束 2] <<<

    # --- B. 计算 Cluster 细分特征 (仅在开启聚类时) ---
    actual_k = 0
    cluster_hex_colors = []
    cluster_stats = {}

    if cluster_method != "None" and 'Cluster' in features_df.columns:
        actual_k = int(features_df['Cluster'].max() + 1)
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('gist_rainbow')
        
        for i in range(actual_k):
            rgba = cmap(i / max(1, k_clusters - 1)) 
            hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
            cluster_hex_colors.append(hex_color)
            
            cdf = features_df[features_df['Cluster'] == i]
            c_mean = cdf[existing_cols].mean()
            c_med = cdf[existing_cols].median()
            
            # 物理单位转换 (兼容双路)
            for feat in linear_feats:
                if feat in c_mean: 
                    c_mean[feat] *= mpp
                    c_med[feat] *= mpp
            for feat in area_feats:
                if feat in c_mean: 
                    c_mean[feat] *= (mpp ** 2)
                    c_med[feat] *= (mpp ** 2)
                
            cluster_stats[i] = {'mean': c_mean, 'median': c_med, 'count': len(cdf)}

    # --- C. 计算整体占比与 Metrics 封装 ---
    tissue_mask = get_tissue_mask_global(frame)
    tissue_area_px = np.sum(tissue_mask)
    #tissue_area_px = h * w 
    print(f"Tissue area (px): {tissue_area_px}, Total area (px): {area_px}")
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
# >>> [关键修复]：将所有提取出的特征统计值打包进 metrics，供 Export 导出 CSV >>>
    # 1. 导出整体 (All Cells) 的特征
    for col in existing_cols:
        if col in report_mean_all:
            metrics[f"{col}_mean"] = float(report_mean_all[col]) if not np.isnan(report_mean_all[col]) else 0.0
            metrics[f"{col}_std"] = float(report_std_all[col]) if not np.isnan(report_std_all[col]) else 0.0
            metrics[f"{col}_median"] = float(report_median_all[col]) if not np.isnan(report_median_all[col]) else 0.0
            
    # 2. 如果开启了聚类，把各个 Cluster (C0, C1...) 的细分特征也导出来
    if actual_k > 0:
        for i in range(actual_k):
            c_mean = cluster_stats[i]['mean']
            c_med = cluster_stats[i]['median']
            metrics[f"C{i}_cell_count"] = cluster_stats[i]['count'] # 记录这个 Cluster 有多少细胞
            
            for col in existing_cols:
                if col in c_mean:
                    metrics[f"C{i}_{col}_mean"] = float(c_mean[col]) if not np.isnan(c_mean[col]) else 0.0
                    metrics[f"C{i}_{col}_median"] = float(c_med[col]) if not np.isnan(c_med[col]) else 0.0
    # <<< [修复结束] <<<
    # 获取动态表头标题
    display_title = display_title_map.get(display_category, display_category)
    
    val_fmt = ".2f" if display_category in ["Nuclear Shape", "Intensity"] else ".1f"
    
    # =========================================================================
    # D. 动态生成 HTML (两种完全不同的视图，且兼容 H&E 与 Muscle)
    # =========================================================================
    
    if cluster_method == "None":
        # ---------------------------------------------------------
        # 视图 1：经典无聚类视图 (Mean ± Std | Median)
        # ---------------------------------------------------------
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
        
        # >>> [核心修改 3]：根据 Image Type 决定加载哪个 Row Block >>>
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
            # 原版 H&E
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
        # <<< [修改结束 3] <<<

        html_text += "</table></div>" 
        
    else:
        # ---------------------------------------------------------
        # 视图 2：聚类后的多列高级透视表 (智能排布 Mean 和 Median)
        # ---------------------------------------------------------
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

        # >>> [核心修改 4]：透视表模式同样区分 H&E 和 Muscle >>>
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
            # 原版 H&E
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
        # <<< [修改结束 4] <<<

        html_text += "</table></div>" 

    text = html_text 
    # =========================================================================

    return cv2.cvtColor(final_vis_img, cv2.COLOR_RGB2BGR).astype(np.uint8), text, metrics
    # ==================================================================================================================================================
# dynamic dashboard with intelligent feature selection, automatic unit conversion, and dual-mode display (classic vs pivot) based on user configs and image type. The code is structured to be easily extensible for future feature categories and image types without hardcoding, ensuring a robust and user-friendly experience.
   # ==================================================================================================================================================
# # =========================================================================
#     # 4. 特征统计与多列对比看板 (完全动态化，从 JSON 提取)
#     # =========================================================================
#     image_type = configs.get('Image Type', 'H&E Cell Analysis')
#     display_category = configs.get('Feature Category', 'Nuclear Size')
    
#     # 动态表头替换 (把 Nuclear 换成 Fiber)
#     display_title = display_category.replace("Nuclear", "Fiber") if image_type == "Muscle Fiber Typing" else display_category

#     # >>> [核心改进 1] 直接从原始 metadata 读取全量特征列表，拒绝 Hardcode >>>
#     if image_type == "Muscle Fiber Typing":
#         all_possible_feats = metadata.get('additional_configs', {}).get('Muscle Features (Multi-Select)', [])
#     else:
#         all_possible_feats = metadata.get('additional_configs', {}).get('H&E Features (Multi-Select)', [])
#     # <<< [改进结束] <<<

#     # >>> [核心改进 2] 智能关键字分类器：根据用户选的 Category 自动挑出对应的特征 >>>
#     target_columns = []
#     for feat in all_possible_feats:
#         f_low = feat.lower()
#         if "size" in display_category.lower():
#             if any(k in f_low for k in ['area', 'perimeter', 'length', 'diameter']): target_columns.append(feat)
#         elif "shape" in display_category.lower():
#             if any(k in f_low for k in ['circularity', 'eccentricity', 'solidity', 'extent']): target_columns.append(feat)
#         elif "intensity" in display_category.lower():
#             if any(k in f_low for k in ['intensity', 'entropy', 'energy', 'std', 'iqr', 'mean', 'max', 'min']): target_columns.append(feat)

#     # 如果智能分类没挑出任何东西，就兜底显示全部特征
#     if not target_columns: 
#         target_columns = all_possible_feats

#     existing_cols = [c for c in target_columns if c in features_df.columns]

#     # --- A. 计算整体特征 (All Cells) 均值、标准差和中位数 ---
#     report_mean_all = features_df[existing_cols].mean()
#     report_std_all = features_df[existing_cols].std()
#     report_median_all = features_df[existing_cols].median()

#     # >>> [核心改进 3] 智能单位推断与自动 MPP 换算 >>>
#     feat_units = {}
#     feat_multipliers = {}
#     for feat in existing_cols:
#         f_low = feat.lower()
#         if 'area' in f_low:
#             feat_multipliers[feat] = mpp ** 2
#             feat_units[feat] = "μm²"
#         elif any(k in f_low for k in ['perimeter', 'length', 'diameter']):
#             feat_multipliers[feat] = mpp
#             feat_units[feat] = "μm"
#         else:
#             feat_multipliers[feat] = 1.0
#             feat_units[feat] = ""

#         # 应用换算
#         if feat in report_mean_all:
#             report_mean_all[feat] *= feat_multipliers[feat]
#             report_std_all[feat] *= feat_multipliers[feat]
#             report_median_all[feat] *= feat_multipliers[feat]
#     # <<< [改进结束] <<<

#     # --- B. 计算 Cluster 细分特征 ---
#     actual_k = 0
#     cluster_hex_colors = []
#     cluster_stats = {}

#     if cluster_method != "None" and 'Cluster' in features_df.columns:
#         actual_k = int(features_df['Cluster'].max() + 1)
#         import matplotlib.pyplot as plt
#         cmap = plt.get_cmap('gist_rainbow')
        
#         for i in range(actual_k):
#             rgba = cmap(i / max(1, k_clusters - 1)) 
#             cluster_hex_colors.append("#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)))
            
#             cdf = features_df[features_df['Cluster'] == i]
#             c_mean = cdf[existing_cols].mean()
#             c_med = cdf[existing_cols].median()
            
#             # 对各 Cluster 应用动态 MPP 换算
#             for feat in existing_cols:
#                 if feat in c_mean:
#                     c_mean[feat] *= feat_multipliers[feat]
#                     c_med[feat] *= feat_multipliers[feat]
                
#             cluster_stats[i] = {'mean': c_mean, 'median': c_med, 'count': len(cdf)}

#     # --- C. 计算整体占比与 Metrics 封装 ---
#     tissue_mask = get_tissue_mask_global(frame)
#     tissue_area_px = np.sum(tissue_mask)
#     nuclei_area_px = np.sum((masks > 0) & tissue_mask)
#     cellularity_score = (nuclei_area_px / tissue_area_px) if tissue_area_px > 0 else 0

#     metrics = {
#         "cell_count": int(len(features_df)),
#         "cellularity_percent": float(cellularity_score * 100),
#         "area_px": frame.shape[0] * frame.shape[1],
#         "orig_img": cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB),
#         "mpp": mpp
#     }

#     val_fmt = ".2f" if display_category in ["Nuclear Shape", "Intensity"] else ".1f"
    
#     # 动态美化特征显示名称 (比如 Nucleus.Intensity.Mean 变成 Mean, axis_major_length 变成 Axis Major Length)
#     def pretty_name(f_name):
#         return f_name.split('.')[-1].replace('_', ' ').title()

#     # =========================================================================
#     # D. 动态生成 HTML (利用循环，消灭全部 Hardcode)
#     # =========================================================================
#     if cluster_method == "None":
#         def make_row_classic(name, key, unit=""):
#             m_val = report_mean_all.get(key, 0)
#             s_val = report_std_all.get(key, 0)
#             med_val = report_median_all.get(key, 0)
#             unit_str = f"&nbsp;<span style='color:#777777; font-size:9pt;'>{unit}</span>" if unit else ""
#             return f"""
#             <tr bgcolor="#ffffff">
#                 <td style="padding: 6px;"><b style="color:#000000;">{name}</b></td>
#                 <td align="right" style="padding: 6px;"><span style="color:#000000; font-size:11pt; font-weight:bold;">{m_val:{val_fmt}}</span> <span style="color:#555555; font-size:9pt;">±{s_val:{val_fmt}}</span>{unit_str}</td>
#                 <td align="right" style="padding: 6px;"><span style="color:#f39c12; font-size:11pt; font-weight:bold;">{med_val:{val_fmt}}</span>{unit_str}</td>
#             </tr>"""
            
#         html_text = f"""
#         <div style="font-family: 'Segoe UI', Arial, sans-serif;">
#             <div style="background-color: #ffffff; padding: 10px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid #1abc9c;">
#                 <span style="color: #555555; font-size: 10pt;">Count:</span> <span style="color: #000000; font-size: 12pt; font-weight: bold; margin-right: 20px;">{metrics['cell_count']}</span>
#                 <span style="color: #555555; font-size: 10pt;">Density/Ratio:</span> <span style="color: #1abc9c; font-size: 12pt; font-weight: bold;">{metrics['cellularity_percent']:.2f}%</span>
#             </div>
#             <table width="100%" cellpadding="0" cellspacing="1" bgcolor="#f2f2f2" style="margin-top: 5px;">
#                 <tr bgcolor="#ffffff">
#                     <th align="left" style="padding: 8px; color: #3498db; font-size: 10pt;">{display_title}</th>
#                     <th align="right" style="padding: 8px; color: #3498db; font-size: 10pt;">Mean ± Std</th>
#                     <th align="right" style="padding: 8px; color: #3498db; font-size: 10pt;">Median</th>
#                 </tr>
#         """
#         # >>> 一行代码搞定所有特征渲染！<<<
#         for feat in existing_cols:
#             html_text += make_row_classic(pretty_name(feat), feat, feat_units[feat])
#         html_text += "</table></div>" 
        
#     else:
#         html_text = f"""
#         <div style="font-family: 'Segoe UI', Arial, sans-serif;">
#             <div style="background-color: #ffffff; padding: 10px; border-radius: 6px; margin-bottom: 10px; border-left: 4px solid #1abc9c;">
#                 <span style="color: #555555; font-size: 10pt;">Total Count:</span> <span style="color: #000000; font-size: 12pt; font-weight: bold; margin-right: 20px;">{metrics['cell_count']}</span>
#                 <span style="color: #555555; font-size: 10pt;">Density/Ratio:</span> <span style="color: #1abc9c; font-size: 12pt; font-weight: bold;">{metrics['cellularity_percent']:.2f}%</span>
#             </div>
#             <table width="100%" cellpadding="0" cellspacing="0" style="text-align: center; background-color: #ffffff; border-radius: 6px; overflow: hidden; border: 1px solid #e0e0e0;">
#                 <tr style="background-color: #f8f9fa;">
#                     <th align="left" style="padding: 8px; color: #333; font-size: 10pt; border-bottom: 2px solid #ccc;">{display_title}</th>
#                     <th align="right" style="padding: 8px; color: #333; font-size: 10pt; border-bottom: 2px solid #ccc;">All Data</th>
#         """
#         for i in range(actual_k):
#             count = cluster_stats[i]["count"]
#             html_text += f'<th align="right" style="padding: 8px; border-bottom: 2px solid #ccc; font-size: 10pt;"><span style="color:{cluster_hex_colors[i]}; font-size:13pt; vertical-align:middle;">■</span> C{i}<br><span style="font-size:8pt; color:#888; font-weight:normal;">n={count}</span></th>'
#         html_text += "</tr>"

#         def make_row_pivot(name, key, unit=""):
#             m_all = report_mean_all.get(key, 0)
#             s_all = report_std_all.get(key, 0)
#             med_all = report_median_all.get(key, 0)
#             unit_str = f" <span style='color:#999; font-size:8pt;'>{unit}</span>" if unit else ""
            
#             row_html = f"""
#             <tr>
#                 <td align="left" style="padding: 8px; border-bottom: 1px solid #eee;"><b style="color:#444;">{name}</b>{unit_str}</td>
#                 <td align="right" style="padding: 8px; border-bottom: 1px solid #eee;">
#                     <div style="color:#000; font-size:10pt; font-weight:bold;">{m_all:{val_fmt}} <span style="color:#888; font-size:8pt; font-weight:normal;">±{s_all:{val_fmt}}</span></div>
#                     <div style="color:#f39c12; font-size:9pt; font-weight:bold; margin-top:2px;">M: {med_all:{val_fmt}}</div>
#                 </td>
#             """
#             for i in range(actual_k):
#                 c_mean = cluster_stats[i]['mean'].get(key, 0)
#                 c_med = cluster_stats[i]['median'].get(key, 0)
#                 row_html += f"""
#                 <td align="right" style="padding: 8px; border-bottom: 1px solid #eee;">
#                     <div style="color:#111; font-weight:600; font-size:10pt;">{c_mean:{val_fmt}}</div>
#                     <div style="color:#f39c12; font-size:9pt; font-weight:bold; margin-top:2px;">M: {c_med:{val_fmt}}</div>
#                 </td>
#                 """
#             row_html += "</tr>"
#             return row_html

#         # >>> 一行代码搞定所有透视表特征渲染！<<<
#         for feat in existing_cols:
#             html_text += make_row_pivot(pretty_name(feat), feat, feat_units[feat])

#         html_text += "</table></div>" 

#     text = html_text 
#     # =========================================================================

#     return cv2.cvtColor(final_vis_img, cv2.COLOR_RGB2BGR).astype(np.uint8), text, metrics

    # ==================================================================================================================================================
# dynamic dashboard with intelligent feature selection, automatic unit conversion, and dual-mode display (classic vs pivot) based on user configs and image type. The code is structured to be easily extensible for future feature categories and image types without hardcoding, ensuring a robust and user-friendly experience.
   # ==================================================================================================================================================

# # =========================================================================
#     # 4. 特征统计与多列对比看板 (动态分发: Classic vs Pivot)
#     # =========================================================================
#     categories = {
#         "Nuclear Size": ['Size.Area', 'Size.Perimeter', 'Size.MajorAxisLength', 'Size.MinorAxisLength', 'Shape.EquivalentDiameter'],
#         "Nuclear Shape": ['Shape.Circularity', 'Shape.Eccentricity', 'Shape.Solidity', 'Shape.Extent'],
#         "Intensity": ['Nucleus.Intensity.Mean', 'Nucleus.Intensity.Std', 'Nucleus.Intensity.IQR', 'Nucleus.Intensity.HistEntropy', 'Nucleus.Intensity.HistEnergy']
#     }
#     target_columns = [col for sublist in categories.values() for col in sublist]
#     existing_cols = [c for c in target_columns if c in features_df.columns]

#     # --- A. 计算整体特征 (All Cells) 均值、标准差和中位数 ---
#     report_mean_all = features_df[existing_cols].mean()
#     report_std_all = features_df[existing_cols].std()
#     report_median_all = features_df[existing_cols].median()

#     linear_feats = ['Size.MajorAxisLength', 'Size.MinorAxisLength', 'Shape.EquivalentDiameter', 'Size.Perimeter']
#     for feat in linear_feats:
#         if feat in report_mean_all:
#             report_mean_all[feat] *= mpp
#             report_std_all[feat] *= mpp
#             report_median_all[feat] *= mpp
#     if 'Size.Area' in report_mean_all:
#         report_mean_all['Size.Area'] *= (mpp ** 2)
#         report_std_all['Size.Area'] *= (mpp ** 2)
#         report_median_all['Size.Area'] *= (mpp ** 2)

#     # --- B. 计算 Cluster 细分特征 (仅在开启聚类时) ---
#     actual_k = 0
#     cluster_hex_colors = []
#     cluster_stats = {}

#     if cluster_method != "None" and 'Cluster' in features_df.columns:
#         actual_k = int(features_df['Cluster'].max() + 1)
#         import matplotlib.pyplot as plt
#         cmap = plt.get_cmap('gist_rainbow')
        
#         for i in range(actual_k):
#             rgba = cmap(i / max(1, k_clusters - 1)) 
#             hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
#             cluster_hex_colors.append(hex_color)
            
#             cdf = features_df[features_df['Cluster'] == i]
#             c_mean = cdf[existing_cols].mean()
#             c_med = cdf[existing_cols].median()
            
#             # 物理单位转换
#             for feat in linear_feats:
#                 if feat in c_mean: 
#                     c_mean[feat] *= mpp
#                     c_med[feat] *= mpp
#             if 'Size.Area' in c_mean: 
#                 c_mean['Size.Area'] *= (mpp ** 2)
#                 c_med['Size.Area'] *= (mpp ** 2)
                
#             cluster_stats[i] = {'mean': c_mean, 'median': c_med, 'count': len(cdf)}

#     # --- C. 计算整体占比与 Metrics 封装 ---
#     tissue_mask = get_tissue_mask_global(frame)
#     tissue_area_px = np.sum(tissue_mask)
#     nuclei_area_px = np.sum((masks > 0) & tissue_mask)
#     cellularity_score = (nuclei_area_px / tissue_area_px) if tissue_area_px > 0 else 0

#     metrics = {
#         "cell_count": int(len(features_df)),
#         "cellularity_percent": float(cellularity_score * 100),
#         "area_px": frame.shape[0] * frame.shape[1],
#         "orig_img": cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB),
#         "mpp": mpp
#     }

#     display_category = configs.get('Feature Category', 'Nuclear Size')
    
#     # >>> [精度控制]：根据当前显示的特征类别，动态决定小数位数 >>>
#     # Shape 特征极其微小（0~1之间），必须保留 2 位小数
#     # Size 和 Intensity 数值较大，保留 1 位能让透视表更加紧凑美观
#     val_fmt = ".2f" if display_category in ["Nuclear Shape", "Intensity"] else ".1f"
#     # <<< [精度控制] <<<
    
#     # =========================================================================
#     # D. 动态生成 HTML (两种完全不同的视图)
#     # =========================================================================
    
#     if cluster_method == "None":
#         # ---------------------------------------------------------
#         # 视图 1：经典无聚类视图 (Mean ± Std | Median)
#         # ---------------------------------------------------------
#         def make_row_classic(name, key, unit=""):
#             m_val = report_mean_all.get(key, 0)
#             s_val = report_std_all.get(key, 0)
#             med_val = report_median_all.get(key, 0)
#             unit_str = f"&nbsp;<span style='color:#777777; font-size:9pt;'>{unit}</span>" if unit else ""
            
#             # 注意这里使用了 {m_val:{val_fmt}} 来动态应用 .1f 或 .2f
#             return f"""
#             <tr bgcolor="#ffffff">
#                 <td style="padding: 6px;"><b style="color:#000000;">{name}</b></td>
#                 <td align="right" style="padding: 6px;">
#                     <span style="color:#000000; font-size:11pt; font-weight:bold;">{m_val:{val_fmt}}</span> 
#                     <span style="color:#555555; font-size:9pt;">±{s_val:{val_fmt}}</span>{unit_str}
#                 </td>
#                 <td align="right" style="padding: 6px;">
#                     <span style="color:#f39c12; font-size:11pt; font-weight:bold;">{med_val:{val_fmt}}</span>{unit_str}
#                 </td>
#             </tr>
#             """
            
#         html_text = f"""
#         <div style="font-family: 'Segoe UI', Arial, sans-serif;">
#             <div style="background-color: #ffffff; padding: 10px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid #1abc9c;">
#                 <span style="color: #555555; font-size: 10pt;">Cell Count:</span> 
#                 <span style="color: #000000; font-size: 12pt; font-weight: bold; margin-right: 20px;">{metrics['cell_count']}</span>
#                 <span style="color: #555555; font-size: 10pt;">Cellularity:</span> 
#                 <span style="color: #1abc9c; font-size: 12pt; font-weight: bold;">{metrics['cellularity_percent']:.2f}%</span>
#             </div>
#             <table width="100%" cellpadding="0" cellspacing="1" bgcolor="#f2f2f2" style="margin-top: 5px;">
#                 <tr bgcolor="#ffffff">
#                     <th align="left" style="padding: 8px; color: #3498db; font-size: 10pt;">{display_category}</th>
#                     <th align="right" style="padding: 8px; color: #3498db; font-size: 10pt;">Mean ± Std</th>
#                     <th align="right" style="padding: 8px; color: #3498db; font-size: 10pt;">Median</th>
#                 </tr>
#         """
        
#         if display_category == "Nuclear Size":
#             html_text += make_row_classic("Area", "Size.Area", "μm²")
#             html_text += make_row_classic("Perimeter", "Size.Perimeter", "μm")
#             html_text += make_row_classic("Major Axis", "Size.MajorAxisLength", "μm")
#             html_text += make_row_classic("Minor Axis", "Size.MinorAxisLength", "μm")
#             html_text += make_row_classic("Eq. Diameter", "Shape.EquivalentDiameter", "μm")
#         elif display_category == "Nuclear Shape":
#             html_text += make_row_classic("Circularity", "Shape.Circularity")
#             html_text += make_row_classic("Eccentricity", "Shape.Eccentricity")
#             html_text += make_row_classic("Solidity", "Shape.Solidity")
#             html_text += make_row_classic("Extent", "Shape.Extent")
#         elif display_category == "Intensity":
#             html_text += make_row_classic("Intensity (Mean)", "Nucleus.Intensity.Mean")
#             html_text += make_row_classic("Intensity (Std)", "Nucleus.Intensity.Std")
#             html_text += make_row_classic("Intensity (IQR)", "Nucleus.Intensity.IQR")
#             html_text += make_row_classic("Intensity (Entropy)", "Nucleus.Intensity.HistEntropy")
#             html_text += make_row_classic("Intensity (Energy)", "Nucleus.Intensity.HistEnergy")
#         else:
#             html_text += make_row_classic("Area", "Size.Area", "μm²")

#         html_text += "</table></div>" 
        
#     else:
#         # ---------------------------------------------------------
#         # 视图 2：聚类后的多列高级透视表 (智能排布 Mean 和 Median)
#         # ---------------------------------------------------------
#         html_text = f"""
#         <div style="font-family: 'Segoe UI', Arial, sans-serif;">
#             <div style="background-color: #ffffff; padding: 10px; border-radius: 6px; margin-bottom: 10px; border-left: 4px solid #1abc9c;">
#                 <span style="color: #555555; font-size: 10pt;">Total Cells:</span> 
#                 <span style="color: #000000; font-size: 12pt; font-weight: bold; margin-right: 20px;">{metrics['cell_count']}</span>
#                 <span style="color: #555555; font-size: 10pt;">Cellularity:</span> 
#                 <span style="color: #1abc9c; font-size: 12pt; font-weight: bold;">{metrics['cellularity_percent']:.2f}%</span>
#             </div>
            
#             <table width="100%" cellpadding="0" cellspacing="0" style="text-align: center; background-color: #ffffff; border-radius: 6px; overflow: hidden; border: 1px solid #e0e0e0;">
#                 <tr style="background-color: #f8f9fa;">
#                     <th align="left" style="padding: 8px; color: #333; font-size: 10pt; border-bottom: 2px solid #ccc;">{display_category}</th>
#                     <th align="right" style="padding: 8px; color: #333; font-size: 10pt; border-bottom: 2px solid #ccc;">All Cells</th>
#         """
#         for i in range(actual_k):
#             count = cluster_stats[i]["count"]
#             html_text += f'<th align="right" style="padding: 8px; border-bottom: 2px solid #ccc; font-size: 10pt;"><span style="color:{cluster_hex_colors[i]}; font-size:13pt; vertical-align:middle;">■</span> C{i}<br><span style="font-size:8pt; color:#888; font-weight:normal;">n={count}</span></th>'
#         html_text += "</tr>"

#         def make_row_pivot(name, key, unit=""):
#             m_all = report_mean_all.get(key, 0)
#             s_all = report_std_all.get(key, 0)
#             med_all = report_median_all.get(key, 0)
#             unit_str = f" <span style='color:#999; font-size:8pt;'>{unit}</span>" if unit else ""
            
#             # 动态应用 {val_fmt}
#             row_html = f"""
#             <tr>
#                 <td align="left" style="padding: 8px; border-bottom: 1px solid #eee;"><b style="color:#444;">{name}</b>{unit_str}</td>
#                 <td align="right" style="padding: 8px; border-bottom: 1px solid #eee;">
#                     <div style="color:#000; font-size:10pt; font-weight:bold;">{m_all:{val_fmt}} <span style="color:#888; font-size:8pt; font-weight:normal;">±{s_all:{val_fmt}}</span></div>
#                     <div style="color:#f39c12; font-size:9pt; font-weight:bold; margin-top:2px;">M: {med_all:{val_fmt}}</div>
#                 </td>
#             """
            
#             for i in range(actual_k):
#                 c_mean = cluster_stats[i]['mean'].get(key, 0)
#                 c_med = cluster_stats[i]['median'].get(key, 0)
#                 row_html += f"""
#                 <td align="right" style="padding: 8px; border-bottom: 1px solid #eee;">
#                     <div style="color:#111; font-weight:600; font-size:10pt;">{c_mean:{val_fmt}}</div>
#                     <div style="color:#f39c12; font-size:9pt; font-weight:bold; margin-top:2px;">M: {c_med:{val_fmt}}</div>
#                 </td>
#                 """
#             row_html += "</tr>"
#             return row_html

#         if display_category == "Nuclear Size":
#             html_text += make_row_pivot("Area", "Size.Area", "μm²")
#             html_text += make_row_pivot("Perimeter", "Size.Perimeter", "μm")
#             html_text += make_row_pivot("Major Axis", "Size.MajorAxisLength", "μm")
#             html_text += make_row_pivot("Minor Axis", "Size.MinorAxisLength", "μm")
#             html_text += make_row_pivot("Eq. Diameter", "Shape.EquivalentDiameter", "μm")
#         elif display_category == "Nuclear Shape":
#             html_text += make_row_pivot("Circularity", "Shape.Circularity")
#             html_text += make_row_pivot("Eccentricity", "Shape.Eccentricity")
#             html_text += make_row_pivot("Solidity", "Shape.Solidity")
#             html_text += make_row_pivot("Extent", "Shape.Extent")
#         elif display_category == "Intensity":
#             html_text += make_row_pivot("Intensity (Mean)", "Nucleus.Intensity.Mean")
#             html_text += make_row_pivot("Intensity (Std)", "Nucleus.Intensity.Std")
#             html_text += make_row_pivot("Intensity (IQR)", "Nucleus.Intensity.IQR")
#             html_text += make_row_pivot("Intensity (Entropy)", "Nucleus.Intensity.HistEntropy")
#             html_text += make_row_pivot("Intensity (Energy)", "Nucleus.Intensity.HistEnergy")
#         else:
#             html_text += make_row_pivot("Area", "Size.Area", "μm²")

#         html_text += "</table></div>" 

#     text = html_text 
#     # =========================================================================

#     return cv2.cvtColor(final_vis_img, cv2.COLOR_RGB2BGR).astype(np.uint8), text, metrics
# # 4. 定义需要存入 metrics 的特征组
#     categories = {
#         "Nuclear Size": [
#     'Size.Area',
#     'Size.Perimeter',
#     'Size.MajorAxisLength',
#     'Size.MinorAxisLength',
#     'Shape.EquivalentDiameter',],

#         "Nuclear Shape": [    
#     'Shape.Circularity',
#     'Shape.Eccentricity',
#     'Shape.Solidity',
#     'Shape.Extent',],

#         "Intensity": [    
#     'Nucleus.Intensity.Mean',
#     'Nucleus.Intensity.Std',
#     'Nucleus.Intensity.IQR',
#     'Nucleus.Intensity.HistEntropy',
#     'Nucleus.Intensity.HistEnergy']
#     }
    
#     # 展平所有列名用于提取
#     target_columns = [col for sublist in categories.values() for col in sublist]
#     existing_cols = [c for c in target_columns if c in features_df.columns]
    
#     # >>> [修改开始] 同时计算均值、标准差和中位数 >>>
#     report_mean = features_df[existing_cols].mean()
#     report_std = features_df[existing_cols].std()
#     report_median = features_df[existing_cols].median() # <--- 新增计算中位数
#     print("frame shape:", frame.shape)
#     print("mpp used for conversion:", mpp)
#     print("Size of Area mean:", report_mean.get('Size.Area', 'N/A'))


    
#     # 物理转换 (微米) - 均值、标准差和中位数都要按相同比例转换
#     linear_feats = ['Size.MajorAxisLength', 'Size.MinorAxisLength', 'Shape.EquivalentDiameter', 'Size.Perimeter']
#     for feat in linear_feats:
#         if feat in report_mean:
#             report_mean[feat] *= mpp
#             report_std[feat] *= mpp
#             report_median[feat] *= mpp
#     if 'Size.Area' in report_mean:
#         report_mean['Size.Area'] *= (mpp ** 2)
#         report_std['Size.Area'] *= (mpp ** 2)
#         report_median['Size.Area'] *= (mpp ** 2)

#     # 5. 计算 Cellularity
#     tissue_mask = get_tissue_mask_global(frame)
#     tissue_area_px = np.sum(tissue_mask)
#     nuclei_area_px = np.sum((masks > 0) & tissue_mask)
#     cellularity_score = (nuclei_area_px / tissue_area_px) if tissue_area_px > 0 else 0

#     # 6. 构建最终的 metrics 字典 (包含所有请求的特征)
#     metrics = {
#         "cell_count": int(len(features_df)),
#         "area_px":   frame.shape[0] * frame.shape[1],
#         "cellularity_percent": float(cellularity_score * 100),
#         "area_px": frame.shape[0] * frame.shape[1],
#         "orig_img":  cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB),
#         "mpp": mpp
#     }
    
#     # 将均值、标准差和中位数分别存入 metrics
#     for col in existing_cols:
#         metrics[f"{col}_mean"] = float(report_mean[col]) if not np.isnan(report_mean[col]) else 0.0
#         metrics[f"{col}_std"] = float(report_std[col]) if not np.isnan(report_std[col]) else 0.0
#         metrics[f"{col}_median"] = float(report_median[col]) if not np.isnan(report_median[col]) else 0.0
#     # <<< [修改结束] <<<

# # 7. 准备可视化和输出文本
#     # if show_overlay:
#     #     final_vis_img = visualize_overlay(frame, masks, alpha=_overlay_alpha)
#     # else:
#     #     final_vis_img = frame

# # >>> [修改开始 3] 酷炫的 HTML 数据仪表盘显示 (适配白色背景) >>>
#     display_category = configs.get('Feature Category', 'Nuclear Size')

#     # 定义一个辅助函数来生成 HTML 表格行，适配白色背景主题
#     def make_row(name, key_prefix, unit=""):
#         mean_val = metrics.get(f"{key_prefix}_mean", 0)
#         std_val = metrics.get(f"{key_prefix}_std", 0)
#         med_val = metrics.get(f"{key_prefix}_median", 0)
        
#         # 单位使用灰色小号字体
#         unit_str = f"&nbsp;<span style='color:#777777; font-size:9pt;'>{unit}</span>" if unit else ""
        
#         return f"""
#         <tr bgcolor="#ffffff">
#             <td style="padding: 6px;"><b style="color:#000000;">{name}</b></td>
#             <td align="right" style="padding: 6px;">
#                 <span style="color:#000000; font-size:11pt; font-weight:bold;">{mean_val:.2f}</span> 
#                 <span style="color:#555555; font-size:9pt;">±{std_val:.2f}</span>{unit_str}
#             </td>
#             <td align="right" style="padding: 6px;">
#                 <span style="color:#f39c12; font-size:11pt; font-weight:bold;">{med_val:.2f}</span>{unit_str}
#             </td>
#         </tr>
#         """

#     # 构建头部：总数和细胞密度卡片 (使用动态数据 metrics)
#     html_text = f"""
#     <div style="font-family: 'Segoe UI', Arial, sans-serif;">
#         <div style="background-color: #ffffff; padding: 10px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid #1abc9c;">
#             <span style="color: #555555; font-size: 10pt;">Cell Count:</span> 
#             <span style="color: #000000; font-size: 12pt; font-weight: bold; margin-right: 20px;">{metrics['cell_count']}</span>
            
#             <span style="color: #555555; font-size: 10pt;">Cellularity:</span> 
#             <span style="color: #1abc9c; font-size: 12pt; font-weight: bold;">{metrics['cellularity_percent']:.2f}%</span>
#         </div>
        
#         <table width="100%" cellpadding="0" cellspacing="1" bgcolor="#f2f2f2" style="margin-top: 5px;">
#             <tr bgcolor="#ffffff">
#                 <th align="left" style="padding: 8px; color: #3498db; font-size: 10pt;">{display_category}</th>
#                 <th align="right" style="padding: 8px; color: #3498db; font-size: 10pt;">Mean ± Std</th>
#                 <th align="right" style="padding: 8px; color: #3498db; font-size: 10pt;">Median</th>
#             </tr>
#     """

#     # 根据下拉菜单动态插入表格数据行
#     if display_category == "Nuclear Size":
#         html_text += make_row("Area", "Size.Area", "μm²")
#         html_text += make_row("Perimeter", "Size.Perimeter", "μm")
#         html_text += make_row("Major Axis", "Size.MajorAxisLength", "μm")
#         html_text += make_row("Minor Axis", "Size.MinorAxisLength", "μm")
#         html_text += make_row("Eq. Diameter", "Shape.EquivalentDiameter", "μm")
        
#     elif display_category == "Nuclear Shape":
#         html_text += make_row("Circularity", "Shape.Circularity")
#         html_text += make_row("Eccentricity", "Shape.Eccentricity")
#         html_text += make_row("Solidity", "Shape.Solidity")
#         html_text += make_row("Extent", "Shape.Extent")
#     elif display_category == "Intensity":
#         html_text += make_row("Intensity (Mean)", "Nucleus.Intensity.Mean")
#         html_text += make_row("Intensity (Std)", "Nucleus.Intensity.Std")
#         html_text += make_row("Intensity (IQR)", "Nucleus.Intensity.IQR")
#         html_text += make_row("Intensity (Entropy)", "Nucleus.Intensity.HistEntropy")
#         html_text += make_row("Intensity (Energy)", "Nucleus.Intensity.HistEnergy")
#     else:
#         html_text += make_row(" Area", "Size.Area", "μm²")

#     # 闭合 HTML 标签
#     html_text += "</table></div>" 
#     text = html_text 
#     # <<< [修改结束 3] <<<

#     return cv2.cvtColor(final_vis_img, cv2.COLOR_RGB2BGR).astype(np.uint8), text, metrics