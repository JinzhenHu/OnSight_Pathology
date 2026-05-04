import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if os.environ.get("BUILD_TYPE", "").upper() == "CPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_DUMMY_DEVICE"] = "1"
import torch
#from instanseg import InstanSeg
from skimage import io
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table
import numpy as np
import pandas as pd
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.morphology import binary_closing, disk, remove_small_objects
from scipy.ndimage import binary_fill_holes
import cv2
from skimage.filters import threshold_otsu
import histomicstk.features as htk_features
from cellpose import models
import histomicstk as htk
from histomicstk.features import compute_nuclei_features
#import seaborn as sns
from skimage.measure import label, regionprops_table
# 初始化模型（nuclei专用）
model = models.CellposeModel(gpu=True)

path = "/Users/hujinzhen/Documents/phd/2026winter/OnSight_Pathology_Revision/image.png"
img = io.imread(path)

print("正在运行分割...")
# 运行分割
masks, _,_ = model.eval(img)
plt.imshow(masks, cmap="nipy_spectral")

print("正在计算特征...")

I_0 = 255
W_estimated = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(img, I_0)
stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
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
#print(W_custom)

# --- 3. 使用专属矩阵进行精准的颜色反卷积 ---
deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(
    img, W_custom, I_0
)

print("正在提取单通道图像...")
# --- 4. 提取出极其精准的单通道图像 ---
im_nuclei_accurate = deconv_result.Stains[:, :, 0]
im_cytoplasm = deconv_result.Stains[:, :, 1]
plt.imshow(im_nuclei_accurate, cmap="gray")
features_df = compute_nuclei_features(
    im_label=masks,
    im_nuclei=im_nuclei_accurate,           # <--- 填入这里
   # im_cytoplasm=im_cytoplasm,     # <--- (可选) 如果你想算细胞质特征，填入这里
    morphometry_features_flag=True,
    fsd_features_flag=False,
    intensity_features_flag=True,
    gradient_features_flag=False,
    haralick_features_flag=False
)


# 1. 定义你指定的特征组
target_columns = [
    # Nuclear Size
    'Size.Area', 'Size.MajorAxisLength', 'Shape.EquivalentDiameter',
    # Nuclear Shape
    'Shape.Circularity', 'Shape.Eccentricity', 'Shape.Solidity', 
    'Shape.Extent', 'Size.Perimeter', 'Shape.FractalDimension',
    # Intensity
    'Nucleus.Intensity.Mean', 'Nucleus.Intensity.Max', 
    'Nucleus.Intensity.Std', 'Nucleus.Intensity.IQR'
]

# 2. 计算均值 (Mean)
# 注意：我们只计算 DataFrame 中实际存在的列，防止报错
existing_cols = [c for c in target_columns if c in features_df.columns]
all_cells_mean = features_df[existing_cols].mean()

# 3. 格式化打印结果
print("\n" + "="*60)
print(f"病理特征统计报告 | 样本路径: ...{path[-30:]}")
print(f"检测到的细胞总数: {len(features_df)}")
print("="*60)

# 分门别类展示，方便病理学解读
categories = {
    "Nuclear Size (细胞核大小/肥大程度)": ['Size.Area', 'Size.MajorAxisLength', 'Shape.EquivalentDiameter', 'Size.Perimeter'],
    "Nuclear Shape (细胞核形状/异型性)": ['Shape.Circularity', 'Shape.Eccentricity', 'Shape.Solidity', 'Shape.Extent', 'Shape.FractalDimension'],
    "Nuclear Intensity (染色强度/深染程度)": ['Nucleus.Intensity.Mean', 'Nucleus.Intensity.Max', 'Nucleus.Intensity.Std', 'Nucleus.Intensity.IQR']
}

for cat_name, cols in categories.items():
    print(f"\n>>> {cat_name}:")
    for col in cols:
        if col in all_cells_mean:
            # 打印列名和对应的均值，保留4位小数
            print(f"    {col:30} : {all_cells_mean[col]:>10.4f}")
        else:
            print(f"    {col:30} : [未计算 - 请检查Flag设置]")

print("\n" + "="*60)