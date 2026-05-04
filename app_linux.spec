# -*- mode: python ; coding: utf-8 -*-
import os
import glob
from PyInstaller.utils.hooks import (
    collect_data_files,
    copy_metadata,
    collect_all,
    collect_dynamic_libs,
    collect_submodules
)


hook_code = """
import os
import sys


if getattr(sys, 'frozen', False):

    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    

    lib_paths = [
        base_dir,
        os.path.join(base_dir, 'torch', 'lib'),
    ]
    
    nvidia_dir = os.path.join(base_dir, 'nvidia')
    if os.path.exists(nvidia_dir):
        for d in os.listdir(nvidia_dir):
            sub_lib = os.path.join(nvidia_dir, d, 'lib')
            if os.path.exists(sub_lib):
                lib_paths.append(sub_lib)
                
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_paths) + (':' + current_ld if current_ld else '')
"""
with open('rthook_cuda_linux_generated.py', 'w', encoding='utf-8') as f:
    f.write(hook_code)



def safe_collect(pkg):
    try:
        datas, binaries, hiddenimports = collect_all(pkg)
        binaries += collect_dynamic_libs(pkg)
        datas += collect_data_files(pkg)
        hiddenimports += collect_submodules(pkg)
        return datas, binaries, hiddenimports
    except Exception:
        return [], [], []


nvidia_packages = [
    "nvidia", "nvidia.cudnn", "nvidia.cublas", "nvidia.cuda_runtime",
    "nvidia.cuda_nvrtc", "nvidia.cusolver", "nvidia.cusparse",
    "nvidia.curand", "nvidia.cufft", "nvidia.nccl", "nvidia.nvjitlink",
]

nv_datas, nv_binaries, nv_hiddenimports = [], [], []
for pkg in nvidia_packages:
    d, b, h = safe_collect(pkg)
    nv_datas += d; nv_binaries += b; nv_hiddenimports += h

block_cipher = None


torch_datas, torch_binaries, torch_hiddenimports = collect_all("torch")
tv_datas, tv_binaries, tv_hiddenimports = collect_all("torchvision")
np_datas, np_binaries, np_hiddenimports = collect_all("numpy")
cv_datas, cv_binaries, cv_hiddenimports = collect_all("cv2")
skimage_datas, skimage_binaries, skimage_hiddenimports = collect_all("skimage")
lazy_datas, lazy_binaries, lazy_hiddenimports = collect_all("lazy_loader")
tf_datas, tv_binaries_tf, tf_hiddenimports = collect_all("transformers")



a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=
        torch_binaries + tv_binaries + np_binaries + cv_binaries + 
        skimage_binaries + lazy_binaries + tv_binaries_tf + nv_binaries,
    datas=
        [("metadata/*.json", "metadata")]
        + [
            ("retinanet/file/config.yaml", "retinanet/file"),
            ("retinanet/file/statistics_sdata.pickle", "retinanet/file"),
        ]
        + collect_data_files("retinanet", includes=["**/*.yaml", "**/*.pickle", "**/*.pth", "**/*.pt"])
        + copy_metadata("fastprogress")
        + torch_datas + tv_datas + np_datas + cv_datas + 
        skimage_datas + lazy_datas + tf_datas + nv_datas,
    hiddenimports=
        torch_hiddenimports + tv_hiddenimports + nv_hiddenimports + np_hiddenimports + 
        cv_hiddenimports + skimage_hiddenimports + lazy_hiddenimports + tf_hiddenimports
        + [
            "torch", "torchvision", "torchvision.ops", "torchvision.transforms",
            "numpy", "cv2", "PyQt6", "skimage", "lazy_loader",
            "transformers", "qwen_vl_utils", "accelerate", "timm", "PIL", "scipy",
        ],
    hookspath=[],
    hooksconfig={},
    

    runtime_hooks=['rthook_cuda_linux_generated.py'], 

    excludes=["tensorboard"], 
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data)


exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True, 
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False, 
    console=True, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='app'
)

######################################################################################################
# Old Version
######################################################################################################
# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import (
    collect_data_files,
    copy_metadata,
    collect_all
)
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_data_files, collect_submodules

def safe_collect(pkg):
    try:
        datas, binaries, hiddenimports = collect_all(pkg)
        binaries += collect_dynamic_libs(pkg)
        datas += collect_data_files(pkg)
        hiddenimports += collect_submodules(pkg)
        return datas, binaries, hiddenimports
    except Exception:
        return [], [], []

nvidia_packages = [
    "nvidia",
    "nvidia.cudnn",
    "nvidia.cublas",
    "nvidia.cuda_runtime",
    "nvidia.cuda_nvrtc",
    "nvidia.cusolver",
    "nvidia.cusparse",
    "nvidia.curand",
    "nvidia.cufft",
    "nvidia.nccl",
    "nvidia.nvjitlink",
]

nv_datas = []
nv_binaries = []
nv_hiddenimports = []

for pkg in nvidia_packages:
    d, b, h = safe_collect(pkg)
    nv_datas += d
    nv_binaries += b
    nv_hiddenimports += h
block_cipher = None



torch_datas, torch_binaries, torch_hiddenimports = collect_all("torch")
tv_datas, tv_binaries, tv_hiddenimports = collect_all("torchvision")
np_datas, np_binaries, np_hiddenimports = collect_all("numpy")
cv_datas, cv_binaries, cv_hiddenimports = collect_all("cv2")


skimage_datas, skimage_binaries, skimage_hiddenimports = collect_all("skimage")
lazy_datas, lazy_binaries, lazy_hiddenimports = collect_all("lazy_loader")


tf_datas, tv_binaries_tf, tf_hiddenimports = collect_all("transformers")

a = Analysis(
    ['app.py'],


    pathex=['.'],

    
    binaries=
        torch_binaries
        + tv_binaries
        + np_binaries
        + cv_binaries
        + skimage_binaries
        + lazy_binaries  
        + tv_binaries_tf
        + nv_binaries,


    datas=
        [("metadata/*.json", "metadata")]
        + [
            ("retinanet/file/config.yaml", "retinanet/file"),
            ("retinanet/file/statistics_sdata.pickle", "retinanet/file"),
        ]
        + collect_data_files(
            "retinanet",
            includes=["**/*.yaml", "**/*.pickle", "**/*.pth", "**/*.pt"]
        )
        + copy_metadata("fastprogress")
        + torch_datas
        + tv_datas
        + np_datas
        + cv_datas
        + skimage_datas 
        + lazy_datas
        + tf_datas   
        + nv_datas,


    hiddenimports=
        torch_hiddenimports
        + tv_hiddenimports
        + nv_hiddenimports
        + np_hiddenimports
        + cv_hiddenimports
        + skimage_hiddenimports 
        + lazy_hiddenimports   
        + tf_hiddenimports
        + [
            "torch",
            "torchvision",
            "torchvision.ops",
            "torchvision.transforms",
            "numpy",
            "cv2",
            "PyQt6",
            "skimage",     
            "lazy_loader",  
       
            "transformers",
            "qwen_vl_utils",
            "accelerate",
            "timm",
            "PIL",
            "scipy",
        ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    excludes=[
        "tensorboard",
        "bitsandbytes",
    ],

    noarchive=False,
    optimize=0,
)




pyz = PYZ(a.pure, a.zipped_data)


exe = EXE(
    pyz,
    a.scripts,
    [],                 
    exclude_binaries=True, 
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='app'         
)