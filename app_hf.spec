# -*- mode: python ; coding: utf-8 -*-
import os as _os
import sys as _sys

# ============================================================================
# BUILD MODE SELECTOR
# ----------------------------------------------------------------------------
# Set environment variable ONSIGHT_BUILD before running pyinstaller:
#   default / "local"   → bundle weights into the installer (LOCAL+HF mode)
#   "hf"                → no bundled weights, all models download from HF
#
# Usage:
#   PowerShell:
#     $env:ONSIGHT_BUILD="local"; pyinstaller app.spec   # bundled
#     $env:ONSIGHT_BUILD="hf";    pyinstaller app.spec   # HF-only
#
#   CMD:
#     set ONSIGHT_BUILD=local && pyinstaller app.spec
#     set ONSIGHT_BUILD=hf    && pyinstaller app.spec
# ============================================================================
_ONSIGHT_BUILD = _os.environ.get("ONSIGHT_BUILD", "local").lower()
_BUNDLE_MODELS = (_ONSIGHT_BUILD == "local")

_banner = "=" * 60
print(f"\n{_banner}\nOnSight build mode: "
      f"{'LOCAL + HF (bundling weights)' if _BUNDLE_MODELS else 'HF-ONLY (no bundled weights)'}"
      f"\n{_banner}\n", file=_sys.stderr)

from PyInstaller.utils.hooks import (
    collect_data_files,
    copy_metadata,
    collect_all
)

block_cipher = None



torch_datas, torch_binaries, torch_hiddenimports = collect_all("torch")
tv_datas, tv_binaries, tv_hiddenimports = collect_all("torchvision")
np_datas, np_binaries, np_hiddenimports = collect_all("numpy")
cv_datas, cv_binaries, cv_hiddenimports = collect_all("cv2")
skimage_datas, skimage_binaries, skimage_hiddenimports = collect_all("skimage")
lazy_datas, lazy_binaries, lazy_hiddenimports = collect_all("lazy_loader")
tf_datas, tv_binaries_tf, tf_hiddenimports = collect_all("transformers")
hf_datas, hf_binaries, hf_hiddenimports = collect_all("huggingface_hub")

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
        + hf_binaries,

    datas=
        [("metadata/*.json", "metadata")]
        + [("local_models/*", "local_models")]
        + ([("bundled_models/*", "bundled_models")] if _BUNDLE_MODELS else [])
        + [
            ("retinanet/file/config.yaml", "retinanet/file"),
            ("retinanet/file/statistics_sdata.pickle", "retinanet/file"),
        ]
        + collect_data_files(
            "retinanet",
            includes=["**/*.yaml", "**/*.pickle", "**/*.pth", "**/*.pt"]
        )
        + copy_metadata("fastprogress")
        + copy_metadata("huggingface_hub")
        + torch_datas
        + tv_datas
        + np_datas
        + cv_datas
        + skimage_datas 
        + lazy_datas
        + tf_datas
        + hf_datas,

    hiddenimports=
        torch_hiddenimports
        + tv_hiddenimports
        + np_hiddenimports
        + cv_hiddenimports
        + skimage_hiddenimports 
        + lazy_hiddenimports   
        + tf_hiddenimports
        + hf_hiddenimports
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
            "huggingface_hub",
            "huggingface_hub.file_download",
            "huggingface_hub.utils",
            "huggingface_hub.utils.tqdm",
            "huggingface_hub.utils._http",
            "model_loader_thread",
            "custom_widgets",
            "custom_widgets.LoadingDialog",
            "custom_widgets.AboutDialog",
            "custom_widgets.LLMChatDialog",
            "custom_widgets.DisclaimerDialog",
            "custom_widgets.ResizeImageDialog",
            "custom_widgets.mag_detector_widget",
            "custom_widgets.overlay_widget",
            "custom_widgets.overlay_widget_attention",
            "custom_widgets.cascade_widget",
            "custom_widgets.CheckableComboBox",
            "custom_widgets.CollapsibleGroupBox",
            "custom_widgets.PulsingDot",
            "custom_widgets.WelcomeDialog",
            "custom_widgets.SpinnerDialog",
            "custom_widgets.DpiWarningDialog",
            "settings",
            "region_selector",
            "utils",
            "utils_clustering",
            "crash_logging",
        ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    excludes=[
        "tensorboard",
            "dask",
            "dask.array",
            "dask.bag",
            "dask.dataframe",
            "dask.dataframe.tseries",
            "dask.dataframe.io",
            "dask.dataframe.io.parquet",
            "dask.dataframe.io.orc",
            "dask.bytes",
            "dask.widgets",
            "dask_expr",
            "distributed",
            "distributed.dashboard",
            "distributed.shuffle",
            "distributed.protocol",
            "torchaudio",
    ],
    noarchive=False,
    optimize=0,
)


pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,          
    a.zipfiles,         
    a.datas,              
    [],
    name='OnSight_HF',                         
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,                       
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='sample_icon.ico',
)

#python -m py_compile app.spec

#打 Local + HF 版（默认，带本地权重）
#:: 清理旧产物
#rmdir /s /q build
#rmdir /s /q dist

#:: 设环境变量并打包
#set ONSIGHT_BUILD=local
#pyinstaller app.spec

#:: 编 installer
#"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DBuildMode=local installer.iss

#:: 用完清环境变量
#set ONSIGHT_BUILD=
#set "ONSIGHT_BUILD=local"&& pyinstaller app.spec
#打纯 HF 版（不带本地权重）
#:: 清理旧产物
#rmdir /s /q build
#rmdir /s /q dist

#:: 设环境变量并打包
#set ONSIGHT_BUILD=hf
#pyinstaller app.spec

#:: 编 installer
#"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DBuildMode=hf installer.iss

#:: 清环境变量
#set ONSIGHT_BUILD=
#set "ONSIGHT_BUILD=hf"&& pyinstaller app_hf.spec


# Cellpose 模型缓存
#open ~/.cellpose/models

# HuggingFace 缓存（你的模型大部分在这）
#open ~/.cache/huggingface

# OnSight 的 HF 缓存（如果你 spec 设了 HF_HOME）
#open ~/Library/Application\ Support/OnSightPathology

# OnSight 的 settings.json
#open ~/Library/Application\ Support/OnSightPathology/Settings