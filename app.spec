# -*- mode: python ; coding: utf-8 -*-

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
            "settings",
            "utils",
            "utils_clustering",
            "crash_logging",
        ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    excludes=[
        "tensorboard",
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='sample_icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='app',
)

#python -m py_compile app.spec