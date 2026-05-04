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
        + tv_binaries_tf,


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
        + tf_datas,   


    hiddenimports=
        torch_hiddenimports
        + tv_hiddenimports
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
    name='app',

    debug=False,
    bootloader_ignore_signals=False,

    strip=False,

    upx=False,

    upx_exclude=[],

    runtime_tmpdir=None,

    console=True,

    disable_windowed_traceback=False,

    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
