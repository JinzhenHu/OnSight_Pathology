# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import (
    collect_data_files,
    copy_metadata,
    collect_all
)

block_cipher = None


# -------- 强制收集关键大库 --------
torch_datas, torch_binaries, torch_hiddenimports = collect_all("torch")
tv_datas, tv_binaries, tv_hiddenimports = collect_all("torchvision")
np_datas, np_binaries, np_hiddenimports = collect_all("numpy")
cv_datas, cv_binaries, cv_hiddenimports = collect_all("cv2")

# [新增] 强制收集 skimage 和 lazy_loader 解决 .pyi 缺失报错
skimage_datas, skimage_binaries, skimage_hiddenimports = collect_all("skimage")
lazy_datas, lazy_binaries, lazy_hiddenimports = collect_all("lazy_loader")

# 🚀 [新增 LLM 相关] 强制收集 transformers 等大模型核心库，防止子进程找不到模块
tf_datas, tv_binaries_tf, tf_hiddenimports = collect_all("transformers")

a = Analysis(
    ['app.py'],

    # 当前目录加入搜索路径
    pathex=['.'],

    # -------- 二进制文件 (DLL / SO) --------
    binaries=
        torch_binaries
        + tv_binaries
        + np_binaries
        + cv_binaries
        + skimage_binaries # [新增]
        + lazy_binaries   # [新增]
        + tv_binaries_tf,

    # -------- 数据文件 --------
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
        + skimage_datas # [新增]
        + lazy_datas
        + tf_datas,   # [新增]

    # -------- 隐式导入 --------
    hiddenimports=
        torch_hiddenimports
        + tv_hiddenimports
        + np_hiddenimports
        + cv_hiddenimports
        + skimage_hiddenimports # [新增]
        + lazy_hiddenimports    # [新增]
        + tf_hiddenimports
        + [
            "torch",
            "torchvision",
            "torchvision.ops",
            "torchvision.transforms",
            "numpy",
            "cv2",
            "PyQt6",
            "skimage",      # [新增]
            "lazy_loader",  # [新增]
            # 🚀 [新增 LLM 相关] 大模型自调用所需的隐式依赖
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


# -------- 单文件 EXE --------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',

    debug=False,
    bootloader_ignore_signals=False,

    strip=False,

    # CUDA / torch 必须关闭
    upx=False,

    upx_exclude=[],

    runtime_tmpdir=None,

    # 建议测试阶段开启
    console=True,

    disable_windowed_traceback=False,

    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# -------- 新增 COLLECT 块：将所有文件收集到一个文件夹中 --------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='app',         # 生成的文件夹名称
)

#rmdir /s /q build
#rmdir /s /q dist