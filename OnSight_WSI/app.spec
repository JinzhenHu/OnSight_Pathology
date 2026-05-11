# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import (
    collect_data_files,
    copy_metadata,
    collect_dynamic_libs
)

openslide_binaries = collect_dynamic_libs("openslide_bin", destdir="openslide_bin")

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=openslide_binaries,
    datas=[("retinanet/file/config.yaml","retinanet/file"),
       ("retinanet/file/statistics_sdata.pickle","retinanet/file")]
    + collect_data_files("retinanet", includes=["**/*.yaml", "**/*.pickle", "**/*.pth", "**/*.pt"])
    + copy_metadata("fastprogress", recursive=True)
    + copy_metadata("uvicorn")
    + copy_metadata("watchfiles")
    + copy_metadata("httptools")
    + copy_metadata("websockets")
    + copy_metadata("python-dotenv")
    + copy_metadata("PyYAML")
    + copy_metadata("colorama"),   # Windows
    hiddenimports=["openslide_bin"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='app',
)
