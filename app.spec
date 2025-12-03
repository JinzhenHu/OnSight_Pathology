# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import (
    collect_data_files,
    copy_metadata      
)


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[("metadata/*.json", "metadata")]
    + [("retinanet/file/config.yaml","retinanet/file"),
       ("retinanet/file/statistics_sdata.pickle","retinanet/file")]
    + collect_data_files("retinanet", includes=["**/*.yaml", "**/*.pickle", "**/*.pth", "**/*.pt"])
    + copy_metadata("fastprogress"),
    hiddenimports=[],
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
