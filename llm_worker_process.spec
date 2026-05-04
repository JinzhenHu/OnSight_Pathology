# llm_worker_process.spec
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata

a = Analysis(
    ['llm_worker_process.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='llm_worker_process',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to True since it's a background process, stdout is used
    disable_windowed_traceback=False,
    onefile=True
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='llm_worker_process',
)
