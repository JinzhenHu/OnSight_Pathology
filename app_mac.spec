import os
import sys
import torch
from PyInstaller.utils.hooks import collect_data_files, copy_metadata, collect_all

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

if sys.stdout is None or not hasattr(sys.stdout, 'fileno'):
    log_file = os.path.expanduser("~/Desktop/onsight_debug.log")
    sys.stdout = open(log_file, 'w', buffering=1)
    sys.stderr = sys.stdout

torch_path = os.path.dirname(torch.__file__)
torch_subfolders = ["bin", "lib", ".libs"]
extra_torch_datas = []
hf_datas, hf_binaries, hf_hiddenimports = collect_all("huggingface_hub")
timm_datas, timm_binaries, timm_hiddenimports = collect_all("timm")
trans_datas, trans_binaries, trans_hiddenimports = collect_all("transformers")

for folder in torch_subfolders:
    full_path = os.path.join(torch_path, folder)
    if os.path.exists(full_path):
        extra_torch_datas.append((full_path, f"torch/{folder}"))

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ("metadata/*.json", "metadata"),
        ("retinanet/file/config.yaml","retinanet/file"),
        ("retinanet/file/statistics_sdata.pickle","retinanet/file"),
    ] 
    + extra_torch_datas
    + collect_data_files("retinanet", includes=["**/*.yaml", "**/*.pickle", "**/*.pth", "**/*.pt"])
    + copy_metadata("fastprogress")
    + copy_metadata("torch")
    + copy_metadata("timm"),
    hiddenimports=[
        'sklearn.utils._cython_blas',
        'torch', 'torchvision', 'fastai', 'fastai.callbacks', 'timm'
    ],
    excludes=['notebook', 'tensorboard'], 
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OnSight_App', 
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,            
    console=False,         
    disable_windowed_traceback=False,
    argv_emulation=False,
    # target_arch='arm64', 
)

coll = COLLECT(
    exe, a.binaries, a.datas, name='OnSight_App',
)

app = BUNDLE(
    coll, name='OnSight_App.app', bundle_identifier='com.onsight.pathology',
)
#pyinstaller --noconfirm app_mac.spec

#rm -rf build dist