# app.spec
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
    copy_metadata,        
)
entry_script = "app.py"

hiddenimports = (
    collect_submodules("torch")
    + collect_submodules("torchvision")
    + collect_submodules("transformers")
    + collect_submodules("huggingface_hub")
    + collect_submodules("sentencepiece")
    + collect_submodules("ultralytics")
    + collect_submodules("bitsandbytes")   
    + collect_submodules("timm")   
    + collect_submodules("transformers_stream_generator")
    + collect_submodules("retinanet")
    + collect_submodules("object_detection_fastai")   # RetinaNet helper    
    + collect_submodules("fastai") 
     + collect_submodules("fastprogress")

   # +collect_submodules("onnxruntime")
)

datas = (
    collect_data_files("torch", include_py_files=False)
    + collect_data_files("transformers", include_py_files=False)
    + collect_data_files("huggingface_hub", include_py_files=False)
    + collect_data_files("ultralytics", includes=["cfg/*.yaml"], include_py_files=False)
    + [("metadata/*.json", "metadata")]    
    + [("retinanet/file/config.yaml","retinanet/file"),
       ("retinanet/file/statistics_sdata.pickle","retinanet/file")]
    + collect_data_files("retinanet", includes=["**/*.yaml", "**/*.pickle", "**/*.pth", "**/*.pt"])
    + copy_metadata("fastprogress")
    + copy_metadata("transformers_stream_generator")
)

#binaries = collect_dynamic_libs("torch")   # CUDA / MKL etc.
binaries = (
     collect_dynamic_libs("torch")          # CUDA / MKL
   #  + collect_dynamic_libs("onnxruntime")
     + collect_dynamic_libs("bitsandbytes")
 )
a = Analysis(
    [entry_script],
    excludes=["torch.utils.tensorboard"],
    hiddenimports=hiddenimports,
    datas=datas,
    binaries=binaries,
    noarchive=False,
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="app",
    console=True,
    icon="sample_icon.ico",
)


#pyinstaller --clean app.spec 

#pyinstaller --clean --noconfirm app.spec 

#pyinstaller --noconfirm app.spec   # no --clean ⇒ seconds
