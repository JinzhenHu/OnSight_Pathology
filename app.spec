# -*- mode: python ; coding: utf-8 -*-
import os as _os
import sys as _sys

# ============================================================================
# BUILD MODE + FORMAT SELECTORS
# ----------------------------------------------------------------------------
# Two independent env vars:
#
#   ONSIGHT_BUILD:
#     "local" (default) → bundle weights into the build
#     "hf"              → no bundled weights, downloads on first use
#
#   ONSIGHT_FORMAT:
#     "onedir"  → folder of files (default for local; Inno Setup wraps it)
#     "onefile" → single .exe    (default for hf; easy to share)
#
# Usage (PowerShell):
#   $env:ONSIGHT_BUILD="local"; pyinstaller app.spec                          # local onedir (Inno)
#   $env:ONSIGHT_BUILD="local"; $env:ONSIGHT_FORMAT="onefile"; pyinstaller app.spec   # local onefile
#   $env:ONSIGHT_BUILD="hf";    pyinstaller app.spec                          # hf onefile
# ============================================================================
_ONSIGHT_BUILD = _os.environ.get("ONSIGHT_BUILD", "local").strip().lower()
_BUNDLE_MODELS = (_ONSIGHT_BUILD == "local")

_ONSIGHT_FORMAT = _os.environ.get(
    "ONSIGHT_FORMAT",
    "onedir" if _BUNDLE_MODELS else "onefile"
).strip().lower()
_ONEFILE = (_ONSIGHT_FORMAT == "onefile")

_banner = "=" * 60
print(
    f"\n{_banner}\n"
    f"OnSight build mode: "
    f"{'LOCAL + HF (bundled weights)' if _BUNDLE_MODELS else 'HF-ONLY (no bundled weights)'}"
    f"\n              format: {'onefile (single .exe)' if _ONEFILE else 'onedir (folder)'}"
    f"\n{_banner}\n",
    file=_sys.stderr,
)

from PyInstaller.utils.hooks import (
    collect_data_files,
    copy_metadata,
    collect_all,
)

block_cipher = None


# ============================================================================
# Recursive metadata collection helper
# ============================================================================
def _meta_recursive(pkg, _seen=None):
    if _seen is None:
        _seen = set()

    import importlib.metadata as ilm

    norm = pkg.lower().replace("_", "-").split(";")[0].split("[")[0].strip()
    if norm in _seen:
        return []
    _seen.add(norm)

    out = []
    try:
        out.extend(copy_metadata(pkg))
    except Exception:
        return out

    try:
        requires = ilm.distribution(pkg).requires or []
    except Exception:
        return out

    for req in requires:
        dep = req.split(";")[0].split()[0]
        for op in (">=", "==", "<=", "<", ">", "~=", "!=", "["):
            dep = dep.split(op)[0]
        dep = dep.strip()
        if dep:
            out.extend(_meta_recursive(dep, _seen))

    return out


# ----------------------------------------------------------------------------
# Heavy dependencies
# ----------------------------------------------------------------------------
torch_datas,   torch_binaries,   torch_hiddenimports   = collect_all("torch")
tv_datas,      tv_binaries,      tv_hiddenimports      = collect_all("torchvision")
np_datas,      np_binaries,      np_hiddenimports      = collect_all("numpy")
cv_datas,      cv_binaries,      cv_hiddenimports      = collect_all("cv2")
skimage_datas, skimage_binaries, skimage_hiddenimports = collect_all("skimage")
lazy_datas,    lazy_binaries,    lazy_hiddenimports    = collect_all("lazy_loader")
tf_datas,      tf_binaries,      tf_hiddenimports      = collect_all("transformers")
hf_datas,      hf_binaries,      hf_hiddenimports      = collect_all("huggingface_hub")
cp_datas,      cp_binaries,      cp_hiddenimports      = collect_all("cellpose")
fr_datas,      fr_binaries,      fr_hiddenimports      = collect_all("fastremap")
fv_datas,      fv_binaries,      fv_hiddenimports      = collect_all("fill_voids")
ultra_datas,   ultra_binaries,   ultra_hiddenimports   = collect_all("ultralytics")
timm_datas,    timm_binaries,    timm_hiddenimports    = collect_all("timm")
accel_datas,   accel_binaries,   accel_hiddenimports   = collect_all("accelerate")
bnb_datas,     bnb_binaries,     bnb_hiddenimports     = collect_all("bitsandbytes")

a = Analysis(
    ['app.py'],
    pathex=['.'],

    binaries=(
        torch_binaries
        + tv_binaries
        + np_binaries
        + cv_binaries
        + skimage_binaries
        + lazy_binaries
        + tf_binaries
        + hf_binaries
        + cp_binaries
        + fr_binaries
        + fv_binaries
        + ultra_binaries
        + timm_binaries
        + accel_binaries
        + bnb_binaries
    ),

    datas=(
        [("metadata/*.json", "metadata")]
        + [("local_models/*", "local_models")]
        + ([("bundled_models/*", "bundled_models")] if _BUNDLE_MODELS else [])
        + [
            ("retinanet/file/config.yaml", "retinanet/file"),
            ("retinanet/file/statistics_sdata.pickle", "retinanet/file"),
        ]
        + collect_data_files(
            "retinanet",
            includes=["**/*.yaml", "**/*.pickle", "**/*.pth", "**/*.pt"],
        )
        + _meta_recursive("fastprogress")
        + _meta_recursive("fastai")
        + _meta_recursive("huggingface_hub")
        + torch_datas
        + tv_datas
        + np_datas
        + cv_datas
        + skimage_datas
        + lazy_datas
        + tf_datas
        + hf_datas
        + cp_datas
        + fr_datas
        + fv_datas
        + ultra_datas
        + timm_datas
        + accel_datas
        + bnb_datas
    ),

    hiddenimports=(
        torch_hiddenimports
        + tv_hiddenimports
        + np_hiddenimports
        + cv_hiddenimports
        + skimage_hiddenimports
        + lazy_hiddenimports
        + tf_hiddenimports
        + hf_hiddenimports
        + [
            # Core ML stack
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
            "transformers.utils.logging",
            "qwen_vl_utils",
            "accelerate",
            "bitsandbytes",
            "timm",
            "PIL",
            "scipy",
            "huggingface_hub",
            "huggingface_hub.file_download",
            "huggingface_hub.utils",
            "huggingface_hub.utils.tqdm",
            "huggingface_hub.utils._http",
            "tqdm",
            "tqdm.auto",
            "tqdm.std",

            # Project modules — top-level
            "settings",
            "utils",
            "utils_clustering",
            "crash_logging",
            "model_loader_thread",
            "region_selector",

            # Project modules — custom widgets
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
            "custom_widgets.DpiStatusIndicator",
        ]
        + cp_hiddenimports
        + fr_hiddenimports
        + fv_hiddenimports
        + ultra_hiddenimports
        + timm_hiddenimports
        + accel_hiddenimports
        + bnb_hiddenimports
    ),

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    excludes=[
        "tensorboard",
        # Dask / distributed — pulled by histomicstk, unused
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
        # Heavy / unused
        "torchaudio",
        "pyarrow",
        "av",
    ],

    noarchive=False,
    optimize=0,
)


pyz = PYZ(a.pure, a.zipped_data)


# ============================================================================
# OUTPUT — branch on _ONEFILE (independent of _BUNDLE_MODELS)
# ----------------------------------------------------------------------------
# Output names keep the build mode in the filename so the two artifacts
# don't collide. Inno Setup hooks onto the onedir local build by name.
# ============================================================================
_OUTPUT_NAME = "app" if _BUNDLE_MODELS else "OnSight_HF"
_COLL_NAME   = "app_local" if _BUNDLE_MODELS else "OnSight_HF_dir"

if _ONEFILE:
    # Single self-extracting .exe.
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name=_OUTPUT_NAME,
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

else:
    # Folder of files. For LOCAL mode, Inno Setup wraps this into an installer.
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=_OUTPUT_NAME,
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
        name=_COLL_NAME,
    )


# ============================================================================
# Build commands cheatsheet 
# ----------------------------------------------------------------------------
# LOCAL + HF, onedir (default - for Inno Setup wrapping):
#   set ONSIGHT_BUILD=local && set ONSIGHT_FORMAT=onedir && pyinstaller app.spec 
#
# LOCAL + HF, onefile (single .exe with bundled weights - very large):
#   set ONSIGHT_BUILD=local && set ONSIGHT_FORMAT=onefile && pyinstaller app.spec
#
# HF-only, onefile (default - easy to email/share):
#   set ONSIGHT_BUILD=hf && set ONSIGHT_FORMAT=onefile && pyinstaller app.spec
#
# HF-only, onedir (but supported):
#   set ONSIGHT_BUILD=hf && set ONSIGHT_FORMAT=onedir && pyinstaller app.spec
# ============================================================================