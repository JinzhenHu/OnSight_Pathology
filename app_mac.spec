# -*- mode: python ; coding: utf-8 -*-
# OnSight Pathology — macOS PyInstaller spec
import os as _os
import sys as _sys
import torch

# ============================================================================
# BUILD MODE SELECTOR
# ----------------------------------------------------------------------------
# Set ONSIGHT_BUILD before running pyinstaller:
#   "local" (default) → bundle weights into the .app    (LOCAL + HF mode)
#   "hf"              → no bundled weights, HF download (HF-ONLY mode)
#
# Usage (zsh / bash):
#   ONSIGHT_BUILD=local pyinstaller --noconfirm app_mac.spec
#   ONSIGHT_BUILD=hf    pyinstaller --noconfirm app_mac.spec
# ============================================================================
_ONSIGHT_BUILD = _os.environ.get("ONSIGHT_BUILD", "local").lower()
_BUNDLE_MODELS = (_ONSIGHT_BUILD == "local")

_banner = "=" * 60
print(
    f"\n{_banner}\n"
    f"OnSight (macOS) build mode: "
    f"{'LOCAL + HF (bundling weights)' if _BUNDLE_MODELS else 'HF-ONLY (no bundled weights)'}\n"
    f"{_banner}\n",
    file=_sys.stderr,
)

# ----------------------------------------------------------------------------
# macOS runtime workarounds
# ----------------------------------------------------------------------------
# OpenMP / MKL ship multiple copies via numpy + torch on Apple Silicon — letting
# both load avoids the "OMP: Error #15" abort on app startup.
_os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Match what the app's runtime sets so HF downloads land in the same cache
# whether the user launched the .app or ran from source.
_os.environ.setdefault("HF_HOME", _os.path.expanduser("~/.cache/huggingface"))

# Capture PyInstaller's own stdout/stderr to a desktop log when running headless.
if _sys.stdout is None or not hasattr(_sys.stdout, "fileno"):
    _log_file = _os.path.expanduser("~/Desktop/onsight_debug.log")
    _sys.stdout = open(_log_file, "w", buffering=1)
    _sys.stderr = _sys.stdout

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    copy_metadata,
)

block_cipher = None

# ----------------------------------------------------------------------------
# Heavy dependencies: collect_all gathers binaries, data files, and hidden
# imports in one shot. Mirrors the Windows spec so behaviour is consistent.
# ----------------------------------------------------------------------------
torch_datas,    torch_binaries,    torch_hiddenimports    = collect_all("torch")
tv_datas,       tv_binaries,       tv_hiddenimports       = collect_all("torchvision")
np_datas,       np_binaries,       np_hiddenimports       = collect_all("numpy")
cv_datas,       cv_binaries,       cv_hiddenimports       = collect_all("cv2")
skimage_datas,  skimage_binaries,  skimage_hiddenimports  = collect_all("skimage")
lazy_datas,     lazy_binaries,     lazy_hiddenimports     = collect_all("lazy_loader")
tf_datas,       tf_binaries,       tf_hiddenimports       = collect_all("transformers")
hf_datas,       hf_binaries,       hf_hiddenimports       = collect_all("huggingface_hub")
timm_datas,     timm_binaries,     timm_hiddenimports     = collect_all("timm")

# torch's bin / lib / .libs are not always picked up automatically on macOS.
torch_path = _os.path.dirname(torch.__file__)
extra_torch_datas = []
for folder in ("bin", "lib", ".libs"):
    full_path = _os.path.join(torch_path, folder)
    if _os.path.exists(full_path):
        extra_torch_datas.append((full_path, f"torch/{folder}"))

# ----------------------------------------------------------------------------
# Project data files. Each is wrapped with an existence check so the spec
# stays usable on dev machines that don't carry every optional folder.
# ----------------------------------------------------------------------------
def _opt(src_glob, dst):
    """Include src_glob → dst only if at least one matching path exists."""
    import glob
    return [(src_glob, dst)] if glob.glob(src_glob) else []

project_datas = (
    _opt("metadata/*.json", "metadata")
    + _opt("local_models/*", "local_models")
    + (_opt("bundled_models/*", "bundled_models") if _BUNDLE_MODELS else [])
    + [
        ("retinanet/file/config.yaml",            "retinanet/file"),
        ("retinanet/file/statistics_sdata.pickle", "retinanet/file"),
    ]
    + collect_data_files(
        "retinanet",
        includes=["**/*.yaml", "**/*.pickle", "**/*.pth", "**/*.pt"],
    )
    + copy_metadata("fastprogress")
    + copy_metadata("huggingface_hub")
    + copy_metadata("torch")
    + copy_metadata("timm")
)

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
        + timm_binaries
    ),

    datas=(
        project_datas
        + extra_torch_datas
        + torch_datas
        + tv_datas
        + np_datas
        + cv_datas
        + skimage_datas
        + lazy_datas
        + tf_datas
        + hf_datas
        + timm_datas
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
        + timm_hiddenimports
        + [
            # Core ML stack
            "torch",
            "torchvision",
            "torchvision.ops",
            "torchvision.transforms",
            "numpy",
            "cv2",
            "skimage",
            "lazy_loader",
            "sklearn.utils._cython_blas",
            "scipy",
            "PIL",

            # LLM stack
            "transformers",
            "transformers.utils.logging", 
            "qwen_vl_utils",
            "accelerate",
            "timm",
            "huggingface_hub",
            "huggingface_hub.file_download",
            "huggingface_hub.utils",
            "huggingface_hub.utils.tqdm",
            "huggingface_hub.utils._http",
            "tqdm",                              
            "tqdm.auto",                         
            "tqdm.std",

            # GUI
            "PyQt6",

            # macOS permissions
            "ApplicationServices",
            "objc",
            "Quartz",

            # Project modules — top-level
            "settings",
            "utils",
            "utils_clustering",
            "crash_logging",
            "device_compat",
            "model_loader_thread",
            "llm_manager",
            "llm_worker_process",

            # Project modules — custom_widgets
            "custom_widgets",
            "custom_widgets.AboutDialog",
            "custom_widgets.cascade_widget",
            "custom_widgets.CheckableComboBox",
            "custom_widgets.CollapsibleGroupBox",
            "custom_widgets.DisclaimerDialog",
            "custom_widgets.LLMChatDialog",
            "custom_widgets.LoadingDialog",
            "custom_widgets.MacPermissionDialog",
            "custom_widgets.mag_detector_widget",
            "custom_widgets.overlay_widget",
            "custom_widgets.overlay_widget_attention",
            "custom_widgets.PulsingDot",
            "custom_widgets.ResizeImageDialog",
            "custom_widgets.WelcomeDialog",
            "custom_widgets.SpinnerDialog",
            "custom_widgets.DpiWarningDialog",
            "region_selector",
            "custom_widgets.DpiStatusIndicator",

            'AppKit',
            'Quartz',
            'objc',
            'PyObjCTools',

        ]
    ),

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    excludes=[
        "tensorboard",
        "notebook",
        # bitsandbytes does not run on Apple Silicon; excluding it keeps the
        # bundle smaller and avoids dlopen errors at startup.
        "bitsandbytes",
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
    name='OnSightPathology_App',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,             # macOS .app — no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64',          # follows the Python you build with;
                               # set 'arm64' for Apple Silicon only,
                               # or 'universal2' for fat binaries
    codesign_identity='-',    # set this once you have a Developer ID
    entitlements_file='onsight.entitlements',    # add an entitlements.plist for notarization
    icon=None,   # macOS expects .icns, not .ico
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=('OnSightPathology_App_local' if _BUNDLE_MODELS else 'OnSightPathology_App_hf'),
)

# ----------------------------------------------------------------------------
# macOS .app bundle.
# info_plist entries are what the OS shows in the permission prompts the
# first time the app asks for Screen Recording / Accessibility / Camera.
# ----------------------------------------------------------------------------
app = BUNDLE(
    coll,
    name='OnSightPathology_App.app',
    bundle_identifier='com.onsight.pathology',
    info_plist={
        'CFBundleName': 'OnSight',
        'CFBundleDisplayName': 'OnSight Pathology',
        'CFBundleShortVersionString': '1.0.1',
        'CFBundleVersion': '1.0.1',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '12.0',
        # User-facing strings shown in macOS permission prompts.
        'NSScreenCaptureUsageDescription':
            "OnSight needs to record the screen to capture the microscope view "
            "for AI analysis.",
        'NSAppleEventsUsageDescription':
            "OnSight uses Apple Events to open the system Settings panes when "
            "guiding you through granting permissions.",
        # OnSight uses pynput to detect mouse clicks for region selection,
        # which goes through the Accessibility API on macOS.
        'NSAccessibilityUsageDescription':
            "OnSight needs Accessibility access to detect mouse clicks when "
            "you select a screen region for analysis.",

        'NSInputMonitoringUsageDescription':
        "OnSight monitors mouse clicks system-wide so you can select a screen "
        "region by clicking outside the OnSight window (e.g. on your microscope viewer).",
        'LSApplicationCategoryType': 'public.app-category.medical',
        'NSRequiresAquaSystemAppearance': False,  
    },
)

# ----------------------------------------------------------------------------
# Build commands
# ----------------------------------------------------------------------------
# Clean previous output:
#   rm -rf build dist
#
# Local + HF build (default — bundles weights):
#   ONSIGHT_BUILD=local pyinstaller --noconfirm app_mac.spec
#
# HF-only build (no bundled weights):
#   ONSIGHT_BUILD=hf    pyinstaller --noconfirm app_mac.spec
#
# Result lives in:
#   dist/OnSightPathology_App.app
# Cellpose 
#open ~/.cellpose/models

# HuggingFace 
#open ~/.cache/huggingface

# OnSight 的 HF 缓存
#open ~/Library/Application\ Support/OnSightPathology

# OnSight 的 settings.json
#open ~/Library/Application\ Support/OnSightPathology/Settings

#chmod +x build_mac.sh
#./build_mac.sh
#codesign --display --verbose=4 dist/OnSightPathology_App.app 2>&1 | head -20

#open /Users/hujinzhen/OnSightPathology