# OnSight WSI - Local Development and Build Guide

This document describes how to run OnSight WSI locally, how to build the Windows GPU executable from source, and how to package it into an installer.

Build and packaging steps are very similar to the main OnSight Pathology project. This README focuses on the differences that apply to the WSI application in this repository.

---

# System Requirements

- Operating System: Windows
- Python: 3.11.9 (recommended)
- NVIDIA GPU with CUDA support

This project is intended for Windows GPU use only. A CPU-only setup is not documented or supported by this README.

---

# What This App Does

OnSight WSI is a PyQt6 desktop application for:

- Running whole-slide image processing on supported slide/image files
- Generating per-slide output folders with thumbnails, tiles, and CSV outputs
- Viewing generated mitosis and HAVOC overlays interactively

Supported input file types include:

- `.svs`
- `.ndpi`
- `.tif`
- `.tiff`
- `.jpg`
- `.jpeg`
- `.png`

Typical per-slide outputs include:

- `thumbnail.jpg`
- `tiles/`
- `info_df_mitosis.csv`
- `info_df_havoc.csv`

If HAVOC is disabled, the HAVOC CSV is not produced.

---

# Running Locally (Development Mode)

## 1. Create a Virtual Environment

```powershell
python -m venv onsightwsi_env
onsightwsi_env\Scripts\activate
```

Confirm Python version:

```powershell
python --version
```

Recommended: Python 3.11.9

---

## 2. Install Dependencies

This repository includes a single dependency file for the Windows GPU configuration:

```powershell
pip install -r requirements.txt
```

This installs the CUDA-enabled PyTorch build used by the application.

---

## 3. Launch the Application

```powershell
python app.py
```

This launches the PyQt6 desktop application directly from source.

---

# HAVOC Model Access

If you enable the HAVOC pipeline, the application will prompt for Hugging Face access the first time it needs the model.

The app uses the gated `prov-gigapath/prov-gigapath` model and will prompt you to:

1. Accept the model terms on Hugging Face
2. Create a read-only Hugging Face token
3. Verify that token in the application

The token is stored securely using the local keyring.

The initial model download is large, so the first HAVOC-enabled run may take some time and requires internet access.

If you only run the mitosis pipeline, this Hugging Face setup is not required.

---

# Building the Executable (PyInstaller)

Ensure PyInstaller is installed:

```powershell
pip install pyinstaller
```

This repository builds a single executable:

- `app.exe`

Build it with:

```powershell
pyinstaller app.spec --noconfirm
```

---

## Post-Build Directory Structure

After building:

```text
dist/
    app/
```

The packaged application is expected to run from:

```text
dist/app/app.exe
```

Model files, metadata, and bundled runtime dependencies are included under:

```text
dist/app/_internal/
```

---

# Creating a Windows Installer (Inno Setup)

OnSight WSI can be packaged using Inno Setup version 6.5 or newer.

Download:

<https://jrsoftware.org/isdl.php>

The repository includes this installer script:

```text
installer.iss
```

The script packages everything under:

```text
dist/app/
```

To generate the installer:

1. Build the application with PyInstaller
2. Open `installer.iss` in Inno Setup
3. Click **Run**

The compiled installer will be generated in:

```text
output/
```

No additional installer changes should be required for the standard build.

---

# Repository Notes

- `app.py` contains the desktop application and slide-processing workflow
- `app.spec` defines the PyInstaller build, including bundled model/config assets
- `installer.iss` defines the Windows installer package
- `retinanet/` and `havoc/` contain model and pipeline code used by the application

---

# Citation

If you use OnSight Pathology in research, please cite the associated publication listed at:

https://onsightpathology.github.io/