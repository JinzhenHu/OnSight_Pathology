; ============================================================================
; Build mode controlled by ISCC command-line define:
;   iscc installer.iss /DBuildMode=local   -> Local+HF installer
;   iscc installer.iss /DBuildMode=hf      -> HF-only installer
;   (default: local)
; ============================================================================
#ifndef BuildMode
  #define BuildMode "local"
#endif

#if BuildMode == "local"
  #define DistDir "dist\app_local"
  #define InstallerSuffix "Local"
  #define AppDisplayName "OnSight Pathology (Bundled Models)"
#else
  #define DistDir "dist\app_hf"
  #define InstallerSuffix "Online"
  #define AppDisplayName "OnSight Pathology (Online Download)"
#endif

[Setup]
AppName={#AppDisplayName}
AppVersion=1.0
DefaultDirName={autopf}\OnSightPathology
DefaultGroupName=OnSightPathology
OutputBaseFilename=OnSightPathologyInstaller_{#InstallerSuffix}
OutputDir=output

; Large-file support — split into ~2 GB chunks for installers over 4 GB.
DiskSpanning=yes
DiskSliceSize=2100000000
SlicesPerDisk=1
ReserveBytes=0

; Compression — multi-threaded LZMA2 cuts both compress and decompress time.
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4

; 64-bit installer (needed when total payload exceeds 2 GB).
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible

SetupIconFile="sample_icon.ico"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "sample_icon.ico"; DestDir: "{app}"; Flags: ignoreversion

; --- Already-compressed or near-incompressible binaries ---
; nocompression: don't LZMA-pack already-dense files (saves install time).
; skipifsourcedoesntexist: don't fail compile if a pattern matches nothing.
Source: "{#DistDir}\*.pth";         DestDir: "{app}"; Flags: ignoreversion recursesubdirs nocompression skipifsourcedoesntexist
Source: "{#DistDir}\*.safetensors"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs nocompression skipifsourcedoesntexist
Source: "{#DistDir}\*.bin";         DestDir: "{app}"; Flags: ignoreversion recursesubdirs nocompression skipifsourcedoesntexist
Source: "{#DistDir}\*.dll";         DestDir: "{app}"; Flags: ignoreversion recursesubdirs nocompression skipifsourcedoesntexist
Source: "{#DistDir}\*.pyd";         DestDir: "{app}"; Flags: ignoreversion recursesubdirs nocompression skipifsourcedoesntexist
Source: "{#DistDir}\*.so";          DestDir: "{app}"; Flags: ignoreversion recursesubdirs nocompression skipifsourcedoesntexist
Source: "{#DistDir}\*.lib";         DestDir: "{app}"; Flags: ignoreversion recursesubdirs nocompression skipifsourcedoesntexist

; cellpose-SAM weights (no extension — just plain "cpsam")
Source: "{#DistDir}\bundled_models\cpsam"; DestDir: "{app}\bundled_models"; Flags: ignoreversion nocompression skipifsourcedoesntexist

; --- Everything else (compressible: .py, .pyc, .json, .yaml, .txt) ---
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\OnSightPathology"; Filename: "{app}\app.exe"; IconFilename: "{app}\sample_icon.ico"
Name: "{group}\Uninstall OnSightPathology"; Filename: "{uninstallexe}"; IconFilename: "{app}\sample_icon.ico"
Name: "{commondesktop}\OnSightPathology"; Filename: "{app}\app.exe"; Tasks: desktopicon; IconFilename: "{app}\sample_icon.ico"

[Run]
Filename: "{app}\app.exe"; Description: "Launch OnSightPathology"; Flags: nowait postinstall skipifsilent