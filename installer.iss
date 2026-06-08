; ============================================================================
; Build mode controlled by ISCC command-line define:
;   iscc installer.iss /DBuildMode=local   → Local+HF installer
;   iscc installer.iss /DBuildMode=hf      → HF-only installer
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
DefaultDirName={pf}\OnSightPathology
DefaultGroupName=OnSightPathology

OutputBaseFilename=OnSightPathologyInstaller_{#InstallerSuffix}
OutputDir=output
DiskSpanning=no
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
SetupIconFile="sample_icon.ico"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "sample_icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\OnSightPathology"; Filename: "{app}\app.exe"; IconFilename: "{app}\sample_icon.ico"
Name: "{group}\Uninstall OnSightPathology"; Filename: "{uninstallexe}"; IconFilename: "{app}\sample_icon.ico"
Name: "{commondesktop}\OnSightPathology"; Filename: "{app}\app.exe"; Tasks: desktopicon; IconFilename: "{app}\sample_icon.ico"

[Run]
Filename: "{app}\app.exe"; Description: "Launch OnSightPathology"; Flags: nowait postinstall skipifsilent