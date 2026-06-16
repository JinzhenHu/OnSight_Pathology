; ============================================================================
; Inno Setup installer for the HF (online-download) build.
;
; Wraps the single-file OnSight_HF.exe into a proper Windows installer:
;   - Installs to Program Files
;   - Adds Start Menu + optional desktop shortcuts
;   - Registers an uninstaller
;
; Use this when you want a polished install experience without bundled models.
; The .exe will download AI models from HuggingFace on first launch.
;
; Build:
;   iscc installer_hf.iss
; ============================================================================

#define AppDisplayName "OnSight Pathology (HF)"
#define ExeName "OnSight_HF.exe"
#define DistDir "dist"

[Setup]
AppName={#AppDisplayName}
AppVersion=1.0
DefaultDirName={autopf}\OnSightPathology
DefaultGroupName=OnSightPathology
OutputBaseFilename=OnSight_Windows_HF
OutputDir=output

; HF build is small (~1-2 GB exe) — no disk spanning needed.
; Keep solid compression on for max shrink, single .exe payload means
; decompression is one continuous stream anyway.
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4

; 64-bit installer.
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible

SetupIconFile="sample_icon.ico"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "sample_icon.ico"; DestDir: "{app}"; Flags: ignoreversion

; The PyInstaller onefile exe is already self-compressed internally, so we
; tell Inno not to compress it again. Avoids 30-60s of extra unpack time
; during install for no real size savings.
Source: "{#DistDir}\{#ExeName}"; DestDir: "{app}"; Flags: ignoreversion nocompression

[Icons]
Name: "{group}\OnSight Pathology";              Filename: "{app}\{#ExeName}"; IconFilename: "{app}\sample_icon.ico"
Name: "{group}\Uninstall OnSight Pathology";    Filename: "{uninstallexe}";   IconFilename: "{app}\sample_icon.ico"
Name: "{commondesktop}\OnSight Pathology";      Filename: "{app}\{#ExeName}"; Tasks: desktopicon; IconFilename: "{app}\sample_icon.ico"

[Run]
Filename: "{app}\{#ExeName}"; Description: "Launch OnSight Pathology"; Flags: nowait postinstall skipifsilent