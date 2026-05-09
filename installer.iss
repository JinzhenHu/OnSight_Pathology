[Setup]
; --- Basic Information ---
AppName=OnSightPathology
AppVersion=1.0
DefaultDirName={pf}\OnSightPathology
DefaultGroupName=OnSightPathology

; Output installer filename
OutputBaseFilename=OnSightPathologyInstaller
OutputDir=output

; --- Packaging Settings ---
; Disable disk spanning to package everything into one EXE
DiskSpanning=no

; Enable maximum compression
Compression=lzma2/max
SolidCompression=yes

; Use separate process for faster compression
LZMAUseSeparateProcess=yes

; --- Icon Settings ---
; Make sure this icon file exists in the script directory
SetupIconFile="sample_icon.ico"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
; Ensure the source path points to the actual application directory
Source: "sample_icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\app\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
; Start Menu shortcut
Name: "{group}\OnSightPathology"; Filename: "{app}\app.exe"; IconFilename: "{app}\sample_icon.ico"

; Uninstall shortcut
Name: "{group}\Uninstall OnSightPathology"; Filename: "{uninstallexe}"; IconFilename: "{app}\sample_icon.ico"

; Desktop shortcut
Name: "{commondesktop}\OnSightPathology"; Filename: "{app}\app.exe"; Tasks: desktopicon; IconFilename: "{app}\sample_icon.ico"

[Run]
; Ask whether to launch after installation
Description: "Launch OnSightPathology"; Filename: "{app}\app.exe"; Flags: nowait postinstall skipifsilent