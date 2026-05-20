[Setup]

AppName=OnSightPathology
AppVersion=1.0
DefaultDirName={pf}\OnSightPathology
DefaultGroupName=OnSightPathology

; Output installer filename
OutputBaseFilename=OnSightPathologyInstaller
OutputDir=output


DiskSpanning=no


Compression=lzma2/max
SolidCompression=yes

LZMAUseSeparateProcess=yes

SetupIconFile="sample_icon.ico"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "sample_icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\app\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\OnSightPathology"; Filename: "{app}\app.exe"; IconFilename: "{app}\sample_icon.ico"

Name: "{group}\Uninstall OnSightPathology"; Filename: "{uninstallexe}"; IconFilename: "{app}\sample_icon.ico"

; Desktop shortcut
Name: "{commondesktop}\OnSightPathology"; Filename: "{app}\app.exe"; Tasks: desktopicon; IconFilename: "{app}\sample_icon.ico"

[Run]
; Ask whether to launch after installation
Description: "Launch OnSightPathology"; Filename: "{app}\app.exe"; Flags: nowait postinstall skipifsilent
