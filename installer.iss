[Setup]
AppName=OnSightPathology
AppVersion=1.0
DefaultDirName={pf}\OnSightPathology
DefaultGroupName=OnSightPathology
OutputBaseFilename=OnSightPathologyInstaller
Compression=lzma
SolidCompression=yes
SetupIconFile="sample_icon.ico"

; --- Add these two lines --- ; ~2 GB per slice (max allowed)
;DiskSpanning=yes
;DiskSliceSize=2000000000   

[Files]
Source: "dist\app\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\OnSightPathology"; Filename: "{app}\app.exe"; IconFilename: "{app}\sample_icon.ico"
Name: "{group}\Uninstall OnSightPathology"; Filename: "{uninstallexe}"; IconFilename: "{app}\sample_icon.ico"
Name: "{commondesktop}\OnSightPathology"; Filename: "{app}\app.exe"; Tasks: desktopicon; IconFilename: "{app}\sample_icon.ico"


[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked