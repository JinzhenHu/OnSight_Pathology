[Setup]
AppName=OnSightPathologyWSI
AppVersion=1.0
DefaultDirName={pf}\OnSightPathologyWSI
DefaultGroupName=OnSightPathologyWSI
OutputBaseFilename=OnSightPathologyWSIInstaller
Compression=lzma
SolidCompression=yes
SetupIconFile="sample_icon.ico"
UninstallDisplayIcon={app}\sample_icon.ico
OutputDir=output

[Files]
Source: "dist\app\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\OnSightPathologyWSI"; Filename: "{app}\app.exe"; IconFilename: "{app}\sample_icon.ico"
Name: "{group}\Uninstall OnSightPathologyWSI"; Filename: "{uninstallexe}"; IconFilename: "{app}\sample_icon.ico"
Name: "{commondesktop}\OnSightPathologyWSI"; Filename: "{app}\app.exe"; Tasks: desktopicon; IconFilename: "{app}\sample_icon.ico"


[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked