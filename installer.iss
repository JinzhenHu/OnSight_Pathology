[Setup]
; --- 基本信息 ---
AppName=OnSightPathology
AppVersion=1.0
DefaultDirName={pf}\OnSightPathology
DefaultGroupName=OnSightPathology
; 最终生成的安装文件名
OutputBaseFilename=OnSightPathologyInstaller
OutputDir=output

; --- 打包设置（关键修改点） ---
; 关闭磁盘分卷，确保所有数据都塞进一个 EXE
DiskSpanning=no
; 开启固实压缩以获得最小体积
Compression=lzma2/max
SolidCompression=yes
; 使用独立进程进行压缩，提高速度
LZMAUseSeparateProcess=yes

; --- 图标设置 ---
; 请确保脚本目录下存在此图标文件
SetupIconFile="sample_icon.ico"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加图标:"; Flags: unchecked

[Files]
; 这里的 Source 必须指向你电脑上实际存放程序的目录
; 建议使用绝对路径或者确保脚本文件和 dist 文件夹在同一目录下
Source: "sample_icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\app\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
; 菜单快捷方式
Name: "{group}\OnSightPathology"; Filename: "{app}\app.exe"; IconFilename: "{app}\sample_icon.ico"
; 卸载快捷方式
Name: "{group}\卸载 OnSightPathology"; Filename: "{uninstallexe}"; IconFilename: "{app}\sample_icon.ico"
; 桌面快捷方式（受 Tasks 勾选控制）
Name: "{commondesktop}\OnSightPathology"; Filename: "{app}\app.exe"; Tasks: desktopicon; IconFilename: "{app}\sample_icon.ico"

[Run]
; 安装完成后询问是否立即运行
Description: "运行 OnSightPathology"; Filename: "{app}\app.exe"; Flags: nowait postinstall skipifsilent