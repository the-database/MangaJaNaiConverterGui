; MangaJaNaiConverterGui setup installer (Inno Setup 6).
;
; Wraps the slim install tree that BuildMangaJaNai produces into a per-user Windows installer.
; Built in CI by deploy.yml:
;   ISCC.exe installer\mangajanaiconvertergui.iss /DAppVersion=<ver> /DSourceDir=<tree>
;
; PER-USER, NO ADMIN, by necessity: MangaJaNaiUpdater updates the app in place, which a Program
; Files install would block for a standard user, so the default location is %LOCALAPPDATA%\Programs.
; Saved settings, the Python backend and models live in %APPDATA%\MangaJaNaiConverterGui (the same
; folder 1.x used), so they carry over from a 1.x install.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #error Define SourceDir (the built install tree) with /DSourceDir=...
#endif

#define AppName "MangaJaNaiConverterGui"
#define Publisher "the-database"
#define AppExe "MangaJaNaiConverterGui.exe"

[Setup]
AppId={{530AF8E6-8505-4160-8FBC-C07D0B9A0910}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
AppPublisherURL=https://github.com/the-database/MangaJaNaiConverterGui
WizardStyle=modern
; Per-user install: no elevation, lands in %LOCALAPPDATA%\Programs\MangaJaNaiConverterGui (not
; %LOCALAPPDATA%\MangaJaNaiConverterGui, where 1.x's Velopack install lives).
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion}
OutputBaseFilename={#AppName}-Setup-{#AppVersion}
SetupIconFile=..\MangaJaNaiConverterGui\Assets\logo.ico
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

; Remove the whole app folder, including files created at runtime (update staging, the updater's
; .old copy). In %APPDATA% remove the Python backend and logs like 1.x's uninstaller did, but keep
; settings and models (which may include models the user added).
[UninstallDelete]
Type: filesandordirs; Name: "{app}"
Type: filesandordirs; Name: "{userappdata}\{#AppName}\python"
Type: filesandordirs; Name: "{userappdata}\{#AppName}\logs"

[Code]
// MangaJaNaiConverterGui 1.x was installed by Velopack to %LOCALAPPDATA%\MangaJaNaiConverterGui.
// Remove it here rather than through its own uninstaller: that runs a hook which deletes
// %APPDATA%\MangaJaNaiConverterGui\python and \models, which this version still uses.

function VelopackDir: String;
begin
  Result := ExpandConstant('{localappdata}\{#AppName}');
end;

function VelopackRetiredDir: String;
begin
  Result := VelopackDir + '.removing';
end;

function HasVelopackInstall: Boolean;
begin
  Result := FileExists(VelopackDir + '\Update.exe');
end;

// Moving the folder fails while 1.x is running, so it doubles as the "is it closed" check, and
// leaves the old install intact if it is still open.
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if HasVelopackInstall then
  begin
    DelTree(VelopackRetiredDir, True, True, True);
    if not RenameFile(VelopackDir, VelopackRetiredDir) then
      Result := 'Close {#AppName} 1.x (and any upscale it is running), then run setup again.' + #13#10#13#10 +
                'Setup replaces the 1.x install. Your settings, Python backend and models are kept.';
  end;
end;

// Runs before this install creates its shortcuts: 1.x's desktop shortcut has the same name.
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssInstall) and DirExists(VelopackRetiredDir) then
  begin
    DelTree(VelopackRetiredDir, True, True, True);
    DeleteFile(ExpandConstant('{userprograms}\{#AppName}.lnk'));
    DeleteFile(ExpandConstant('{userdesktop}\{#AppName}.lnk'));
    RegDeleteKeyIncludingSubkeys(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{#AppName}');
  end;
end;
