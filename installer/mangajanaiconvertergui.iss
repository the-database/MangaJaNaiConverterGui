; MangaJaNaiConverterGui setup installer (Inno Setup 6).
;
; Wraps the slim install tree that BuildMangaJaNai produces into a per-user Windows installer.
; Built in CI by deploy.yml:
;   ISCC.exe installer\mangajanaiconvertergui.iss /DAppVersion=<ver> /DSourceDir=<tree>
;
; PER-USER, NO ADMIN, by necessity: MangaJaNaiUpdater updates the app in place, which a Program
; Files install would block for a standard user, so the default location is %LOCALAPPDATA%\Programs.
; Saved settings and user-added models live in %APPDATA%\MangaJaNaiConverterGui (the same folder 1.x
; used), so they carry over from a 1.x install.
;
; The TensorRT runtime, the builder resources for the user's GPU and the default ONNX models are not
; bundled: they download post-install as component packs via MangaJaNaiUpdater.exe, matched to the
; detected GPU, keeping the installer small. A failed download is non-fatal: the app finishes the
; setup on its first launch.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #error Define SourceDir (the built install tree) with /DSourceDir=...
#endif

#define AppName "MangaJaNaiConverterGui"
#define Publisher "the-database"
#define AppExe "MangaJaNaiConverterGui.exe"
#define UpdaterExe "MangaJaNaiUpdater.exe"

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
; The updater needs its own entry (not just the wildcard) so ExtractTemporaryFile can run it for
; GPU detection during the wizard; exclude it from the wildcard to avoid listing it twice.
Source: "{#SourceDir}\{#UpdaterExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\*"; DestDir: "{app}"; Excludes: "{#UpdaterExe}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

; Remove the whole app folder, including files created at runtime (component packs, components.json,
; update staging, the updater's .old copy). In %APPDATA% remove the TensorRT engine cache and logs,
; but keep settings and models (which may include models the user added).
[UninstallDelete]
Type: filesandordirs; Name: "{app}"
Type: filesandordirs; Name: "{userappdata}\{#AppName}\engines"
Type: filesandordirs; Name: "{userappdata}\{#AppName}\logs"
Type: filesandordirs; Name: "{userappdata}\{#AppName}\python"

[Code]
var
  CompPage: TInputOptionWizardPage;
  GpuName: String;
  HasNvidia: Boolean;
  Packs: String;   // comma-separated, from MangaJaNaiUpdater --recommend (best-effort)

// Run MangaJaNaiUpdater.exe --recommend (from the given exe path), capturing its KEY=value lines into
// the globals. Returns False if it could not be run.
function RunRecommend(ExePath: String): Boolean;
var
  TmpFile, Cmd: String;
  Lines: TArrayOfString;
  i, eq: Integer;
  key, val: String;
  rc: Integer;
begin
  Result := False;
  TmpFile := ExpandConstant('{tmp}\mjn-recommend.txt');
  Cmd := '/C ""' + ExePath + '" --recommend > "' + TmpFile + '" 2>&1"';
  if not Exec(ExpandConstant('{cmd}'), Cmd, '', SW_HIDE, ewWaitUntilTerminated, rc) then
    Exit;
  if not LoadStringsFromFile(TmpFile, Lines) then
    Exit;
  for i := 0 to GetArrayLength(Lines) - 1 do
  begin
    eq := Pos('=', Lines[i]);
    if eq > 0 then
    begin
      key := Copy(Lines[i], 1, eq - 1);
      val := Copy(Lines[i], eq + 1, Length(Lines[i]) - eq);
      if key = 'NVIDIA' then HasNvidia := (val = '1')
      else if key = 'GPU' then GpuName := val
      else if key = 'PACKS' then Packs := val;
    end;
  end;
  Result := True;
end;

procedure InitializeWizard;
var
  trtLabel: String;
begin
  GpuName := '';
  HasNvidia := False;
  Packs := '';

  ExtractTemporaryFile('{#UpdaterExe}');
  RunRecommend(ExpandConstant('{tmp}\{#UpdaterExe}'));

  CompPage := CreateInputOptionPage(wpSelectTasks,
    'Components', 'Choose which AI components to install for your hardware.',
    'Selected components download after the core files are copied. You can change this any time ' +
    'from {#AppName}''s App Settings.',
    False, False);

  CompPage.Add('Upscaling models (MangaJaNai and IllustrationJaNai)');
  if HasNvidia then
    trtLabel := 'TensorRT engine (fastest) - for ' + GpuName
  else
    trtLabel := 'TensorRT engine - requires an NVIDIA GPU (your GPU uses the built-in DirectML engine)';
  CompPage.Add(trtLabel);

  CompPage.Values[0] := True;
  // TensorRT applies to NVIDIA GPUs only; other GPUs use the built-in DirectML engine.
  CompPage.Values[1] := HasNvidia;
  CompPage.CheckListBox.ItemEnabled[1] := HasNvidia;
end;

// Install one pack via the installed updater, showing it in the status label (downloads are large).
// Returns True on success.
function InstallPack(Pack: String): Boolean;
var
  rc: Integer;
begin
  WizardForm.StatusLabel.Caption := 'Downloading component: ' + Pack + ' (this may take a few minutes)...';
  WizardForm.Refresh;
  Result := Exec(ExpandConstant('{app}\{#UpdaterExe}'), '--install ' + Pack, ExpandConstant('{app}'),
                 SW_HIDE, ewWaitUntilTerminated, rc) and (rc = 0);
end;

// Install the recommended packs the user kept selected: models-* under the models option, the rest
// (trt-*) under the TensorRT option. Returns the count that failed.
function InstallSelectedPacks: Integer;
var
  csv, pack: String;
  comma: Integer;
  isModels: Boolean;
begin
  Result := 0;
  csv := Trim(Packs);
  while csv <> '' do
  begin
    comma := Pos(',', csv);
    if comma > 0 then
    begin
      pack := Trim(Copy(csv, 1, comma - 1));
      csv := Copy(csv, comma + 1, Length(csv) - comma);
    end
    else
    begin
      pack := Trim(csv);
      csv := '';
    end;
    if pack = '' then
      Continue;
    isModels := Pos('models-', pack) = 1;
    if (isModels and CompPage.Values[0]) or ((not isModels) and CompPage.Values[1]) then
      if not InstallPack(pack) then
        Result := Result + 1;
  end;
end;

// MangaJaNaiConverterGui 1.x was installed by Velopack to %LOCALAPPDATA%\MangaJaNaiConverterGui.
// Remove it here rather than through its own uninstaller: that runs a hook which deletes
// %APPDATA%\MangaJaNaiConverterGui\models, which may hold models the user added.

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
                'Setup replaces the 1.x install. Your settings and models are kept.';
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  failed: Integer;
begin
  if CurStep = ssInstall then
  begin
    // Before this install creates its shortcuts: 1.x's desktop shortcut has the same name.
    if DirExists(VelopackRetiredDir) then
    begin
      DelTree(VelopackRetiredDir, True, True, True);
      DeleteFile(ExpandConstant('{userprograms}\{#AppName}.lnk'));
      DeleteFile(ExpandConstant('{userdesktop}\{#AppName}.lnk'));
      RegDeleteKeyIncludingSubkeys(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{#AppName}');
    end;
    // 1.x's Python backend (several GB): 2.0 runs on its own inference engine instead.
    DelTree(ExpandConstant('{userappdata}\{#AppName}\python'), True, True, True);
  end
  else if CurStep = ssPostInstall then
  begin
    // Re-detect from the now-installed updater: it knows the installed version (so the matching
    // packs), and the network may be available now even if it wasn't at wizard start.
    RunRecommend(ExpandConstant('{app}\{#UpdaterExe}'));
    failed := InstallSelectedPacks;
    WizardForm.StatusLabel.Caption := '';
    if failed > 0 then
      MsgBox('Some components could not be downloaded (you may be offline). {#AppName} will still ' +
             'open and finish setting them up, or use App Settings later.',
             mbInformation, MB_OK);
  end;
end;
