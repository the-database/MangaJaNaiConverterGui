# Assembles the install tree and the full-package zip the updater downloads.
# Used by .github/workflows/deploy.yml; runnable locally for testing:
#
#   dotnet publish MangaJaNaiConverterGui/MangaJaNaiConverterGui.csproj -c Release -r win-x64 --self-contained -o app-publish
#   dotnet publish MangaJaNaiUpdater/MangaJaNaiUpdater.csproj -c Release -o updater-publish
#   ./installer/build-package.ps1 -Version 2.0.0 -AppDir app-publish -UpdaterExe updater-publish/MangaJaNaiUpdater.exe -OutDir publish
#
# Outputs, under OutDir:
#   MangaJaNaiConverterGui-v<ver>/                       the install tree (the Inno installer's SourceDir)
#   MangaJaNaiConverterGui-full-package-<ver>.zip        the tree, rooted at MangaJaNaiConverterGui-v<ver>/
param(
    [Parameter(Mandatory)][string]$Version,
    [Parameter(Mandatory)][string]$AppDir,
    [Parameter(Mandatory)][string]$UpdaterExe,
    [Parameter(Mandatory)][string]$OutDir
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$appName = 'MangaJaNaiConverterGui'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$tree = Join-Path (Resolve-Path $OutDir) "$appName-v$Version"
if (Test-Path $tree) { Remove-Item -Recurse -Force $tree }
New-Item -ItemType Directory $tree | Out-Null

Copy-Item -Recurse (Join-Path $AppDir '*') $tree
Copy-Item $UpdaterExe (Join-Path $tree 'MangaJaNaiUpdater.exe')
# Debug symbols (SkiaSharp's alone are ~100 MB) aren't useful to users, and __pycache__ folders are
# leftovers from running the backend from a dev checkout.
Get-ChildItem -Recurse -File -Filter *.pdb $tree | Remove-Item
Get-ChildItem -Recurse -Directory -Filter __pycache__ $tree | Remove-Item -Recurse
Set-Content -NoNewline -Encoding ascii (Join-Path $tree 'version.txt') $Version

# files: everything this version ships, so the next update can remove what it drops.
# user_preserve: install-relative paths an update never overwrites or removes (settings, Python
# and models live in %APPDATA%, so nothing in the install needs it yet).
$files = Get-ChildItem -Recurse -File $tree |
    ForEach-Object { [System.IO.Path]::GetRelativePath($tree, $_.FullName).Replace('\', '/') }
$files = @($files) + 'manifest.json' | Sort-Object
$manifest = [ordered]@{
    package_version = $Version
    app_executable  = "$appName.exe"
    platform        = 'win-x64'
    user_preserve   = @()
    files           = $files
}
$manifest | ConvertTo-Json -Depth 3 | Set-Content -Encoding utf8NoBOM (Join-Path $tree 'manifest.json')

$zip = Join-Path (Resolve-Path $OutDir) "$appName-full-package-$Version.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
[System.IO.Compression.ZipFile]::CreateFromDirectory($tree, $zip, [System.IO.Compression.CompressionLevel]::Optimal, $true)

Write-Output "Install tree: $tree"
Write-Output "Full package: $zip"
