// Package builder for MangaJaNaiConverterGui (Inno-installer distribution model, Windows).
// Ported from VideoJaNai's BuildVideoJaNai, without ffmpeg/RIFE/Linux.
//
// Assembles the redistributable install tree (the app + MangaJaNaiUpdater + the inference dir: libaji,
// the mangajanai_upscale backend, ONNX Runtime/DirectML, the VC++ runtime) and, under --packs, carves
// the heavy optional pieces (TensorRT runtime, per-GPU-generation builder resources, the ONNX models)
// into component-*.7z + packs.json release assets, then slims them out of the core. End users download
// the small core plus the packs their hardware needs.
//
//   BuildMangaJaNai <version> --app-dir <dir> --updater <exe> [--packs] [--out <dir>]
//
// Outputs, under --out (default: next to this exe):
//   MangaJaNaiConverterGui-v<ver>/                    the (slim) install tree, the Inno installer's SourceDir
//   MangaJaNaiConverterGui-full-package-<ver>.zip     that tree, the updater's full update
//   packs-v<ver>/component-*.7z + packs.json          component packs (with --packs)
//
// Dev overrides (skip large downloads on local builds):
//   TRT_LOCAL_ZIP           -> a downloaded TensorRT "-external" zip
//   TRT_LOCAL_DIR           -> an existing inference dir holding the TensorRT runtime + cudart (e.g. VideoJaNai's)
//   AJI_LOCAL_ZIP           -> a locally built aji-windows-x64.zip
//   AJI_LOCAL_DIR           -> a directory of aji files copied flat into inference/
//   MANGAJANAI_CLI_LOCAL    -> a local mangajanai_upscale.exe, or a directory/zip holding it
//   MANGAJANAI_MODELS_DIR   -> a local directory of the model archives (models-*.7z) or loose .onnx files
//   VC_CRT_DIR              -> a Microsoft.VC14x.CRT directory (default: found through vswhere)
using ICSharpCode.SharpZipLib.Core;
using ICSharpCode.SharpZipLib.Zip;
using SevenZipExtractor;
using System.Diagnostics;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using static Downloader;

// Third-party component versions. Bump these together when cutting a release.
//
// The TensorRT runtime + trtexec come straight from NVIDIA's public download (the "-external" zip),
// the same pins as VideoJaNai. TrtCudaVersion is its own pin, NOT derivable from TrtVersion: guessing
// it 404s.
const string TrtVersion        = "11.4.0.106";
const string TrtCudaVersion    = "13.4";
// NVIDIA publishes no checksum beside the zip, so this is recorded from a verified download.
const string TrtSha256Win      = "b6fd86301ad803492adf7a06de29ce45e3968d29637a1437d0fb0994f412c980";

// CUDA runtime, from NVIDIA's CUDA redistributable (per-file checksums in
// developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.1.json).
const string CudartVersion     = "13.4.49";
const string CudartSha256Win   = "e6663f3d3e8949eedc2d5ab92c7c5b9fa3f2a222086d91c42bfc5a38bf2b0225";

// the-database/animejanai-inference. v0.18.0 added the still-image ABI (aji_image.h) the backend uses;
// v0.18.2 makes DirectML device ids the card's PCI address, so they match TensorRT's and survive reboots.
const string AjiVersion        = "v0.18.2";
const string AjiSha256Win      = "e0c9c08335c234682f2f3583f1775d0274fc2516dfb28aeb70b668aed38ed481";
const string SevenZipVersion   = "2602";                  // 7-zip "extra": 7za.exe for the updater's packs
// DirectML backend runtime (aji_dml.dll loads both from its own directory). The last DirectML-flavored
// ORT releases (DML is in sustained engineering); same pins as VideoJaNai.
const string OrtDmlVersion     = "1.24.4";                // Microsoft.ML.OnnxRuntime.DirectML (NuGet)
const string DirectMLVersion   = "1.15.4";                // Microsoft.AI.DirectML (NuGet)

// The upscaling backend (yosh repo, crates/mangajanai-cli): mangajanai_upscale.exe, built on
// yosh-image + libaji, released on its own mangajanai-cli-v* tags. MANGAJANAI_CLI_LOCAL overrides it
// with a local build.
const string CliRepo           = "the-database/yosh";
const string CliVersion        = "mangajanai-cli-v0.1.0";
const string CliAsset          = "mangajanai-cli-windows-x64.zip";
const string CliSha256         = "1f43f8e953c300971785ce4388208dee50973146e9f70739477a67f1d45f4a6d";

// The default ONNX models (opset 17, fp16 where the model allows: one set for TensorRT and DirectML),
// published as an archive on each model's MangaJaNai release. Local builds can use
// MANGAJANAI_MODELS_DIR instead.
const string ModelsRepo        = "the-database/MangaJaNai";
const string MangaJaNaiModelsTag        = "1.0.0";
const string MangaJaNaiModelsAsset      = "MangaJaNai_V1_ONNX.7z";
const string IllustrationJaNaiModelsTag   = "3.0.0";
const string IllustrationJaNaiModelsAsset = "IllustrationJaNai_V3_ONNX.7z";
(string Tag, string Asset)[] modelArchives =
[
    (MangaJaNaiModelsTag, MangaJaNaiModelsAsset),
    (IllustrationJaNaiModelsTag, IllustrationJaNaiModelsAsset),
];

const string AppName = "MangaJaNaiConverterGui";

if (args.Length < 1 || args[0].StartsWith("--"))
{
    throw new ArgumentException("Version is required.  Usage: BuildMangaJaNai <version> --app-dir <dir> --updater <exe> [--packs] [--out <dir>]");
}

var version = args[0];

string? ArgValue(string flag)
{
    int i = Array.IndexOf(args, flag);
    return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
}

var outDirectory = Path.GetFullPath(ArgValue("--out") ?? AppContext.BaseDirectory);
var installDirectory = Path.Combine(outDirectory, $"{AppName}-v{version}");
var inferencePath = Path.Combine(installDirectory, "inference");
var modelsPath = Path.Combine(installDirectory, "models");
var workDirectory = Path.Combine(outDirectory, "build-work");

// Standalone 7-Zip console: builds the component packs, and ships at the install root for the
// updater to install them (manifest archive_tool).
async Task InstallSevenZip()
{
    Console.WriteLine("Downloading 7-Zip standalone console...");
    var downloadUrl = $"https://www.7-zip.org/a/7z{SevenZipVersion}-extra.7z";
    var targetPath = Path.Combine(workDirectory, "7z-extra.7z");
    await DownloadFileAsync(downloadUrl, targetPath, p => Console.WriteLine($"Downloading 7-Zip ({p}%)..."));

    var targetExtractPath = Path.Combine(workDirectory, "7z-extra");
    Directory.CreateDirectory(targetExtractPath);
    using (ArchiveFile archiveFile = new(targetPath))
    {
        archiveFile.Extract(targetExtractPath);
    }
    File.Copy(Path.Combine(targetExtractPath, "x64", "7za.exe"), Path.Combine(installDirectory, "7za.exe"), true);
}

async Task InstallInferenceRuntime()
{
    Directory.CreateDirectory(inferencePath);
    var localDir = Environment.GetEnvironmentVariable("TRT_LOCAL_DIR");
    if (!string.IsNullOrEmpty(localDir))
    {
        Console.WriteLine($"Using local TensorRT runtime: {localDir}");
        foreach (var f in Directory.GetFiles(localDir).Where(f => (KeepInferenceFile(Path.GetFileName(f)) && !Path.GetFileName(f).StartsWith("ffmpeg") && !Path.GetFileName(f).StartsWith("DirectML")) || Path.GetFileName(f) == "TensorRT_Acknowledgements.txt"))
        {
            File.Copy(f, Path.Combine(inferencePath, Path.GetFileName(f)), true);
        }
        return;
    }
    // NVIDIA's "-external" zip. Only bin/ ships; the headers, import libs, cmake/ and python/ are
    // build-time material for animejanai-inference.
    var trtDir = string.Join('.', TrtVersion.Split('.').Take(3));
    var asset = $"TensorRT-Enterprise-{TrtVersion}-Windows-amd64-cuda-{TrtCudaVersion}-Release-external.zip";
    var downloadUrl = $"https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/{trtDir}/zip/{asset}";

    var localZip = Environment.GetEnvironmentVariable("TRT_LOCAL_ZIP");
    string targetPath;
    if (!string.IsNullOrEmpty(localZip))
    {
        Console.WriteLine($"Using local TensorRT archive: {localZip}");
        targetPath = localZip;
    }
    else
    {
        Console.WriteLine($"Downloading TensorRT {TrtVersion} (CUDA {TrtCudaVersion}) from NVIDIA...");
        targetPath = Path.Combine(workDirectory, asset);
        double lastProgress = -1;
        await DownloadFileAsync(downloadUrl, targetPath, p =>
        {
            if (p >= lastProgress + 5) { Console.WriteLine($"Downloading TensorRT ({p}%)..."); lastProgress = p; }
        });
    }
    VerifySha256(targetPath, TrtSha256Win);

    Console.WriteLine("Extracting TensorRT runtime...");
    ExtractZipFiltered(targetPath, entry =>
    {
        var name = Path.GetFileName(entry);
        if (entry.Contains("/bin/", StringComparison.Ordinal) && KeepInferenceFile(name))
            return Path.Combine(inferencePath, name);
        // NVIDIA's own attribution file, kept beside the binaries it covers.
        if (name.Equals("Acknowledgements.txt", StringComparison.OrdinalIgnoreCase))
            return Path.Combine(inferencePath, "TensorRT_Acknowledgements.txt");
        return null;
    });

    await InstallCudart();
}

// cudart is not part of the TensorRT archive; it comes from NVIDIA's CUDA redistributable.
async Task InstallCudart()
{
    var stem = $"cuda_cudart-windows-x86_64-{CudartVersion}-archive";
    var downloadUrl = $"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/{stem}.zip";
    var targetPath = Path.Combine(workDirectory, $"{stem}.zip");
    Console.WriteLine($"Downloading CUDA runtime {CudartVersion}...");
    await DownloadFileAsync(downloadUrl, targetPath, _ => { });
    VerifySha256(targetPath, CudartSha256Win);

    ExtractZipFiltered(targetPath, entry =>
    {
        var name = Path.GetFileName(entry);
        if (name.Equals("LICENSE", StringComparison.OrdinalIgnoreCase))
            return Path.Combine(inferencePath, "CUDA_LICENSE.txt");
        return KeepInferenceFile(name) ? Path.Combine(inferencePath, name) : null;
    });
}

// Which inference-runtime files from NVIDIA's archives actually ship.
static bool KeepInferenceFile(string name) =>
    name is "nvinfer_11.dll" or "nvinfer_plugin_11.dll" or "nvonnxparser_11.dll" or "trtexec.exe" ||
    name.StartsWith("cudart64_", StringComparison.OrdinalIgnoreCase) ||
    name.StartsWith("nvinfer_builder_resource_", StringComparison.OrdinalIgnoreCase) ||
    name.Contains("LICENSE", StringComparison.OrdinalIgnoreCase);

static void VerifySha256(string path, string expected)
{
    using var stream = File.OpenRead(path);
    var actual = Convert.ToHexString(SHA256.HashData(stream));
    if (!actual.Equals(expected, StringComparison.OrdinalIgnoreCase))
    {
        throw new InvalidOperationException(
            $"SHA-256 mismatch for {Path.GetFileName(path)}{Environment.NewLine}" +
            $"  expected {expected}{Environment.NewLine}" +
            $"  actual   {actual.ToLowerInvariant()}");
    }
    Console.WriteLine($"  sha256 verified: {Path.GetFileName(path)}");
}

async Task InstallAji()
{
    Directory.CreateDirectory(inferencePath);

    var localDir = Environment.GetEnvironmentVariable("AJI_LOCAL_DIR");
    if (!string.IsNullOrEmpty(localDir))
    {
        Console.WriteLine($"Using local aji files: {localDir}");
        foreach (var f in Directory.GetFiles(localDir))
        {
            File.Copy(f, Path.Combine(inferencePath, Path.GetFileName(f)), true);
        }
        return;
    }

    var localZip = Environment.GetEnvironmentVariable("AJI_LOCAL_ZIP");
    string targetPath;
    if (!string.IsNullOrEmpty(localZip))
    {
        Console.WriteLine($"Using local aji build: {localZip}");
        targetPath = localZip;
    }
    else
    {
        const string asset = "aji-windows-x64.zip";
        Console.WriteLine($"Downloading {asset} {AjiVersion}...");
        targetPath = Path.Combine(workDirectory, asset);
        await DownloadFileAsync($"https://github.com/the-database/animejanai-inference/releases/download/{AjiVersion}/{asset}",
            targetPath, p => Console.WriteLine($"Downloading {asset} ({p}%)..."));
        VerifySha256(targetPath, AjiSha256Win);
    }
    // Only the libraries: the video tools (aji_encode, harnesses) aren't used by the image backend.
    ExtractZipFiltered(targetPath, entry =>
    {
        var name = Path.GetFileName(entry);
        return name.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) || name.Contains("LICENSE", StringComparison.OrdinalIgnoreCase)
            ? Path.Combine(inferencePath, name)
            : null;
    });
}

// ONNX Runtime + DirectML for the DirectML backend. They ship in the core rather than a component
// pack: DirectML is what non-NVIDIA GPUs run on, and those get no TensorRT packs. The .nupkg files
// are plain zips; only the x64 runtime DLLs (and the DirectML license, which the redistribution terms
// require keeping intact) go into the package.
async Task InstallOrtDml()
{
    Directory.CreateDirectory(inferencePath);
    var packages = new (string Name, string Version, string[] CopyFromTo)[]
    {
        ("Microsoft.ML.OnnxRuntime.DirectML", OrtDmlVersion, new[]
        {
            "runtimes/win-x64/native/onnxruntime.dll", "onnxruntime.dll",
        }),
        ("Microsoft.AI.DirectML", DirectMLVersion, new[]
        {
            "bin/x64-win/DirectML.dll", "DirectML.dll",
            "LICENSE.txt", "DirectML_LICENSE.txt",
        }),
    };
    foreach (var (name, pkgVersion, copies) in packages)
    {
        Console.WriteLine($"Downloading {name} {pkgVersion}...");
        var targetPath = Path.Combine(workDirectory, $"{name}.{pkgVersion}.nupkg");
        await DownloadFileAsync($"https://www.nuget.org/api/v2/package/{name}/{pkgVersion}", targetPath, _ => { });
        var tempDirectory = Path.Combine(workDirectory, $"{name}-temp");
        ExtractZip(targetPath, tempDirectory);
        for (var i = 0; i < copies.Length; i += 2)
        {
            File.Copy(Path.Combine(tempDirectory, copies[i].Replace('/', Path.DirectorySeparatorChar)),
                      Path.Combine(inferencePath, copies[i + 1]), true);
        }
    }
}

// libaji's DLLs link the MSVC runtime dynamically; ship it app-local (allowed by the VC++
// redistributable terms) so a machine without the VC++ Redistributable installed still loads them.
void InstallVcRuntime()
{
    var crtDir = Environment.GetEnvironmentVariable("VC_CRT_DIR");
    if (string.IsNullOrEmpty(crtDir))
    {
        var vswhere = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86),
            "Microsoft Visual Studio", "Installer", "vswhere.exe");
        if (File.Exists(vswhere))
        {
            var psi = new ProcessStartInfo(vswhere,
                @"-latest -products * -find VC\Redist\MSVC\**\x64\Microsoft.VC14*.CRT\vcruntime140.dll")
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
            };
            using var p = Process.Start(psi)!;
            var found = p.StandardOutput.ReadToEnd()
                .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .FirstOrDefault(l => !l.Contains(@"\onecore\", StringComparison.OrdinalIgnoreCase));
            p.WaitForExit();
            crtDir = found is null ? null : Path.GetDirectoryName(found);
        }
    }
    if (string.IsNullOrEmpty(crtDir) || !Directory.Exists(crtDir))
    {
        Console.WriteLine("WARNING: VC++ runtime not found (set VC_CRT_DIR); not bundled.");
        return;
    }
    Console.WriteLine($"Copying VC++ runtime from {crtDir}...");
    foreach (var dll in Directory.GetFiles(crtDir, "*.dll"))
    {
        File.Copy(dll, Path.Combine(inferencePath, Path.GetFileName(dll)), true);
    }
}

// The upscaling backend executable.
async Task InstallBackendCli()
{
    const string exe = "mangajanai_upscale.exe";
    var local = Environment.GetEnvironmentVariable("MANGAJANAI_CLI_LOCAL");
    string? archive = null;
    if (!string.IsNullOrEmpty(local))
    {
        Console.WriteLine($"Using local backend: {local}");
        if (File.Exists(local) && local.EndsWith(".exe", StringComparison.OrdinalIgnoreCase))
        {
            File.Copy(local, Path.Combine(inferencePath, exe), true);
            return;
        }
        if (Directory.Exists(local))
        {
            File.Copy(Path.Combine(local, exe), Path.Combine(inferencePath, exe), true);
            return;
        }
        archive = local;
    }
    else if (CliVersion.Length > 0)
    {
        Console.WriteLine($"Downloading {CliAsset} {CliVersion}...");
        archive = Path.Combine(workDirectory, CliAsset);
        await DownloadFileAsync($"https://github.com/{CliRepo}/releases/download/{CliVersion}/{CliAsset}",
            archive, p => Console.WriteLine($"Downloading {CliAsset} ({p}%)..."));
        VerifySha256(archive, CliSha256);
    }
    if (archive is null)
    {
        Console.WriteLine($"WARNING: no backend release pinned and MANGAJANAI_CLI_LOCAL unset; {exe} not bundled.");
        return;
    }
    ExtractZipFiltered(archive, entry =>
    {
        var name = Path.GetFileName(entry);
        return name.Equals(exe, StringComparison.OrdinalIgnoreCase) || name.Contains("LICENSE", StringComparison.OrdinalIgnoreCase)
            ? Path.Combine(inferencePath, name.Equals(exe, StringComparison.OrdinalIgnoreCase) ? exe : $"mangajanai_upscale_{name}")
            : null;
    });
}

// The default ONNX models, into models/ (split into the models-* component packs under --packs).
async Task InstallModels()
{
    Directory.CreateDirectory(modelsPath);
    var local = Environment.GetEnvironmentVariable("MANGAJANAI_MODELS_DIR");
    var archives = new List<string>();
    if (!string.IsNullOrEmpty(local))
    {
        Console.WriteLine($"Using local models: {local}");
        foreach (var onnx in Directory.GetFiles(local, "*.onnx"))
        {
            File.Copy(onnx, Path.Combine(modelsPath, Path.GetFileName(onnx)), true);
        }
        archives.AddRange(modelArchives.Select(a => Path.Combine(local, a.Asset)).Where(File.Exists));
    }
    else
    {
        foreach (var (tag, asset) in modelArchives)
        {
            Console.WriteLine($"Downloading {asset} ({tag})...");
            var targetPath = Path.Combine(workDirectory, asset);
            double lastProgress = -1;
            await DownloadFileAsync($"https://github.com/{ModelsRepo}/releases/download/{tag}/{asset}", targetPath, p =>
            {
                if (p >= lastProgress + 10) { Console.WriteLine($"Downloading {asset} ({p}%)..."); lastProgress = p; }
            });
            archives.Add(targetPath);
        }
    }
    foreach (var archive in archives)
    {
        Console.WriteLine($"Extracting {Path.GetFileName(archive)}...");
        using ArchiveFile archiveFile = new(archive);
        archiveFile.Extract(modelsPath);
    }
}

// The TensorRT SLA requires this attribution when redistributing the runtime.
void WriteThirdPartyNotices()
{
    var notice =
        "Third-party components in this directory\n" +
        "========================================\n\n" +
        "NVIDIA TensorRT runtime (nvinfer_11.dll, nvinfer_plugin_11.dll, nvonnxparser_11.dll,\n" +
        "nvinfer_builder_resource_*.dll, trtexec.exe) and NVIDIA CUDA runtime (cudart64_*.dll),\n" +
        "redistributed under the NVIDIA TensorRT Software License Agreement and\n" +
        "CUDA Toolkit EULA:\n\n" +
        "    This software contains source code provided by NVIDIA Corporation.\n\n" +
        "These files are obtained from NVIDIA's redistributable TensorRT and CUDA\n" +
        "downloads, redistributed under the same terms. NVIDIA's own acknowledgements\n" +
        "accompany them as TensorRT_Acknowledgements.txt.\n\n" +
        "ONNX Runtime (onnxruntime.dll), (c) Microsoft Corporation, redistributed\n" +
        "under the MIT license (https://github.com/microsoft/onnxruntime/blob/main/LICENSE).\n\n" +
        "DirectML (DirectML.dll), (c) Microsoft Corporation, redistributed as the\n" +
        "DirectML Redistributable Package under the Microsoft Software License Terms\n" +
        "shipped alongside it as DirectML_LICENSE.txt (use on Windows and Xbox only).\n\n" +
        "Microsoft Visual C++ runtime (msvcp140*.dll, vcruntime140*.dll, concrt140.dll),\n" +
        "(c) Microsoft Corporation, redistributed under the Visual Studio redistributable terms.\n\n" +
        "aji.dll / aji_trt.dll / aji_dml.dll:\n" +
        "https://github.com/the-database/animejanai-inference\n\n" +
        "mangajanai_upscale.exe (built on yosh-engine):\n" +
        "https://github.com/the-database/yosh\n";
    File.WriteAllText(Path.Combine(inferencePath, "THIRD_PARTY_NOTICES.txt"), notice);
}

// version.txt + manifest.json at the install root. The updater reads these to know the installed
// version, which files the previous version shipped (to prune ones a new version drops), which paths
// to preserve, and the dep versions that tell whether an installed component pack is still current.
// Written last, after the packs are carved out, so "files" lists the core only.
void WriteVersionAndManifest()
{
    File.WriteAllText(Path.Combine(installDirectory, "version.txt"), version);

    var files = Directory.GetFiles(installDirectory, "*", SearchOption.AllDirectories)
        .Select(f => Path.GetRelativePath(installDirectory, f).Replace('\\', '/'))
        .Append("manifest.json")
        .Distinct()
        .Order(StringComparer.Ordinal)
        .ToArray();

    var manifest = new
    {
        package_version = version,
        app_executable = $"{AppName}.exe",
        archive_tool = "7za.exe",
        platform = "win-x64",
        // The versions that govern each component pack's content; bumping one makes the updater
        // reinstall that pack.
        deps = new
        {
            aji = AjiVersion,
            inference_runtime = $"trt-{TrtVersion}",
            ort_dml = $"{OrtDmlVersion}+{DirectMLVersion}",
            backend = CliVersion,
            // a pack's content changes only when its archive does
            models_mangajanai = $"{MangaJaNaiModelsAsset}@{MangaJaNaiModelsTag}",
            models_illustrationjanai = $"{IllustrationJaNaiModelsAsset}@{IllustrationJaNaiModelsTag}",
            sevenzip = SevenZipVersion,
        },
        // Runtime-created data in the install folder that an update must never touch.
        user_preserve = new[] { "components.json" },
        files,
    };

    var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
    File.WriteAllText(Path.Combine(installDirectory, "manifest.json"), json);
}

// The full package the updater downloads: the install tree, rooted at its versioned folder name.
void WriteFullPackage()
{
    var zip = Path.Combine(outDirectory, $"{AppName}-full-package-{version}.zip");
    if (File.Exists(zip)) File.Delete(zip);
    Console.WriteLine($"Writing {Path.GetFileName(zip)}...");
    System.IO.Compression.ZipFile.CreateFromDirectory(installDirectory, zip, CompressionLevel.Optimal, includeBaseDirectory: true);
    Console.WriteLine($"  {new FileInfo(zip).Length / 1048576} MB");
}

// Selective zip extraction: mapEntry returns the destination for an entry, or null to skip it. Used
// for the TensorRT and CUDA archives, where only a handful of files out of ~2.5 GB are wanted.
static void ExtractZipFiltered(string archivePath, Func<string, string?> mapEntry)
{
    using var fsInput = File.OpenRead(archivePath);
    using var zf = new ICSharpCode.SharpZipLib.Zip.ZipFile(fsInput);
    var buffer = new byte[81920];
    foreach (ZipEntry zipEntry in zf)
    {
        if (!zipEntry.IsFile) continue;
        var destination = mapEntry(zipEntry.Name);
        if (destination == null) continue;
        var directoryName = Path.GetDirectoryName(destination);
        if (directoryName?.Length > 0) Directory.CreateDirectory(directoryName);
        Console.WriteLine($"  {Path.GetFileName(destination)}");
        using var zipStream = zf.GetInputStream(zipEntry);
        using Stream fsOutput = File.Create(destination);
        StreamUtils.Copy(zipStream, fsOutput, buffer);
    }
}

static void ExtractZip(string archivePath, string outFolder) =>
    ExtractZipFiltered(archivePath, entry => Path.Combine(outFolder, entry));

static void CopyDirectory(string srcDir, string targetDir)
{
    Directory.CreateDirectory(targetDir);
    foreach (string file in Directory.GetFiles(srcDir))
    {
        File.Copy(file, Path.Combine(targetDir, Path.GetFileName(file)), true);
    }
    foreach (string subDir in Directory.GetDirectories(srcDir))
    {
        CopyDirectory(subDir, Path.Combine(targetDir, Path.GetFileName(subDir)));
    }
}

// The released full package is the slim core: the TensorRT runtime, per-GPU builder resources and
// the models ship only as component packs, installed on demand. Keeps the base download small.
void SlimInstallTree(List<string> packFiles)
{
    Console.WriteLine("Slimming install tree (component packs ship separately)...");
    long removed = 0;
    foreach (var rel in packFiles)
    {
        var abs = Path.Combine(installDirectory, rel);
        if (!File.Exists(abs)) continue;
        removed += new FileInfo(abs).Length;
        File.Delete(abs);
    }
    foreach (var dir in Directory.GetDirectories(installDirectory, "*", SearchOption.AllDirectories)
                                 .OrderByDescending(d => d.Length))
    {
        if (!Directory.EnumerateFileSystemEntries(dir).Any()) Directory.Delete(dir);
    }
    Console.WriteLine($"Slimmed {removed / 1048576} MB out of the package tree.");
}

// Component packs: rooted 7z archives + a packs.json index, downloaded on demand by the updater /
// in-app manager. Archive paths are relative to the install root, so extraction over an install IS
// installation. Returns the expanded per-file list packed (so --packs can strip them from the core).
async Task<List<string>> EmitComponentPacks()
{
    Console.WriteLine("Emitting component packs...");
    var packsDir = Path.Combine(outDirectory, $"packs-v{version}");
    if (Directory.Exists(packsDir)) Directory.Delete(packsDir, true);
    Directory.CreateDirectory(packsDir);
    var sevenZ = Path.Combine(installDirectory, "7za.exe");

    string[] Rel(IEnumerable<string> paths) => paths.Select(f => Path.GetRelativePath(installDirectory, f)).ToArray();
    string[] ModelFiles(string marker) => Directory.Exists(modelsPath)
        ? Rel(Directory.GetFiles(modelsPath, "*.onnx").Where(f => Path.GetFileName(f).Contains(marker, StringComparison.OrdinalIgnoreCase)))
        : [];

    // Dep = the manifest.json deps key that governs a pack's content, so the updater can skip an
    // already-present pack whose dep is unchanged across releases.
    var packs = new List<(string Name, string Dep, string[] Files)>
    {
        ("trt-runtime", "inference_runtime", Rel(Directory.GetFiles(inferencePath).Where(f =>
        {
            var n = Path.GetFileName(f);
            bool builderResource = n.Contains("builder_resource");
            bool runtime = n.StartsWith("nvinfer") || n.StartsWith("nvonnxparser") ||
                           n.StartsWith("cudart64_") || n == "trtexec.exe" ||
                           n is "CUDA_LICENSE.txt" or "TensorRT_Acknowledgements.txt";
            return runtime && !builderResource;
        }))),
        ("models-mangajanai", "models_mangajanai", ModelFiles("MangaJaNai_")),
        ("models-illustrationjanai", "models_illustrationjanai", ModelFiles("IllustrationJaNai_")),
    };
    foreach (var f in Directory.GetFiles(inferencePath, "*builder_resource_*"))
    {
        // nvinfer_builder_resource_sm120_11.dll -> trt-sm120
        var m = System.Text.RegularExpressions.Regex.Match(Path.GetFileName(f), @"builder_resource_([a-z0-9]+)[._]");
        if (m.Success)
        {
            packs.Add(($"trt-{m.Groups[1].Value}", "inference_runtime", Rel([f])));
        }
    }

    var index = new List<object>();
    var packedFiles = new List<string>();
    foreach (var (name, dep, files) in packs)
    {
        if (files.Length == 0)
        {
            Console.WriteLine($"  component-{name}: nothing to pack in this tree, skipped");
            continue;
        }
        var archive = Path.Combine(packsDir, $"component-{name}.7z");
        var fileArgs = string.Join(' ', files.Select(f => $"\"{f}\""));
        var psi = new ProcessStartInfo
        {
            FileName = sevenZ,
            // -spf2: store the relative paths as given
            Arguments = $"a -spf2 -mx=3 \"{archive}\" {fileArgs}",
            WorkingDirectory = installDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        using (var proc = Process.Start(psi)!)
        {
            // drain both pipes: a full one would stall the archiver
            var stdout = proc.StandardOutput.ReadToEndAsync();
            var stderr = proc.StandardError.ReadToEndAsync();
            await proc.WaitForExitAsync();
            if (proc.ExitCode != 0) throw new InvalidOperationException($"7za failed for pack {name}: {await stderr}");
            await stdout;
        }

        var normalized = files.Select(f => f.Replace('\\', '/')).ToArray();
        // installed_bytes: what this pack occupies once extracted. The updater compares it against the
        // files on disk to decide whether an install is genuinely current, instead of trusting
        // components.json alone.
        long installedBytes = files.Sum(f => new FileInfo(Path.Combine(installDirectory, f)).Length);
        index.Add(new
        {
            name,
            dep,
            asset = Path.GetFileName(archive),
            bytes = new FileInfo(archive).Length,
            installed_bytes = installedBytes,
            files = normalized,
        });
        packedFiles.AddRange(normalized);
        Console.WriteLine($"  component-{name}.7z ({new FileInfo(archive).Length / 1048576} MB, {files.Length} files)");
    }
    File.WriteAllText(Path.Combine(packsDir, "packs.json"),
        JsonSerializer.Serialize(new { package_version = version, packs = index },
            new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"Packs written to {packsDir}");
    return packedFiles;
}

if (Directory.Exists(installDirectory)) Directory.Delete(installDirectory, true);
if (Directory.Exists(workDirectory)) Directory.Delete(workDirectory, true);
Directory.CreateDirectory(installDirectory);
Directory.CreateDirectory(workDirectory);

// The published app (self-contained win-x64) forms the install root.
var appDir = ArgValue("--app-dir") ?? throw new ArgumentException("--app-dir <published app> is required.");
Console.WriteLine($"Copying {AppName} from {appDir}...");
CopyDirectory(Path.GetFullPath(appDir), installDirectory);
// Debug symbols (SkiaSharp's alone are ~100 MB) aren't useful to users.
foreach (var pdb in Directory.GetFiles(installDirectory, "*.pdb", SearchOption.AllDirectories)) File.Delete(pdb);

await InstallSevenZip();
await InstallInferenceRuntime();
await InstallAji();
await InstallOrtDml();
InstallVcRuntime();
await InstallBackendCli();
await InstallModels();

// The custom updater ships at the install root next to the app.
var updater = ArgValue("--updater") ?? throw new ArgumentException("--updater <MangaJaNaiUpdater.exe> is required.");
File.Copy(updater, Path.Combine(installDirectory, "MangaJaNaiUpdater.exe"), true);

WriteThirdPartyNotices();

if (args.Contains("--packs"))
{
    SlimInstallTree(await EmitComponentPacks());
}

WriteVersionAndManifest();
WriteFullPackage();
Directory.Delete(workDirectory, true);
Console.WriteLine($"Built {installDirectory}");
