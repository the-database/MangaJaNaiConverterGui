// MangaJaNaiUpdater — keeps an installed MangaJaNaiConverterGui folder up to date in place, and
// manages its component packs (the in-app component manager and the installer shell this). Ships at
// the install root next to MangaJaNaiConverterGui.exe. Ported from VideoJaNaiUpdater.
//
//   MangaJaNaiUpdater.exe --check        prints UPDATE_AVAILABLE <ver> | UP_TO_DATE <ver>
//   MangaJaNaiUpdater.exe --download     downloads + unpacks the update into update-staging/, prints UPDATE_READY <ver>
//   MangaJaNaiUpdater.exe --apply        waits for the app to close, applies (staged or freshly downloaded), relaunches
//                         [--no-relaunch] the app's install-on-close: no relaunch; failures go to update-error.txt
//   MangaJaNaiUpdater.exe --components [--json]  detect GPU, list installed/available packs + recommendation
//   MangaJaNaiUpdater.exe --install X    download + install component pack X
//   MangaJaNaiUpdater.exe --remove X     delete component pack X's files
//   MangaJaNaiUpdater.exe --auto         install everything the detected hardware needs (or refresh stale packs)
//   MangaJaNaiUpdater.exe --recommend    KEY=value hardware summary for the Inno installer
//
// An update is the latest GitHub release's full package (MangaJaNaiConverterGui-full-package-<ver>.zip,
// rooted at MangaJaNaiConverterGui-v<ver>/). Files the previous version shipped (manifest.json "files")
// that the new one no longer has are removed; manifest.json "user_preserve" paths are never touched.
// Saved settings and user models live in %APPDATA%\MangaJaNaiConverterGui, outside the install.
//
// Component packs are subsets of the install (the TensorRT runtime, per-GPU-generation builder
// resources, the default ONNX models, the opt-in TensorRT-RTX backend), emitted by BuildMangaJaNai as
// component-<name>.7z + packs.json release assets. Archives are rooted at the install dir, so
// extraction is installation; packs.json carries each pack's file list, so removal is deletion.
// Installed state lives in components.json at the install root. Packs always come from the installed
// version's release, so they match its builds.
//
// Dev overrides: MANGAJANAI_UPDATE_URL=<url> reads the latest release from another GitHub-style
// releases/latest endpoint (a fork's, or a local test server's); MANGAJANAI_PACKS_DIR=<dir> reads
// packs.json + component-*.7z from a local directory.

using System.Diagnostics;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text.Json;
using static Downloader;

const string AppName = "MangaJaNaiConverterGui";
const string Repo = "the-database/MangaJaNaiConverterGui";
string apiLatest = Environment.GetEnvironmentVariable("MANGAJANAI_UPDATE_URL") is { Length: > 0 } url
    ? url
    : $"https://api.github.com/repos/{Repo}/releases/latest";

// Single-file exe: BaseDirectory is the folder the exe runs from = the install root.
string installDir = AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, '/');
string localManifest = Path.Combine(installDir, "manifest.json");
string appExe = ReadManifestString(localManifest, "app_executable", $"{AppName}.exe");
string archiveTool = ReadManifestString(localManifest, "archive_tool", "7za.exe");

string mode = args.Length > 0 ? args[0].ToLowerInvariant() : "--check";

// A background --download unpacks the update here (inside the install, so each install has its own),
// and --apply installs from it without downloading again. ready.json marks it complete.
string stagingRoot = Path.Combine(installDir, "update-staging");
string stagingMarker = Path.Combine(stagingRoot, "ready.json");
// An update that failed to install, for the app to show on its next start.
string updateErrorFile = Path.Combine(installDir, "update-error.txt");

try
{
    switch (mode)
    {
        case "--check":
            await CheckAsync();
            return 0;
        case "--download":
            return await DownloadAsync();
        case "--apply":
            return await ApplyAsync(args.Contains("--no-relaunch"));
        case "--components":
            await ComponentsAsync(null, args.Contains("--json"));
            return 0;
        case "--install":
            return await InstallComponentAsync(args.Length > 1 ? args[1] : "");
        case "--remove":
            return RemoveComponent(args.Length > 1 ? args[1] : "");
        case "--auto":
            return await AutoComponentsAsync();
        case "--recommend":
            return await RecommendAsync();
        default:
            Console.WriteLine("Usage: MangaJaNaiUpdater.exe [--check|--download|--apply [--no-relaunch]|--components [--json]|--install <pack>|--remove <pack>|--auto|--recommend]");
            return 2;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Updater error: {ex.Message}");
    return 1;
}

// ---- modes ---------------------------------------------------------------------------------

async Task CheckAsync()
{
    var release = await GetLatestReleaseAsync();
    string local = ReadLocalVersion();
    if (IsNewer(release.Tag, local))
    {
        Console.WriteLine($"UPDATE_AVAILABLE {release.Tag}");
    }
    else
    {
        Console.WriteLine($"UP_TO_DATE {local}");
    }
}

async Task<int> DownloadAsync()
{
    var release = await GetLatestReleaseAsync();
    string local = ReadLocalVersion();
    if (!IsNewer(release.Tag, local))
    {
        Console.WriteLine($"UP_TO_DATE {local}");
        return 0;
    }
    if (ReadStaged(release.Tag) is null)
    {
        await StageAsync(release);
    }
    Console.WriteLine($"UPDATE_READY {release.Tag}");
    return 0;
}

async Task<int> ApplyAsync(bool noRelaunch)
{
    // The app quits right after starting us; wait for it, and anything else running from the install
    // (a background --download), to release file locks.
    if (!WaitForInstallProcesses(TimeSpan.FromSeconds(60)))
    {
        if (noRelaunch)
        {
            // Another window is still open; installing on its close covers it.
            return 0;
        }
        Console.WriteLine($"Close all {AppName} windows, then try again.");
        return 1;
    }

    try
    {
        var release = await GetLatestReleaseAsync();
        string local = ReadLocalVersion();
        if (!IsNewer(release.Tag, local))
        {
            Console.WriteLine($"Already up to date (v{local}).");
            TryDelete(() => Directory.Delete(stagingRoot, true));
        }
        else
        {
            Console.WriteLine($"Updating from v{local} to v{release.Tag}...");
            string root = ReadStaged(release.Tag) ?? await StageAsync(release);

            Console.WriteLine("Applying update (your settings, models and engines are kept)...");
            ApplyOver(root, installDir);

            TryDelete(() => Directory.Delete(stagingRoot, true));
            TryDelete(() => File.Delete(updateErrorFile));
            Console.WriteLine($"Update to v{release.Tag} complete.");
        }
    }
    catch (Exception ex)
    {
        // The staged files are kept, so the next attempt doesn't download again.
        TryDelete(() => File.WriteAllText(updateErrorFile, ex.Message));
        Console.WriteLine($"Updater error: {ex.Message}");
        if (!noRelaunch)
        {
            RelaunchApp();
        }
        return 1;
    }

    if (!noRelaunch)
    {
        RelaunchApp();
    }
    return 0;
}

// Downloads and unpacks the release into update-staging/, marking it complete only at the end.
// Returns the unpacked folder to copy over the install.
async Task<string> StageAsync(Release release)
{
    TryDelete(() => Directory.Delete(stagingRoot, true));
    string work = Path.Combine(stagingRoot, "download");
    string staging = Path.Combine(stagingRoot, "tree");
    Directory.CreateDirectory(work);

    var asset = release.Assets.FirstOrDefault(a =>
            a.Name.Contains("full-package-", StringComparison.OrdinalIgnoreCase) &&
            a.Name.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
        ?? throw new InvalidOperationException("No full-package asset found on the latest release.");
    string archive = Path.Combine(work, asset.Name);
    await DownloadWithProgress(asset, archive);

    Console.WriteLine("Extracting...");
    Directory.CreateDirectory(staging);
    ZipFile.ExtractToDirectory(archive, staging, overwriteFiles: true);
    TryDelete(() => Directory.Delete(work, true));

    string sourceRoot = FindVersionedRoot(staging);
    File.WriteAllText(stagingMarker, JsonSerializer.Serialize(new
    {
        tag = release.Tag,
        root = Path.GetRelativePath(stagingRoot, sourceRoot),
    }));
    return sourceRoot;
}

// The staged update for this release, if one finished unpacking.
string? ReadStaged(string tag)
{
    try
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(stagingMarker));
        var r = doc.RootElement;
        string root = Path.Combine(stagingRoot, r.GetProperty("root").GetString() ?? "");
        return r.GetProperty("tag").GetString() == tag && Directory.Exists(root) ? root : null;
    }
    catch
    {
        return null;
    }
}

async Task DownloadWithProgress(Asset asset, string dest)
{
    double last = -10;
    Console.WriteLine($"Downloading {asset.Name}...");
    await DownloadFileAsync(asset.Url, dest, p =>
    {
        if (p >= last + 5)
        {
            Console.WriteLine($"  {asset.Name}: {p}%");
            last = p;
        }
    });
}

// The full package's root is the versioned folder.
string FindVersionedRoot(string staging)
{
    var dir = Directory.GetDirectories(staging)
        .FirstOrDefault(d => Path.GetFileName(d).StartsWith($"{AppName}-v", StringComparison.OrdinalIgnoreCase));
    return dir ?? staging;
}

// ---- apply ---------------------------------------------------------------------------------

// Copy the extracted tree over the install, then remove files the previous version shipped that the
// new one doesn't. user_preserve paths from the local manifest are never overwritten or removed.
void ApplyOver(string sourceRoot, string targetRoot)
{
    string targetManifest = Path.Combine(targetRoot, "manifest.json");
    var preserve = ReadStringSet(targetManifest, "user_preserve");
    // Read before the copy replaces manifest.json.
    var oldFiles = ReadStringSet(targetManifest, "files");
    // Kept for --install/--auto: a pack whose governing dep didn't change needs no new download.
    TryDelete(() => File.Copy(targetManifest, Path.Combine(targetRoot, "manifest.prev.json"), true));

    // Self-update: a running exe can't be overwritten, but it can be renamed out of the way first.
    string ownExe = Environment.ProcessPath ?? "";
    string ownName = string.IsNullOrEmpty(ownExe) ? "" : Path.GetFileName(ownExe);
    string newUpdater = Path.Combine(sourceRoot, ownName);

    EnsureReplaceable(sourceRoot, targetRoot, preserve, ownName);

    if (!string.IsNullOrEmpty(ownExe) && File.Exists(newUpdater) &&
        Path.GetFullPath(ownExe).Equals(Path.GetFullPath(Path.Combine(targetRoot, ownName)), StringComparison.OrdinalIgnoreCase))
    {
        string old = ownExe + ".old";
        TryDelete(() => File.Delete(old));
        TryDelete(() => File.Move(ownExe, old));
    }

    CopyTree(sourceRoot, targetRoot, sourceRoot, preserve);
    RemoveStale(sourceRoot, targetRoot, oldFiles, preserve);
}

// Fails before anything is changed when a file the update replaces is in use (another program has it
// open), so a failed install leaves the old version intact instead of half-updated.
static void EnsureReplaceable(string sourceRoot, string targetRoot, HashSet<string> preserve, string ownName)
{
    foreach (var file in Directory.EnumerateFiles(sourceRoot, "*", SearchOption.AllDirectories))
    {
        string rel = NormalizeRel(Path.GetRelativePath(sourceRoot, file));
        string dst = Path.Combine(targetRoot, rel);
        if (IsPreserved(rel, preserve) || rel.Equals(ownName, StringComparison.OrdinalIgnoreCase) || !File.Exists(dst))
        {
            continue;
        }
        try
        {
            using var _ = new FileStream(dst, FileMode.Open, FileAccess.ReadWrite,
                FileShare.ReadWrite | FileShare.Delete);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            throw new IOException($"{rel} is in use by another program. Close it and try again.", e);
        }
    }
}

void CopyTree(string srcDir, string dstDir, string sourceRoot, HashSet<string> preserve)
{
    Directory.CreateDirectory(dstDir);

    foreach (var file in Directory.GetFiles(srcDir))
    {
        string rel = NormalizeRel(Path.GetRelativePath(sourceRoot, file));
        if (IsPreserved(rel, preserve))
        {
            continue;
        }
        string dst = Path.Combine(dstDir, Path.GetFileName(file));
        TryDelete(() => { if (File.Exists(dst)) File.SetAttributes(dst, FileAttributes.Normal); });
        File.Copy(file, dst, true);
    }

    foreach (var sub in Directory.GetDirectories(srcDir))
    {
        string rel = NormalizeRel(Path.GetRelativePath(sourceRoot, sub));
        if (IsPreserved(rel, preserve))
        {
            continue;
        }
        CopyTree(sub, Path.Combine(dstDir, Path.GetFileName(sub)), sourceRoot, preserve);
    }
}

// Only files the old manifest lists are candidates, so anything the user added is left alone.
static void RemoveStale(string sourceRoot, string targetRoot, HashSet<string> oldFiles, HashSet<string> preserve)
{
    foreach (var rel in oldFiles)
    {
        if (IsPreserved(rel, preserve) || File.Exists(Path.Combine(sourceRoot, rel)))
        {
            continue;
        }
        string dst = Path.Combine(targetRoot, rel);
        TryDelete(() => File.Delete(dst));
        // Drop folders the removal left empty, up to the install root.
        for (var dir = Path.GetDirectoryName(dst);
             dir != null && !Path.GetFullPath(dir).Equals(Path.GetFullPath(targetRoot), StringComparison.OrdinalIgnoreCase);
             dir = Path.GetDirectoryName(dir))
        {
            if (!Directory.Exists(dir) || Directory.EnumerateFileSystemEntries(dir).Any())
            {
                break;
            }
            TryDelete(() => Directory.Delete(dir));
        }
    }
}

static bool IsPreserved(string rel, HashSet<string> preserve)
{
    foreach (var p in preserve)
    {
        if (rel.Equals(p, StringComparison.OrdinalIgnoreCase) || rel.StartsWith(p + "/", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }
    }
    return false;
}

static string NormalizeRel(string rel) => rel.Replace('\\', '/');

// ---- helpers -------------------------------------------------------------------------------

static string ReadManifestString(string manifestPath, string key, string fallback)
{
    try
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
        if (doc.RootElement.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String)
        {
            var s = v.GetString();
            if (!string.IsNullOrEmpty(s)) return s;
        }
    }
    catch { /* missing/unreadable manifest -> fallback */ }
    return fallback;
}

static HashSet<string> ReadStringSet(string manifestPath, string key)
{
    var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    try
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
        if (doc.RootElement.TryGetProperty(key, out var arr))
        {
            foreach (var e in arr.EnumerateArray())
            {
                var v = e.GetString();
                if (!string.IsNullOrEmpty(v)) set.Add(NormalizeRel(v));
            }
        }
    }
    catch { /* unreadable -> empty: nothing preserved or pruned */ }
    return set;
}

string ReadLocalVersion()
{
    string path = Path.Combine(installDir, "version.txt");
    return File.Exists(path) ? File.ReadAllText(path).Trim() : "0.0.0";
}

Task<Release> GetLatestReleaseAsync() => GetReleaseAsync(apiLatest);

// Component packs must match this installation's builds, so they come from the installed version's
// release (including prereleases), not the latest one.
Task<Release> GetInstalledReleaseAsync()
{
    string tag = ReadLocalVersion();
    if (string.IsNullOrWhiteSpace(tag) || tag == "0.0.0")
    {
        throw new InvalidOperationException("Cannot select component packs: the installed version is unknown.");
    }
    return GetReleaseAsync($"https://api.github.com/repos/{Repo}/releases/tags/{Uri.EscapeDataString(tag)}");
}

static HttpClient NewClient()
{
    var client = new HttpClient();
    client.DefaultRequestHeaders.UserAgent.ParseAdd("MangaJaNaiUpdater");
    return client;
}

async Task<Release> GetReleaseAsync(string url)
{
    using var client = NewClient();
    string json = await client.GetStringAsync(url);
    using var doc = JsonDocument.Parse(json);
    var root = doc.RootElement;
    string tag = root.GetProperty("tag_name").GetString() ?? "";
    var assets = new List<Asset>();
    if (root.TryGetProperty("assets", out var arr))
    {
        foreach (var a in arr.EnumerateArray())
        {
            assets.Add(new Asset(
                a.GetProperty("name").GetString() ?? "",
                a.GetProperty("browser_download_url").GetString() ?? ""));
        }
    }
    return new Release(tag, assets);
}

void RelaunchApp()
{
    string app = Path.Combine(installDir, appExe);
    if (!File.Exists(app))
    {
        return;
    }
    try
    {
        Process.Start(new ProcessStartInfo { FileName = app, UseShellExecute = true, WorkingDirectory = installDir });
    }
    catch (Exception e) { Console.WriteLine($"Could not relaunch {AppName}: {e.Message}"); }
}

// Waits until no other process runs from the install folder; false if some still do at the timeout.
bool WaitForInstallProcesses(TimeSpan timeout)
{
    string root = Path.GetFullPath(installDir) + Path.DirectorySeparatorChar;
    int self = Environment.ProcessId;
    var sw = Stopwatch.StartNew();
    while (true)
    {
        bool busy = Process.GetProcesses().Any(p =>
        {
            try
            {
                return p.Id != self && p.MainModule?.FileName is string f &&
                       Path.GetFullPath(f).StartsWith(root, StringComparison.OrdinalIgnoreCase);
            }
            catch { return false; } // exited, or not ours to inspect
            finally { p.Dispose(); }
        });
        if (!busy)
        {
            return true;
        }
        if (sw.Elapsed >= timeout)
        {
            return false;
        }
        Thread.Sleep(500);
    }
}

static void TryDelete(Action action)
{
    try { action(); } catch { /* best effort */ }
}

// Numeric-dotted semver compare; "3.2.10" > "3.2.9". Non-numeric parts count as 0.
static bool IsNewer(string remote, string local)
{
    int[] R = Parse(remote), L = Parse(local);
    for (int i = 0; i < Math.Max(R.Length, L.Length); i++)
    {
        int r = i < R.Length ? R[i] : 0;
        int l = i < L.Length ? L[i] : 0;
        if (r != l) return r > l;
    }
    return false;

    static int[] Parse(string v)
    {
        var parts = v.TrimStart('v', 'V').Split('.');
        var nums = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            nums[i] = int.TryParse(new string(parts[i].TakeWhile(char.IsDigit).ToArray()), out var n) ? n : 0;
        }
        return nums;
    }
}

// ---- component manager ----------------------------------------------------------------------

async Task<PackIndex> GetPackIndexAsync()
{
    string? local = Environment.GetEnvironmentVariable("MANGAJANAI_PACKS_DIR");
    string json;
    List<Asset> assets;
    if (!string.IsNullOrEmpty(local))
    {
        json = File.ReadAllText(Path.Combine(local, "packs.json"));
        assets = Directory.GetFiles(local, "component-*.7z")
            .Select(f => new Asset(Path.GetFileName(f), f)).ToList();
    }
    else
    {
        var release = await GetInstalledReleaseAsync();
        var idx = release.Assets.FirstOrDefault(a => a.Name == "packs.json")
            ?? throw new InvalidOperationException($"Release {release.Tag} publishes no component packs (packs.json missing).");
        using var client = NewClient();
        json = await client.GetStringAsync(idx.Url);
        assets = release.Assets;
    }

    var packs = new List<Pack>();
    using var doc = JsonDocument.Parse(json);
    foreach (var e in doc.RootElement.GetProperty("packs").EnumerateArray())
    {
        var files = e.GetProperty("files").EnumerateArray()
            .Select(f => f.GetString() ?? "").Where(f => f.Length > 0).ToList();
        string asset = e.GetProperty("asset").GetString() ?? "";
        packs.Add(new Pack(
            e.GetProperty("name").GetString() ?? "",
            asset,
            assets.FirstOrDefault(a => a.Name == asset)?.Url,
            e.GetProperty("bytes").GetInt64(),
            files,
            e.TryGetProperty("dep", out var d) ? d.GetString() : null,
            e.TryGetProperty("installed_bytes", out var ib) ? ib.GetInt64() : 0));
    }
    return new PackIndex(
        doc.RootElement.TryGetProperty("package_version", out var v) ? v.GetString() ?? "" : "", packs);
}

// One dependency version out of a manifest's deps (e.g. "inference_runtime", "models"). "" if the
// manifest/key is missing - used to decide whether an already-present pack's content is unchanged.
static string ReadManifestDep(string manifestPath, string depKey)
{
    try
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
        if (doc.RootElement.TryGetProperty("deps", out var deps) &&
            deps.TryGetProperty(depKey, out var v) && v.ValueKind == JsonValueKind.String)
        {
            return v.GetString() ?? "";
        }
    }
    catch { /* missing/unreadable manifest -> "" */ }
    return "";
}

// Does the pack's content on disk actually match the published pack? components.json records only
// what the updater believes it installed; a different upstream build is a different total size, so
// comparing sizes catches a stale runtime the record still calls current.
bool PackContentMatches(Pack pack)
{
    if (pack.InstalledBytes <= 0)
    {
        return true;
    }
    long actual = 0;
    foreach (var f in pack.Files)
    {
        var fi = new FileInfo(Path.Combine(installDir, f));
        if (!fi.Exists)
        {
            return false;
        }
        actual += fi.Length;
    }
    if (actual != pack.InstalledBytes)
    {
        Console.WriteLine($"{pack.Name}: on-disk content does not match the published pack " +
                          $"({actual:N0} vs {pack.InstalledBytes:N0} bytes); reinstalling.");
        return false;
    }
    return true;
}

// Installed state: components.json, else inferred from what's on disk.
Dictionary<string, string> ReadInstalledComponents(PackIndex index)
{
    string path = Path.Combine(installDir, "components.json");
    var installed = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    if (File.Exists(path))
    {
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            foreach (var e in doc.RootElement.GetProperty("installed").EnumerateObject())
            {
                installed[e.Name] = e.Value.GetString() ?? "";
            }
            return installed;
        }
        catch { /* fall through to inference */ }
    }
    foreach (var pack in index.Packs)
    {
        if (pack.Files.Count > 0 && pack.Files.All(f => File.Exists(Path.Combine(installDir, f))))
        {
            installed[pack.Name] = "(found on disk)";
        }
    }
    return installed;
}

void WriteInstalledComponents(Dictionary<string, string> installed)
{
    File.WriteAllText(Path.Combine(installDir, "components.json"),
        JsonSerializer.Serialize(new { installed }, new JsonSerializerOptions { WriteIndented = true }));
}

string? PackVersionMismatch(PackIndex index)
{
    string localVersion = ReadLocalVersion();
    return localVersion != "" && index.PackageVersion != "" && localVersion != index.PackageVersion
        ? $"This installation is v{localVersion} but the published packs are for v{index.PackageVersion}."
        : null;
}

// NVIDIA detection via NVML (ships with the driver); its absence means a non-NVIDIA GPU, which runs
// the built-in DirectML engine and needs no TensorRT packs. Every NVIDIA GPU is reported: NVML
// enumerates in PCI order while CUDA picks the fastest GPU first, so recommending only device 0 can
// leave the GPU TensorRT builds on without its builder kernels on multi-GPU systems.
static (bool HasNvidia, string Sm, string GpuName) DetectGpu()
{
    try
    {
        if (Nvml.nvmlInit_v2() != 0)
        {
            return (false, "", "");
        }
        try
        {
            if (Nvml.nvmlDeviceGetCount_v2(out uint count) != 0 || count == 0)
            {
                return (false, "", "");
            }
            var sms = new List<string>();
            var names = new List<string>();
            for (uint i = 0; i < count; i++)
            {
                if (Nvml.nvmlDeviceGetHandleByIndex_v2(i, out var dev) != 0)
                {
                    continue;
                }
                Nvml.nvmlDeviceGetCudaComputeCapability(dev, out int major, out int minor);
                var name = new byte[96];
                Nvml.nvmlDeviceGetName(dev, name, (uint)name.Length);
                names.Add(System.Text.Encoding.ASCII.GetString(name).TrimEnd('\0'));
                var sm = $"sm{major}{minor}";
                if (!sms.Contains(sm))
                {
                    sms.Add(sm);
                }
            }
            return sms.Count == 0 ? (false, "", "") : (true, string.Join(", ", sms), string.Join(" + ", names));
        }
        finally { Nvml.nvmlShutdown(); }
    }
    catch
    {
        return (false, "", "");
    }
}

// The models every install needs, plus the TensorRT runtime and the builder pack for each NVIDIA GPU
// generation (the exact pack if published, else the PTX fallback). Never trt-rtx: the TensorRT-RTX
// backend is experimental and opt-in, so only the user installs it (App Settings > Components). Once
// installed, PreselectPack keeps it in --auto, which refreshes it when its dep (aji) moves.
List<string> RecommendedPacks(PackIndex index, bool hasNvidia, string sm)
{
    var rec = index.Packs.Where(p => p.Name.StartsWith("models-", StringComparison.Ordinal)).Select(p => p.Name).ToList();
    if (hasNvidia)
    {
        rec.Add("trt-runtime");
        foreach (var s in sm.Split(", ", StringSplitOptions.RemoveEmptyEntries))
        {
            var pack = index.Packs.Any(p => p.Name == $"trt-{s}") ? $"trt-{s}" : "trt-ptx";
            if (!rec.Contains(pack))
            {
                rec.Add(pack);
            }
        }
    }
    return rec;
}

bool PreselectPack(Pack pack, Dictionary<string, string> installed, List<string> rec) =>
    installed.ContainsKey(pack.Name) || rec.Contains(pack.Name);

async Task ComponentsAsync(PackIndex? prefetched, bool json = false)
{
    var index = prefetched ?? await GetPackIndexAsync();
    var installed = ReadInstalledComponents(index);
    var (hasNvidia, sm, gpu) = DetectGpu();
    var rec = RecommendedPacks(index, hasNvidia, sm);
    if (json)
    {
        // consumed by the in-app component manager; keep keys stable
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            package_version = index.PackageVersion,
            version_mismatch = PackVersionMismatch(index),
            gpu = new { nvidia = hasNvidia, sm, name = gpu },
            packs = index.Packs.Select(p => new
            {
                name = p.Name,
                bytes = p.Bytes,
                installed = installed.ContainsKey(p.Name),
                recommended = rec.Contains(p.Name),
                preselect = PreselectPack(p, installed, rec),
            }),
        }));
        return;
    }
    if (PackVersionMismatch(index) is string warn)
    {
        Console.WriteLine(warn);
        Console.WriteLine();
    }
    Console.WriteLine(hasNvidia
        ? $"GPU: {gpu} ({sm}) - TensorRT recommended"
        : "GPU: no NVIDIA device detected - using the built-in DirectML engine (no TensorRT components needed)");
    Console.WriteLine($"Recommended packs: {string.Join(", ", rec)}");
    Console.WriteLine();
    foreach (var pack in index.Packs)
    {
        string state = installed.ContainsKey(pack.Name) ? "installed" : rec.Contains(pack.Name) ? "RECOMMENDED" : "available";
        Console.WriteLine($"  {pack.Name,-26} {pack.Bytes / 1048576,6} MB  {state}");
    }
}

async Task<int> InstallComponentAsync(string name)
{
    if (string.IsNullOrEmpty(name))
    {
        Console.WriteLine("--install needs a pack name (see --components)");
        return 2;
    }
    var index = await GetPackIndexAsync();
    var pack = index.Packs.FirstOrDefault(p => p.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
    if (pack is null)
    {
        Console.WriteLine($"Unknown pack '{name}'. Available: {string.Join(", ", index.Packs.Select(p => p.Name))}");
        return 2;
    }
    if (pack.Url is null)
    {
        Console.WriteLine($"Pack '{name}' has no downloadable asset on the installed version's release.");
        return 1;
    }
    if (PackVersionMismatch(index) is string warn)
    {
        // a pack from a different release can mismatch the installed builds
        Console.WriteLine(warn);
        Console.WriteLine("Update first (MangaJaNaiUpdater --apply), then install components.");
        return 1;
    }

    // Skip the (large) download when the identical content is already on disk: every file is present,
    // the total size matches the index, AND the dep that governs it is unchanged - vs. the pre-update
    // manifest snapshot or vs. the dep version components.json recorded on a prior install.
    string target = pack.Dep is null ? "" : ReadManifestDep(localManifest, pack.Dep);
    bool filesPresent = pack.Files.Count > 0 && pack.Files.All(f => File.Exists(Path.Combine(installDir, f)));
    if (filesPresent && target != "" && PackContentMatches(pack))
    {
        string prev = pack.Dep is null ? "" : ReadManifestDep(Path.Combine(installDir, "manifest.prev.json"), pack.Dep);
        var current = ReadInstalledComponents(index);
        current.TryGetValue(pack.Name, out var recorded);
        if (prev == target || recorded == target)
        {
            current[pack.Name] = target;
            WriteInstalledComponents(current);
            Console.WriteLine($"{pack.Name} already up to date ({target}); skipping download.");
            return 0;
        }
    }

    string work = Path.Combine(Path.GetTempPath(), "mangajanai-packs");
    Directory.CreateDirectory(work);
    string archive = Path.Combine(work, pack.Asset);
    if (pack.Url.Contains("://"))
    {
        await DownloadWithProgress(new Asset(pack.Asset, pack.Url), archive);
    }
    else
    {
        File.Copy(pack.Url, archive, true); // MANGAJANAI_PACKS_DIR dev path
    }
    Console.WriteLine($"Installing {pack.Name}...");
    ExtractPack(archive, installDir);
    TryDelete(() => File.Delete(archive));

    var installed = ReadInstalledComponents(index);
    // Record the dep version (not the package version) so a later reinstall can tell whether this
    // pack's content actually changed; fall back to the package version for an untagged pack.
    installed[pack.Name] = target != "" ? target : index.PackageVersion;
    WriteInstalledComponents(installed);
    Console.WriteLine($"{pack.Name} installed.");
    return 0;
}

int RemoveComponent(string name)
{
    var index = GetPackIndexAsync().GetAwaiter().GetResult();
    var pack = index.Packs.FirstOrDefault(p => p.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
    if (pack is null)
    {
        Console.WriteLine($"Unknown pack '{name}'. Available: {string.Join(", ", index.Packs.Select(p => p.Name))}");
        return 2;
    }
    int gone = 0;
    foreach (var f in pack.Files)
    {
        string abs = Path.Combine(installDir, f);
        if (File.Exists(abs))
        {
            TryDelete(() => { File.SetAttributes(abs, FileAttributes.Normal); File.Delete(abs); });
            gone++;
        }
    }
    var installed = ReadInstalledComponents(index);
    installed.Remove(pack.Name);
    WriteInstalledComponents(installed);
    Console.WriteLine($"{pack.Name} removed ({gone} files). Engine caches and models you added are untouched.");
    return 0;
}

async Task<int> AutoComponentsAsync()
{
    var index = await GetPackIndexAsync();
    var (hasNvidia, sm, gpu) = DetectGpu();
    var rec = RecommendedPacks(index, hasNvidia, sm);
    var installed = ReadInstalledComponents(index);
    // "Needs installing" is not just "absent": a recorded pack is stale when the dep that governs it
    // moved (a TensorRT or model bump) or its files on disk are not this pack's content.
    var missing = index.Packs
        .Where(p => PreselectPack(p, installed, rec))
        .Where(p => !installed.TryGetValue(p.Name, out var recorded) ||
                    (p.Dep is not null && ReadManifestDep(localManifest, p.Dep) is string t && t != "" && recorded != t) ||
                    !PackContentMatches(p))
        .Select(p => p.Name).ToList();
    if (missing.Count == 0)
    {
        Console.WriteLine("Everything the detected hardware needs is already installed.");
        await ComponentsAsync(index, false);
        return 0;
    }
    Console.WriteLine($"Installing for {(hasNvidia ? gpu : "this GPU")}: {string.Join(", ", missing)}");
    foreach (var name in missing)
    {
        int rc = await InstallComponentAsync(name);
        if (rc != 0)
        {
            return rc;
        }
    }
    return 0;
}

// Hardware summary for the Inno installer's component picker, as KEY=value lines. GPU identity is
// local NVML and always available; PACKS needs the pack index - if that fails it is emitted empty and
// the installer leaves component setup to the app's first launch.
async Task<int> RecommendAsync()
{
    var (hasNvidia, sm, gpu) = DetectGpu();
    Console.WriteLine($"NVIDIA={(hasNvidia ? 1 : 0)}");
    Console.WriteLine($"GPU={gpu}");
    string packs = "";
    try
    {
        var index = await GetPackIndexAsync();
        packs = string.Join(",", RecommendedPacks(index, hasNvidia, sm));
    }
    catch
    {
        // offline / no published packs yet
    }
    Console.WriteLine($"PACKS={packs}");
    return 0;
}

// Component packs are 7z archives rooted at the install dir; the bundled 7-Zip console extracts them.
void ExtractPack(string archive, string outDir)
{
    var psi = new ProcessStartInfo
    {
        FileName = Path.Combine(installDir, archiveTool),
        Arguments = $"x \"{archive}\" -o\"{outDir}\" -y",
        UseShellExecute = false,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        CreateNoWindow = true,
    };
    using var p = Process.Start(psi) ?? throw new InvalidOperationException($"Failed to start {archiveTool}");
    var stdout = p.StandardOutput.ReadToEndAsync();
    string err = p.StandardError.ReadToEnd();
    p.WaitForExit();
    stdout.Wait();
    if (p.ExitCode != 0)
    {
        throw new InvalidOperationException($"Extracting {Path.GetFileName(archive)} failed (exit {p.ExitCode}): {err}");
    }
}

static class Nvml
{
    [DllImport("nvml.dll")]
    public static extern int nvmlInit_v2();
    [DllImport("nvml.dll")]
    public static extern int nvmlShutdown();
    [DllImport("nvml.dll")]
    public static extern int nvmlDeviceGetCount_v2(out uint deviceCount);
    [DllImport("nvml.dll")]
    public static extern int nvmlDeviceGetHandleByIndex_v2(uint index, out IntPtr device);
    [DllImport("nvml.dll")]
    public static extern int nvmlDeviceGetCudaComputeCapability(IntPtr device, out int major, out int minor);
    [DllImport("nvml.dll")]
    public static extern int nvmlDeviceGetName(IntPtr device, byte[] name, uint length);
}

record Release(string Tag, List<Asset> Assets);
record Asset(string Name, string Url);
// Dep: the manifest deps key that governs the pack's content. InstalledBytes: the pack's total size
// once extracted, to verify what is on disk really is this pack.
record Pack(string Name, string Asset, string? Url, long Bytes, List<string> Files, string? Dep, long InstalledBytes = 0);
record PackIndex(string PackageVersion, List<Pack> Packs);
