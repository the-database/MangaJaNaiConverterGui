// MangaJaNaiUpdater — keeps an installed MangaJaNaiConverterGui folder up to date in place.
// Ships at the install root next to MangaJaNaiConverterGui.exe. Ported from VideoJaNaiUpdater,
// without the component packs.
//
//   MangaJaNaiUpdater.exe --check        prints UPDATE_AVAILABLE <ver> | UP_TO_DATE <ver>
//   MangaJaNaiUpdater.exe --download     downloads + unpacks the update into update-staging/, prints UPDATE_READY <ver>
//   MangaJaNaiUpdater.exe --apply        waits for the app to close, applies (staged or freshly downloaded), relaunches
//                         [--no-relaunch] the app's install-on-close: no relaunch; failures go to update-error.txt
//
// An update is the latest GitHub release's full package (MangaJaNaiConverterGui-full-package-<ver>.zip,
// rooted at MangaJaNaiConverterGui-v<ver>/). Files the previous version shipped (manifest.json "files")
// that the new one no longer has are removed; manifest.json "user_preserve" paths are never touched.
// Saved settings, Python and models live in %APPDATA%\MangaJaNaiConverterGui, outside the install.
//
// Dev override: MANGAJANAI_UPDATE_URL=<url> reads the latest release from another GitHub-style
// releases/latest endpoint (a fork's, or a local test server's).

using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;
using static Downloader;

const string AppName = "MangaJaNaiConverterGui";
string apiLatest = Environment.GetEnvironmentVariable("MANGAJANAI_UPDATE_URL") is { Length: > 0 } url
    ? url
    : "https://api.github.com/repos/the-database/MangaJaNaiConverterGui/releases/latest";

// Single-file exe: BaseDirectory is the folder the exe runs from = the install root.
string installDir = AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, '/');
string localManifest = Path.Combine(installDir, "manifest.json");
string appExe = ReadManifestString(localManifest, "app_executable", $"{AppName}.exe");

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
        default:
            Console.WriteLine("Usage: MangaJaNaiUpdater.exe [--check|--download|--apply [--no-relaunch]]");
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

            Console.WriteLine("Applying update (your settings, Python backend and models are kept)...");
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

async Task<Release> GetLatestReleaseAsync()
{
    using var client = new HttpClient();
    client.DefaultRequestHeaders.UserAgent.ParseAdd("MangaJaNaiUpdater");
    string json = await client.GetStringAsync(apiLatest);
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

record Release(string Tag, List<Asset> Assets);
record Asset(string Name, string Url);
