using Avalonia.Collections;
using MangaJaNaiConverterGui.Services;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace MangaJaNaiConverterGui.Tests
{
    // A backend with these models installed and no upscaling engine (so device and model-info
    // queries return nothing).
    public sealed class FakeBackendService(string[] models) : IBackendService
    {
        public string AppStateFolder => Path.GetTempPath();
        public string AppStateFilename => "appstate2.json";
        public string AppStatePath => Path.Combine(AppStateFolder, AppStateFilename);
        public string LogsDirectory => AppStateFolder;
        public string InferenceDirectory => Path.Combine(AppStateFolder, "no-such-inference-dir");
        public string BackendExePath => Path.Combine(InferenceDirectory, "mangajanai_upscale.exe");
        public bool IsBackendInstalled => false;
        public bool IsTensorRtRtxInstalled => false;
        public bool IsTensorRtRuntimeInstalled => false;
        public string UserModelsDirectory => AppStateFolder;
        public string BundledModelsDirectory => AppStateFolder;
        public bool AreModelsInstalled => true;
        public string EngineCacheDirectory => AppStateFolder;
        public AvaloniaList<string> AllModels { get; } = new(models);
        public bool RefreshModels() => false;
        public string? ResolveModelPath(string? model) => null;
    }

    // Not an installed copy: no updates, no components.
    public sealed class FakeUpdateManagerService : IUpdateManagerService
    {
        public bool IsInstalled => false;
        public string AppVersion => "2.0.0";
        public string InstallDir => Path.GetTempPath();
        public string UpdaterPath => Path.Combine(InstallDir, "no-such-updater.exe");
        public Task<string?> CheckForUpdateAsync() => Task.FromResult<string?>(null);
        public void ApplyUpdateAndRestart() { }
        public Task<string?> DownloadUpdateAsync(Action<string>? onLine, CancellationToken ct) => Task.FromResult<string?>(null);
        public void ApplyUpdateOnExit() { }
        public string? TakeLastUpdateError() => null;
        public Task<string> GetComponentsJsonAsync() => Task.FromResult("");
        public Task<int> RunUpdaterStreamingAsync(string arguments, Action<string>? onLine, CancellationToken ct = default) => Task.FromResult(0);
    }
}
