using Avalonia.Collections;
using Splat;
using System;
using System.IO;
using System.Linq;

namespace MangaJaNaiConverterGui.Services
{
    public class BackendService : IBackendService
    {
        public const string NoModel = "No Model";

        private readonly IUpdateManagerService _updateManagerService;

        public BackendService(IUpdateManagerService? updateManagerService = null)
        {
            _updateManagerService = updateManagerService ?? Locator.Current.GetService<IUpdateManagerService>()!;
        }

        private bool Installed => _updateManagerService?.IsInstalled ?? false;

        // MANGAJANAI_STATE_DIR redirects the saved app state (workflows, settings), so test and dev
        // launches never touch a real install's state. An install keeps settings and user models in
        // %APPDATA% (where 1.x kept them too, so they carry over); a dev build next to the working
        // directory.
        public string AppStateFolder =>
            Environment.GetEnvironmentVariable("MANGAJANAI_STATE_DIR") is { Length: > 0 } stateDir
                ? stateDir
                : Installed
                    ? Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData, Environment.SpecialFolderOption.Create), @"MangaJaNaiConverterGui")
                    : Path.GetFullPath(@".");

        public string AppStateFilename => "appstate2.json";
        public string AppStatePath => Path.Join(AppStateFolder, AppStateFilename);
        public string LogsDirectory => Path.Combine(AppStateFolder, "logs");
        public string UserModelsDirectory => Path.Combine(AppStateFolder, "models");
        public string EngineCacheDirectory => Path.Combine(AppStateFolder, "engines");

        // MANGAJANAI_BACKEND_DIR points a dev build at a local inference folder.
        public string InferenceDirectory =>
            Environment.GetEnvironmentVariable("MANGAJANAI_BACKEND_DIR") is { Length: > 0 } backendDir
                ? backendDir
                : Path.Combine(AppContext.BaseDirectory, "inference");

        public string BackendExePath => Path.Combine(InferenceDirectory, "mangajanai_upscale.exe");

        public bool IsBackendInstalled => File.Exists(BackendExePath) && File.Exists(Path.Combine(InferenceDirectory, "aji.dll"));

        public bool IsTensorRtRtxInstalled => File.Exists(Path.Combine(InferenceDirectory, "aji_trtrtx.dll"));

        // trtexec.exe and cudart ship only in trt-runtime: TensorRT builds with the one, and both
        // TensorRT backends link the other.
        public bool IsTensorRtRuntimeInstalled => File.Exists(Path.Combine(InferenceDirectory, "trtexec.exe")) && HasFile(InferenceDirectory, "cudart64_*.dll");

        private static bool HasFile(string dir, string pattern)
        {
            try
            {
                return Directory.Exists(dir) && Directory.EnumerateFiles(dir, pattern).Any();
            }
            catch (IOException)
            {
                return false;
            }
        }

        public string BundledModelsDirectory => Path.GetFullPath(Path.Combine(InferenceDirectory, "..", "models"));

        public bool AreModelsInstalled => ModelFiles(BundledModelsDirectory).Any() || ModelFiles(UserModelsDirectory).Any();

        private static string[] ModelFiles(string dir)
        {
            try
            {
                return Directory.Exists(dir) ? Directory.GetFiles(dir, "*.onnx") : [];
            }
            catch (IOException)
            {
                return [];
            }
        }

        private AvaloniaList<string>? _allModels;

        public AvaloniaList<string> AllModels => _allModels ??= ScanModels();

        public bool RefreshModels()
        {
            var models = ScanModels();
            if (_allModels == null)
            {
                _allModels = models;
                return true;
            }
            // Unchanged: leave the list alone, so nothing bound to it is reset.
            if (_allModels.SequenceEqual(models))
            {
                return false;
            }
            // Update in place: every chain's model list shares this collection.
            _allModels.Clear();
            _allModels.AddRange(models);
            return true;
        }

        public string? ResolveModelPath(string? model)
        {
            if (string.IsNullOrWhiteSpace(model) || model == NoModel)
            {
                return null;
            }
            try
            {
                if (Path.IsPathRooted(model))
                {
                    return File.Exists(model) ? model : null;
                }
                foreach (var dir in new[] { UserModelsDirectory, BundledModelsDirectory })
                {
                    var path = Path.Combine(dir, model);
                    if (File.Exists(path))
                    {
                        return path;
                    }
                }
            }
            catch (ArgumentException)
            {
            }
            return null;
        }

        private AvaloniaList<string> ScanModels()
        {
            var names = ModelFiles(UserModelsDirectory).Concat(ModelFiles(BundledModelsDirectory))
                .Select(Path.GetFileName)
                .OfType<string>()
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .Order(StringComparer.OrdinalIgnoreCase)
                .ToList();
            names.Add(NoModel);
            return new AvaloniaList<string>(names);
        }
    }
}
