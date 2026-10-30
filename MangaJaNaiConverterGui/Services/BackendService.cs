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

        public void RefreshModels()
        {
            var models = ScanModels();
            if (_allModels == null)
            {
                _allModels = models;
                return;
            }
            // Update in place: every chain's model list shares this collection.
            _allModels.Clear();
            _allModels.AddRange(models);
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
